"""Latency on a large synthetic archive.

Seeds the demo archive, bulk-loads N extra fabricated messages with COPY,
embeds everything through the normal pipeline, then times the agent-facing
calls. Use a scratch database: it resets the public schema.

    DATABASE_URL=... uv run python scripts/scale_bench.py --messages 250000
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from datetime import UTC, datetime, timedelta

from beeper_pg_sync.db import connect
from inbox_recall import demo
from inbox_recall.embedders import HashEmbedder
from inbox_recall.eval import CASES
from inbox_recall.pipeline import EmbedPipeline
from inbox_recall.query import SearchRequest
from inbox_recall.schema import migrate
from inbox_recall.search import Searcher

WORDS = (
    "coffee dinner tomorrow tonight weekend train flight hotel concert tickets birthday party movie game "
    "work meeting deadline project apartment rent groceries gym run hike beach dog cat kids school doctor "
    "car bike bus traffic rain snow sunny photos video playlist song book podcast recipe pasta tacos pizza "
    "sushi wine beer brunch lunch call later soon maybe definitely sorry thanks congrats love miss laugh"
).split()


def timed(fn, runs: int) -> list[float]:
    out = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    now = datetime.now(UTC)
    rng = random.Random(11)
    with connect() as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        migrate(conn)
        demo.seed(conn, now, filler=0)
        chats = [r["id"] for r in conn.execute("SELECT id FROM conversations")]
        people = [r["id"] for r in conn.execute("SELECT id FROM participants WHERE NOT is_self")]

        started = time.perf_counter()
        start = now - timedelta(days=8 * 365)
        span = int((now - start).total_seconds())
        with (
            conn.cursor() as cur,
            cur.copy("COPY messages (conversation_id, source, sender_id, direction, sent_at, body) FROM STDIN") as copy,
        ):
            for _ in range(args.messages):
                outgoing = rng.random() < 0.4
                body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 14)))
                copy.write_row(
                    (
                        rng.choice(chats),
                        "scale",
                        None if outgoing else rng.choice(people),
                        "outgoing" if outgoing else "incoming",
                        start + timedelta(seconds=rng.randrange(span)),
                        body,
                    )
                )
        conn.execute("ANALYZE")
        load = time.perf_counter() - started

        embedder = HashEmbedder(256)
        stats = EmbedPipeline(conn, embedder, batch_size=500).run()
        conn.execute("VACUUM ANALYZE")  # settle autovacuum before timing
        conn.execute("CHECKPOINT")
        total = conn.execute("SELECT count(*) AS n FROM messages").fetchone()["n"]
        print(f"archive: {total:,} messages (bulk load {load:.1f}s)")
        print(f"embedding: {stats.embedded:,} messages in {stats.seconds:.1f}s with {embedder.name}")

        searcher = Searcher(conn, embedder, now=now)
        anchor = conn.execute("SELECT id FROM messages ORDER BY random() LIMIT 1").fetchone()["id"]
        calls = {
            "search (hybrid, no filters)": [
                lambda c=c: searcher.search(SearchRequest(query=c.query)) for c in CASES if not c.filters
            ],
            "search (hybrid + filters)": [
                lambda c=c: searcher.search(SearchRequest(query=c.query, **c.filters)) for c in CASES if c.filters
            ],
            "search (keyword only)": [
                lambda c=c: searcher.search(SearchRequest(query=c.query, mode="keyword")) for c in CASES
            ],
            "browse (filters, no query)": [
                lambda: searcher.search(SearchRequest(sender=["maya"], when="last year", limit=50)),
                lambda: searcher.search(SearchRequest(chat=["Book Club"], limit=50)),
            ],
            "get_conversation": [lambda: searcher.conversation(message_id=anchor, before=20, after=20)],
        }
        for fn_list in calls.values():  # warm caches once
            for fn in fn_list:
                fn()
        print()
        print("| call | p50 ms | p95 ms |")
        print("| --- | ---: | ---: |")
        for name, fns in calls.items():
            samples = [ms for fn in fns for ms in timed(fn, args.runs)]
            samples.sort()
            p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
            print(f"| {name} | {statistics.median(samples):.0f} | {p95:.0f} |")


if __name__ == "__main__":
    main()
