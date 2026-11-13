"""Embedding pipeline against real Postgres + pgvector."""

import threading
import time
from datetime import timedelta

import pytest
from conftest import NOW

from beeper_pg_sync.model import Batch, Conversation, Message, Participant
from beeper_pg_sync.store import Store
from inbox_recall import demo
from inbox_recall.embedders import HashEmbedder
from inbox_recall.pipeline import EmbedPipeline
from inbox_recall.schema import ensure_model, index_name


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()["n"]


def test_migrations_apply_on_top_of_the_core_schema(conn):
    versions = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions == ["0001_init", "1000_search"]
    exts = {r["extname"] for r in conn.execute("SELECT extname FROM pg_extension")}
    assert {"vector", "pg_trgm"} <= exts


def test_embeds_everything_once_then_only_changes(conn, embedder):
    demo.seed(conn, NOW, filler=50)
    pipeline = EmbedPipeline(conn, embedder, batch_size=16)
    total = count(conn, "SELECT count(*) AS n FROM messages WHERE body <> '' AND NOT deleted")
    assert pipeline.pending() == total

    first = pipeline.run()
    assert first.embedded == total and first.batches == -(-total // 16)
    assert pipeline.pending() == 0
    assert pipeline.run().embedded == 0  # nothing new: no work

    target = demo.message_ids(conn, ["wa-maya:lisbon-5"])["wa-maya:lisbon-5"]
    before = conn.execute("SELECT embedding::text AS v FROM message_embeddings WHERE message_id = %s", (target,))
    old_vector = before.fetchone()["v"]
    conn.execute("UPDATE messages SET body = 'changed plans: staying in Porto instead' WHERE id = %s", (target,))
    assert pipeline.pending() == 1
    assert pipeline.run().embedded == 1
    new_vector = conn.execute(
        "SELECT embedding::text AS v FROM message_embeddings WHERE message_id = %s", (target,)
    ).fetchone()["v"]
    assert new_vector != old_vector

    conn.execute("UPDATE messages SET deleted = true WHERE id = %s", (target,))
    stats = pipeline.run()
    assert stats.removed == 1 and stats.embedded == 0
    assert count(conn, "SELECT count(*) AS n FROM message_embeddings WHERE message_id = %s", target) == 0

    state = Store(conn).get_state("inbox-recall", f"embed:{embedder.name}")
    assert state["status"] == "ok" and state["detail"]["model"] == embedder.name


def test_interrupted_runs_resume_where_they_stopped(conn, embedder):
    demo.seed(conn, NOW, filler=30)
    pipeline = EmbedPipeline(conn, embedder, batch_size=7)
    total = pipeline.pending()
    assert pipeline.run(limit=10).embedded == 10
    assert pipeline.pending() == total - 10
    assert pipeline.run().embedded == total - 10


def test_failed_batches_leave_earlier_batches_committed(conn):
    demo.seed(conn, NOW, filler=30)

    class Flaky(HashEmbedder):
        calls = 0

        def embed(self, texts):
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise RuntimeError("provider went away")
            return super().embed(texts)

    pipeline = EmbedPipeline(conn, Flaky(64), batch_size=5)
    total = pipeline.pending()
    with pytest.raises(RuntimeError):
        pipeline.run()
    assert count(conn, "SELECT count(*) AS n FROM message_embeddings") == 10
    assert pipeline.run().embedded == total - 10
    assert pipeline.pending() == 0


def test_each_model_gets_its_own_hnsw_index_and_rejects_dimension_changes(conn):
    demo.seed(conn, NOW, filler=0)
    EmbedPipeline(conn, HashEmbedder(64)).run()
    EmbedPipeline(conn, HashEmbedder(32)).run()
    indexes = {
        r["indexname"] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'message_embeddings'")
    }
    assert {index_name("hash-v1@64"), index_name("hash-v1@32")} <= indexes
    with pytest.raises(ValueError, match="registered with 64"):
        ensure_model(conn, "hash-v1@64", "hash", 128)

    conn.execute("SET enable_seqscan = off")
    vector = "[" + ",".join(["0.125"] * 64) + "]"
    plan = "\n".join(
        r["QUERY PLAN"]
        for r in conn.execute(
            "EXPLAIN SELECT message_id FROM message_embeddings WHERE model = 'hash-v1@64' "
            "ORDER BY embedding::vector(64) <=> %s::vector(64) LIMIT 5",
            (vector,),
        )
    )
    conn.execute("RESET enable_seqscan")
    assert index_name("hash-v1@64") in plan


def test_context_comes_from_earlier_messages_in_the_same_chat(conn, embedder):
    demo.seed(conn, NOW, filler=0)
    seen = []

    class Recording(HashEmbedder):
        def embed(self, texts):
            seen.extend(texts)
            return super().embed(texts)

    EmbedPipeline(conn, Recording(64), context_messages=2).run()
    reply = next(t for t in seen if t.endswith("me: omg yes book it before someone else does"))
    assert reply.splitlines() == [
        "whatsapp chat: Maya Chen",
        "Maya Chen: found a place in Alfama, the one with the blue tiled balcony and the view of the river",
        "me: omg yes book it before someone else does",
    ]  # the message 3 days earlier is outside the 6 hour context window


def test_watch_mode_wakes_on_new_messages(conn, dsn, embedder):
    demo.seed(conn, NOW, filler=0)
    pipeline = EmbedPipeline(conn, embedder)
    stop = threading.Event()
    worker = threading.Thread(target=pipeline.watch, args=(dsn,), kwargs={"interval": 60, "stop": stop.is_set})

    from beeper_pg_sync.db import connect

    with connect(dsn) as watch_conn:
        pipeline.conn = watch_conn
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while pipeline.pending() and time.monotonic() < deadline:
                time.sleep(0.1)
            Store(conn).ingest(
                Batch(
                    source="demo",
                    conversation=Conversation(source_conversation_id="wa-maya", network="whatsapp", title="Maya Chen"),
                    messages=[
                        Message(
                            source_message_id="wa-maya:new-1",
                            body="landing at 7:50, see you at arrivals",
                            sent_at=NOW - timedelta(minutes=5),
                            direction="incoming",
                            sender=Participant(native_id="15550100201", display_name="Maya Chen"),
                        )
                    ],
                )
            )
            new_id = demo.message_ids(conn, ["wa-maya:new-1"])["wa-maya:new-1"]
            while time.monotonic() < deadline:
                if count(conn, "SELECT count(*) AS n FROM message_embeddings WHERE message_id = %s", new_id):
                    break
                time.sleep(0.1)
            assert count(conn, "SELECT count(*) AS n FROM message_embeddings WHERE message_id = %s", new_id) == 1
        finally:
            stop.set()
            worker.join(timeout=10)
