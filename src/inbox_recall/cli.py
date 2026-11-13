"""inbox-recall command line."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from beeper_pg_sync.db import connect
from beeper_pg_sync.store import Store

from . import demo
from . import eval as bench
from .config import RecallSettings
from .embedders import from_settings
from .mcp import McpServer, serve_stdio
from .pipeline import EmbedPipeline
from .query import parse_operators
from .schema import drop_model, migrate
from .search import Searcher
from .tools import BY_NAME

log = logging.getLogger("inbox_recall")


def searcher_factory(settings: RecallSettings):
    embedder = from_settings(settings)

    @contextmanager
    def open_searcher() -> Iterator[Searcher]:
        with connect(settings.database_url) as conn:
            yield Searcher(conn, embedder, tz=settings.timezone)

    return open_searcher


def _print(tool: str, result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, default=str, indent=2, ensure_ascii=False))
    else:
        print(BY_NAME[tool].text(result))


def cmd_migrate(settings: RecallSettings, args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        applied = migrate(conn)
    print("applied: " + ", ".join(applied) if applied else "schema is up to date")
    return 0


def _pipeline(conn, settings: RecallSettings) -> EmbedPipeline:
    embedder = from_settings(settings)
    if not embedder:
        raise RuntimeError("set EMBED_PROVIDER (openai, ollama, lmstudio, sentence-transformers or hash) to embed")
    return EmbedPipeline(
        conn, embedder, batch_size=settings.embed_batch_size, context_messages=settings.embed_context_messages
    )


def cmd_embed(settings: RecallSettings, args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        if args.drop_model:
            print("dropped" if drop_model(conn, args.drop_model) else "no such model")
            return 0
        pipeline = _pipeline(conn, settings)
        if args.status:
            print(json.dumps({"model": pipeline.model, "pending": pipeline.pending()}))
            return 0
        if args.watch:
            stop = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            try:
                pipeline.watch(settings.database_url or "", interval=args.interval, stop=stop.is_set)
            except KeyboardInterrupt:
                pass
            return 0
        total = pipeline.pending() if args.limit is None else min(args.limit, pipeline.pending())

        def progress(stats) -> None:
            if sys.stderr.isatty():
                print(f"\r{pipeline.model}: {stats.embedded}/{total}", end="", file=sys.stderr, flush=True)

        stats = pipeline.run(limit=args.limit, progress=progress)
        if sys.stderr.isatty() and stats.embedded:
            print(file=sys.stderr)
        print(json.dumps(stats.as_dict()))
    return 0


def cmd_search(settings: RecallSettings, args: argparse.Namespace) -> int:
    text, inline = parse_operators(" ".join(args.query))
    arguments: dict[str, Any] = {
        "query": text,
        "sender": (args.sender or []) + inline.get("sender", []),
        "person": (args.person or []) + inline.get("person", []),
        "chat": (args.chat or []) + inline.get("chat", []),
        "network": (args.network or []) + inline.get("network", []),
        "direction": args.direction or inline.get("direction"),
        "when": args.when or inline.get("when"),
        "after": args.after or inline.get("after"),
        "before": args.before or inline.get("before"),
        "mode": args.mode,
        "context": args.context,
        "limit": args.limit,
        "cursor": args.cursor,
    }
    with searcher_factory(settings)() as searcher:
        _print("search_messages", BY_NAME["search_messages"].run(searcher, arguments), args.json)
    return 0


def cmd_thread(settings: RecallSettings, args: argparse.Namespace) -> int:
    arguments = {
        "message_id": args.message_id,
        "conversation_id": args.conversation,
        "before": args.before,
        "after": args.after,
        "when": args.when,
        "limit": args.limit,
        "cursor": args.cursor,
    }
    with searcher_factory(settings)() as searcher:
        _print("get_conversation", BY_NAME["get_conversation"].run(searcher, arguments), args.json)
    return 0


def cmd_chats(settings: RecallSettings, args: argparse.Namespace) -> int:
    arguments = {"query": " ".join(args.query), "network": args.network, "person": args.person, "when": args.when}
    with searcher_factory(settings)() as searcher:
        _print(
            "list_conversations",
            BY_NAME["list_conversations"].run(searcher, {**arguments, "limit": args.limit}),
            args.json,
        )
    return 0


def cmd_people(settings: RecallSettings, args: argparse.Namespace) -> int:
    with searcher_factory(settings)() as searcher:
        result = searcher.people(" ".join(args.query), args.network, args.limit)
        _print("find_people", result, args.json)
    return 0


def cmd_stats(settings: RecallSettings, args: argparse.Namespace) -> int:
    with searcher_factory(settings)() as searcher:
        _print("archive_stats", searcher.stats(), args.json)
    return 0


def cmd_mcp(settings: RecallSettings, args: argparse.Namespace) -> int:
    serve_stdio(McpServer(searcher_factory(settings)))
    return 0


def _embed_in_background(settings: RecallSettings, stop: threading.Event) -> None:
    def work() -> None:
        try:
            with connect(settings.database_url) as conn:
                _pipeline(conn, settings).watch(settings.database_url or "", stop=stop.is_set)
        except Exception:  # noqa: BLE001
            log.exception("embedding worker stopped")

    threading.Thread(target=work, name="embed", daemon=True).start()


def cmd_serve(settings: RecallSettings, args: argparse.Namespace) -> int:
    from .api import serve

    with connect(settings.database_url) as conn:
        migrate(conn)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if args.embed:
        _embed_in_background(settings, stop)
    httpd = serve(searcher_factory(settings), settings.token, args.host or settings.host, args.port or settings.port)
    threading.Thread(target=lambda: (stop.wait(), httpd.shutdown()), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


def cmd_demo(settings: RecallSettings, args: argparse.Namespace) -> int:
    now = datetime.now(settings.timezone)
    with connect(settings.database_url) as conn:
        migrate(conn)
        existing = conn.execute("SELECT count(*) AS n FROM messages WHERE source <> %s", (demo.SOURCE,)).fetchone()
        if existing["n"] and not args.force:
            raise RuntimeError("this database already holds real messages; use a separate database for the demo")
        conn.execute(
            "DELETE FROM conversations WHERE id IN (SELECT conversation_id FROM conversation_sources "
            "WHERE source = %s)",
            (demo.SOURCE,),
        )
        conn.execute("DELETE FROM conversation_sources WHERE source = %s", (demo.SOURCE,))
        counts = demo.seed(conn, now, filler=args.filler)
        Store(conn).put_state(demo.SOURCE, "seed", status="ok", cursor=None, frontier=now, detail=counts, success=True)
        print(f"seeded fabricated archive: {json.dumps(counts)}")
        if from_settings(settings):
            stats = _pipeline(conn, settings).run()
            print(f"embedded: {json.dumps(stats.as_dict())}")
        else:
            print("semantic search is off; set EMBED_PROVIDER (e.g. hash or ollama) and run `inbox-recall embed`")
    return 0


def cmd_eval(settings: RecallSettings, args: argparse.Namespace) -> int:
    embedder = from_settings(settings)
    with connect(settings.database_url) as conn:
        state = Store(conn).get_state(demo.SOURCE, "seed")
        if not state:
            raise RuntimeError("no demo archive here; run `inbox-recall demo` against a scratch database first")
        if embedder:
            pending = EmbedPipeline(conn, embedder).pending()
            if pending:
                raise RuntimeError(f"{pending} messages still need embeddings; run `inbox-recall embed`")
        now = state["frontier"].astimezone(settings.timezone)
        report = bench.run(conn, embedder, now, tuple(args.mode) if args.mode else None)
    print(json.dumps(report, indent=2) if args.json else bench.table(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inbox-recall",
        description="Hybrid search over your Beeper + Instagram message archive, for you and your agents.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="create or upgrade the sync + search schema").set_defaults(func=cmd_migrate)

    emb = sub.add_parser("embed", help="embed new and edited messages (incremental, resumable)")
    emb.add_argument("--limit", type=int, help="stop after this many messages")
    emb.add_argument("--watch", action="store_true", help="keep running; wake on new messages")
    emb.add_argument("--interval", type=float, default=300.0, help="--watch fallback poll interval in seconds")
    emb.add_argument("--status", action="store_true", help="print how many messages are waiting")
    emb.add_argument("--drop-model", metavar="NAME", help="delete one model's vectors and index")
    emb.set_defaults(func=cmd_embed)

    s = sub.add_parser("search", help="search messages (supports from: with: in: on: when: after: before: is:)")
    s.add_argument("query", nargs="*")
    s.add_argument("--from", dest="sender", action="append", help="sender name or handle; 'me' for yours")
    s.add_argument("--with", dest="person", action="append", help="only chats that include this person")
    s.add_argument("--in", dest="chat", action="append", help="chat title fragment or conversation id")
    s.add_argument("--network", action="append")
    s.add_argument("--direction", choices=["incoming", "outgoing"])
    s.add_argument("--when", help="plain English: 'last spring', 'march 2024', 'past 2 weeks'")
    s.add_argument("--after", help="ISO date")
    s.add_argument("--before", help="ISO date")
    s.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    s.add_argument("--context", type=int, default=2)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--cursor")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    th = sub.add_parser("thread", help="read the conversation around a message")
    th.add_argument("message_id", type=int, nargs="?")
    th.add_argument("--conversation", type=int)
    th.add_argument("--before", type=int, default=10)
    th.add_argument("--after", type=int, default=10)
    th.add_argument("--when")
    th.add_argument("--limit", type=int, default=50)
    th.add_argument("--cursor")
    th.add_argument("--json", action="store_true")
    th.set_defaults(func=cmd_thread)

    ch = sub.add_parser("chats", help="list conversations")
    ch.add_argument("query", nargs="*")
    ch.add_argument("--network", action="append")
    ch.add_argument("--with", dest="person", action="append")
    ch.add_argument("--when")
    ch.add_argument("--limit", type=int, default=20)
    ch.add_argument("--json", action="store_true")
    ch.set_defaults(func=cmd_chats)

    pp = sub.add_parser("people", help="find people across networks")
    pp.add_argument("query", nargs="*")
    pp.add_argument("--network", action="append")
    pp.add_argument("--limit", type=int, default=20)
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_people)

    st = sub.add_parser("stats", help="archive and embedding coverage")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stats)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(func=cmd_mcp)

    sv = sub.add_parser("serve", help="HTTP API + MCP over HTTP (POST /mcp)")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.add_argument("--embed", action="store_true", help="also run the embedding worker")
    sv.set_defaults(func=cmd_serve)

    dm = sub.add_parser("demo", help="load the fabricated demo archive into a scratch database")
    dm.add_argument("--filler", type=int, default=3000)
    dm.add_argument("--force", action="store_true")
    dm.set_defaults(func=cmd_demo)

    ev = sub.add_parser("eval", help="run the retrieval benchmark on the demo archive")
    ev.add_argument("--mode", action="append", choices=["keyword", "semantic", "hybrid"])
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.func(RecallSettings(), args)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
