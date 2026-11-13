"""Incremental, resumable embedding of the message archive.

The work queue is the database itself: a message needs a vector when it has
text, is not deleted, and either has no embedding for the active model or was
edited after its embedding was made (`messages.content_updated_at` moved past
`message_embeddings.content_updated_at`). Each batch commits on its own, so a
crash or Ctrl-C loses at most one batch and the next run picks up the rest.

Short chat messages ("yes!", "booked it") carry little meaning alone, so each
message is embedded together with the chat title and a few preceding messages
from the same conversation. That is what lets "the Lisbon flights" find a reply
that never says Lisbon.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import psycopg

from beeper_pg_sync.store import NOTIFY_CHANNEL, Store

from .embedders import Embedder, to_pgvector
from .schema import ensure_model

log = logging.getLogger(__name__)

STATE_SOURCE = "inbox-recall"
MAX_BODY_CHARS = 2000
MAX_CONTEXT_CHARS = 280

PENDING_SQL = """
SELECT m.id, m.body, m.direction, m.content_updated_at, m.conversation_id, m.sent_at,
       coalesce(p.display_name, m.sender_name) AS sender,
       c.title, c.network
FROM messages m
JOIN conversations c ON c.id = m.conversation_id
LEFT JOIN participants p ON p.id = m.sender_id
LEFT JOIN message_embeddings e ON e.message_id = m.id AND e.model = %(model)s
WHERE m.id > %(after)s
  AND NOT m.deleted AND m.body <> ''
  AND (e.message_id IS NULL OR e.content_updated_at < m.content_updated_at)
ORDER BY m.id
LIMIT %(limit)s
"""

CONTEXT_SQL = """
SELECT t.id AS target, w.sender, w.direction, w.body
FROM unnest(%(ids)s::bigint[]) AS t(id)
JOIN messages tm ON tm.id = t.id
CROSS JOIN LATERAL (
    SELECT coalesce(p.display_name, m.sender_name) AS sender, m.direction, m.body, m.sent_at, m.id
    FROM messages m LEFT JOIN participants p ON p.id = m.sender_id
    WHERE m.conversation_id = tm.conversation_id
      AND tm.sent_at IS NOT NULL
      AND m.sent_at IS NOT NULL
      AND (m.sent_at, m.id) < (tm.sent_at, tm.id)
      AND m.sent_at > tm.sent_at - interval '6 hours'
      AND NOT m.deleted AND m.body <> ''
    ORDER BY m.sent_at DESC, m.id DESC
    LIMIT %(n)s
) w
ORDER BY t.id, w.sent_at, w.id
"""

PENDING_COUNT_SQL = """
SELECT count(*) AS n
FROM messages m
LEFT JOIN message_embeddings e ON e.message_id = m.id AND e.model = %(model)s
WHERE NOT m.deleted AND m.body <> ''
  AND (e.message_id IS NULL OR e.content_updated_at < m.content_updated_at)
"""


def speaker(sender: str | None, direction: str) -> str:
    if direction == "outgoing":
        return "me"
    return sender or "someone"


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def document_text(row: dict, context: list[dict]) -> str:
    """The text that gets embedded for one message."""
    title = row["title"] or "untitled chat"
    lines = [f"{row['network']} chat: {title}"]
    for prior in context:
        lines.append(f"{speaker(prior['sender'], prior['direction'])}: {clip(prior['body'], MAX_CONTEXT_CHARS)}")
    lines.append(f"{speaker(row['sender'], row['direction'])}: {clip(row['body'], MAX_BODY_CHARS)}")
    return "\n".join(lines)


@dataclass
class EmbedStats:
    embedded: int = 0
    removed: int = 0
    batches: int = 0
    seconds: float = 0.0
    model: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "embedded": self.embedded,
            "removed": self.removed,
            "batches": self.batches,
            "seconds": round(self.seconds, 2),
        }


class EmbedPipeline:
    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder,
        *,
        batch_size: int = 64,
        context_messages: int = 2,
    ):
        self.conn = conn
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.context_messages = max(0, context_messages)
        self._registered = (
            conn.execute("SELECT 1 FROM embedding_models WHERE model = %s", (embedder.name,)).fetchone() is not None
        )

    @property
    def model(self) -> str:
        return self.embedder.name

    def pending(self) -> int:
        return self.conn.execute(PENDING_COUNT_SQL, {"model": self.model}).fetchone()["n"]

    def prune(self) -> int:
        """Drop vectors for messages that were deleted or emptied since they were embedded."""
        cur = self.conn.execute(
            """DELETE FROM message_embeddings e USING messages m
               WHERE e.message_id = m.id AND e.model = %s AND (m.deleted OR m.body = '')""",
            (self.model,),
        )
        return cur.rowcount or 0

    def _context(self, ids: list[int]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {i: [] for i in ids}
        if not self.context_messages:
            return out
        for row in self.conn.execute(CONTEXT_SQL, {"ids": ids, "n": self.context_messages}):
            out[row["target"]].append(row)
        return out

    def _store(self, rows: list[dict], vectors: list[list[float]]) -> None:
        if not self._registered:
            ensure_model(self.conn, self.model, self.embedder.provider, len(vectors[0]))
            self._registered = True
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO message_embeddings (message_id, model, embedding, content_updated_at)
                   VALUES (%s, %s, %s::vector, %s)
                   ON CONFLICT (message_id, model) DO UPDATE SET
                       embedding = EXCLUDED.embedding,
                       content_updated_at = EXCLUDED.content_updated_at,
                       embedded_at = now()""",
                [
                    (row["id"], self.model, to_pgvector(vec), row["content_updated_at"])
                    for row, vec in zip(rows, vectors, strict=True)
                ],
            )

    def run(self, limit: int | None = None, progress: Callable[[EmbedStats], None] | None = None) -> EmbedStats:
        """Embed everything pending (or up to `limit` messages). Safe to interrupt and rerun."""
        stats = EmbedStats(model=self.model)
        started = time.monotonic()
        stats.removed = self.prune()
        after = 0
        while limit is None or stats.embedded < limit:
            size = self.batch_size if limit is None else min(self.batch_size, limit - stats.embedded)
            rows = self.conn.execute(PENDING_SQL, {"model": self.model, "after": after, "limit": size}).fetchall()
            if not rows:
                break
            context = self._context([r["id"] for r in rows])
            prefix = self.embedder.document_prefix
            texts = [prefix + document_text(r, context[r["id"]]) for r in rows]
            vectors = self.embedder.embed(texts)
            self._store(rows, vectors)
            after = rows[-1]["id"]
            stats.embedded += len(rows)
            stats.batches += 1
            if progress:
                progress(stats)
        stats.seconds = time.monotonic() - started
        if stats.embedded or stats.removed:
            Store(self.conn).put_state(
                STATE_SOURCE,
                f"embed:{self.model}",
                status="ok",
                cursor=None,
                frontier=None,
                detail=stats.as_dict(),
                success=True,
            )
        return stats

    def watch(self, dsn: str, interval: float = 300.0, stop: Callable[[], bool] = lambda: False) -> None:
        """Embed now, then again whenever the sync layer NOTIFYs or `interval` passes."""
        with psycopg.connect(dsn, autocommit=True) as listener:
            listener.execute(f"LISTEN {NOTIFY_CHANNEL}")
            while not stop():
                stats = self.run()
                if stats.embedded or stats.removed:
                    log.info("embedded %s", stats.as_dict())
                deadline = time.monotonic() + interval
                woke = False
                while not woke and not stop() and time.monotonic() < deadline:
                    woke = any(True for _ in listener.notifies(timeout=min(5.0, interval), stop_after=1))
                if woke:
                    time.sleep(1.0)  # let a burst of ingests settle into one pass
                    for _ in listener.notifies(timeout=0.01):
                        pass
