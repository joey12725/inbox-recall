"""Hybrid retrieval over the message archive.

Three rankers run against the same filters and are merged with weighted
reciprocal rank fusion (RRF):

- semantic: cosine ANN over the active model's partial HNSW index
- keyword: Postgres full-text search (English stemming), OR-of-terms with a
  bonus for messages that contain every term
- fuzzy: trigram word similarity, for names, typos and partial words in short
  queries, used when full-text search finds fewer than three messages with
  every term

Hits come back as conversation excerpts: each hit plus N messages either side,
with overlapping windows from the same chat merged, so an agent reads threads
rather than isolated lines. Every message carries its stable id for citations.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime, tzinfo
from functools import lru_cache
from typing import Any

import psycopg
from psycopg import sql

from .embedders import Embedder, to_pgvector
from .query import SearchRequest, decode_cursor, encode_cursor
from .schema import vector_version
from .timeparse import TimeRange, parse_when

RRF_K = 60
WEIGHTS = {"semantic": 1.0, "keyword": 1.0, "fuzzy": 0.6}
ALL_TERMS_BONUS = 0.5 / (RRF_K + 1)  # a message containing every query term is rarely a false positive
FUZZY_WHEN_FEWER_THAN = 3  # trigram matching backs up full-text search when exact terms come up short
DEFAULT_POOL = 200
MAX_POOL = 1000
FUZZY_MAX_WORDS = 4
FUZZY_THRESHOLD = 0.55
SELF_NAMES = {"me", "myself", "i", "self", "you"}

MESSAGE_COLUMNS = """
    m.id, m.conversation_id, c.title AS chat, c.network, c.kind AS chat_kind,
    coalesce(p.display_name, m.sender_name) AS sender, p.handle AS sender_handle,
    m.direction, m.sent_at, m.time_precision, m.kind, m.body, m.edited_at,
    m.reply_to_message_id, m.attachments, m.reactions
"""
FROM_MESSAGES = """
FROM messages m
JOIN conversations c ON c.id = m.conversation_id
LEFT JOIN participants p ON p.id = m.sender_id
"""


def like(value: str) -> str:
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def terms(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1 or t.isdigit()]


def websearch_syntax(text: str) -> bool:
    return '"' in text or re.search(r"(^|\s)-\w", text) is not None or " or " in f" {text.lower()} "


class Filters:
    """WHERE clauses shared by every ranker and by browsing."""

    def __init__(self) -> None:
        self.clauses: list[str] = ["NOT m.deleted", "m.body <> ''"]
        self.params: dict[str, Any] = {}
        self.applied: dict[str, Any] = {}

    def add(self, clause: str, **params: Any) -> None:
        self.clauses.append(clause)
        self.params.update(params)

    @property
    def sql(self) -> str:
        return " AND ".join(self.clauses)


class Searcher:
    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder | None = None,
        *,
        tz: tzinfo = UTC,
        now: datetime | None = None,
        pool: int = DEFAULT_POOL,
    ):
        self.conn = conn
        self.embedder = embedder
        self.tz = tz
        self._now = now
        self.pool = pool
        self._embed_query = lru_cache(maxsize=256)(self._embed_query_uncached)

    # -- helpers ------------------------------------------------------------

    def now(self) -> datetime:
        return (self._now or datetime.now(UTC)).astimezone(self.tz)

    def _embed_query_uncached(self, text: str) -> str:
        assert self.embedder is not None
        return to_pgvector(self.embedder.embed([self.embedder.query_prefix + text])[0])

    def _model_dims(self) -> int | None:
        if not self.embedder:
            return None
        row = self.conn.execute("SELECT dims FROM embedding_models WHERE model = %s", (self.embedder.name,)).fetchone()
        return row["dims"] if row else None

    def time_range(self, when: str | None, after: datetime | None, before: datetime | None) -> TimeRange:
        rng = parse_when(when, self.now()) if when else TimeRange(None, None)
        start, end = rng.start, rng.end
        if after:
            after = after if after.tzinfo else after.replace(tzinfo=self.tz)
            start = max(start, after) if start else after
        if before:
            before = before if before.tzinfo else before.replace(tzinfo=self.tz)
            end = min(end, before) if end else before
        return TimeRange(start, end, rng.phrase)

    def _person_conversations(self, names: list[str]) -> list[int]:
        ids: set[int] = set()
        for name in names:
            pattern = like(name)
            people = [
                r["id"]
                for r in self.conn.execute(
                    "SELECT id FROM participants WHERE display_name ILIKE %(p)s OR handle ILIKE %(p)s", {"p": pattern}
                )
            ]
            if people:
                ids.update(
                    r["conversation_id"]
                    for r in self.conn.execute(
                        "SELECT DISTINCT conversation_id FROM messages WHERE sender_id = ANY(%s)", (people,)
                    )
                )
            ids.update(
                r["id"] for r in self.conn.execute("SELECT id FROM conversations WHERE title ILIKE %s", (pattern,))
            )
        return sorted(ids)

    def filters(self, req: SearchRequest) -> Filters:
        f = Filters()
        if req.sender:
            parts = []
            for i, name in enumerate(req.sender):
                if name.lower() in SELF_NAMES:
                    parts.append("m.direction = 'outgoing'")
                    continue
                key = f"sender{i}"
                f.params[key] = like(name)
                parts.append(
                    f"(p.display_name ILIKE %({key})s OR p.handle ILIKE %({key})s OR m.sender_name ILIKE %({key})s)"
                )
            f.add("(" + " OR ".join(parts) + ")")
            f.applied["sender"] = req.sender
        if req.person:
            f.add("m.conversation_id = ANY(%(person_chats)s)", person_chats=self._person_conversations(req.person))
            f.applied["person"] = req.person
        if req.chat:
            parts = []
            for i, chat in enumerate(req.chat):
                key = f"chat{i}"
                if chat.isdigit():
                    f.params[key] = int(chat)
                    parts.append(f"c.id = %({key})s")
                else:
                    f.params[key] = like(chat)
                    parts.append(f"c.title ILIKE %({key})s")
            f.add("(" + " OR ".join(parts) + ")")
            f.applied["chat"] = req.chat
        if req.network:
            f.add("c.network = ANY(%(networks)s)", networks=req.network)
            f.applied["network"] = req.network
        if req.direction:
            f.add("m.direction = %(direction)s", direction=req.direction)
            f.applied["direction"] = req.direction
        rng = self.time_range(req.when, req.after, req.before)
        if rng.start:
            f.add("m.sent_at >= %(t_start)s", t_start=rng.start)
        if rng.end:
            f.add("m.sent_at < %(t_end)s", t_end=rng.end)
        if rng.start or rng.end:
            f.applied["time"] = rng.label + (f' ("{rng.phrase}")' if rng.phrase else "")
        return f

    # -- rankers ------------------------------------------------------------

    def _semantic(self, query: str, f: Filters, k: int, dims: int) -> list[int]:
        assert self.embedder is not None
        q = sql.SQL(
            "SELECT m.id, (e.embedding::vector({dims}) <=> %(qvec)s::vector({dims})) AS distance "
            "FROM message_embeddings e JOIN messages m ON m.id = e.message_id "
            "JOIN conversations c ON c.id = m.conversation_id LEFT JOIN participants p ON p.id = m.sender_id "
            "WHERE e.model = {model} AND {where} ORDER BY distance LIMIT %(k)s"
        ).format(dims=sql.Literal(dims), model=sql.Literal(self.embedder.name), where=sql.SQL(f.sql))
        params = {**f.params, "qvec": self._embed_query(query), "k": k}
        with self.conn.transaction():
            if vector_version(self.conn) >= (0, 8):
                self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            self.conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(max(40, min(1000, k)))))
            rows = self.conn.execute(q, params).fetchall()
        rows.sort(key=lambda r: (r["distance"], -r["id"]))  # relaxed_order may return near-sorted rows
        return [r["id"] for r in rows]

    def _keyword(self, query: str, f: Filters, k: int) -> tuple[list[int], set[int]]:
        """Full-text matches for any query term; also returns the ids that contain every term."""
        if websearch_syntax(query):
            tsquery = "websearch_to_tsquery('english', %(q)s)"
            params = {"q": query}
        else:
            words = terms(query)
            if not words:
                return [], set()
            tsquery = "to_tsquery('english', %(q)s)"
            params = {"q": " | ".join(words)}
        q = f"""
            SELECT m.id, s.all_terms, ts_rank_cd(m.search_tsv, tq.q, 1) AS score
            {FROM_MESSAGES}
            CROSS JOIN (SELECT {tsquery} AS q, plainto_tsquery('english', %(plain)s) AS every) tq
            CROSS JOIN LATERAL (SELECT m.search_tsv @@ tq.every AS all_terms) s
            WHERE m.search_tsv @@ tq.q AND {f.sql}
            ORDER BY s.all_terms DESC, score DESC, m.sent_at DESC NULLS LAST, m.id DESC
            LIMIT %(k)s"""
        rows = self.conn.execute(q, {**f.params, **params, "plain": query, "k": k}).fetchall()
        return [r["id"] for r in rows], {r["id"] for r in rows if r["all_terms"]}

    def _fuzzy(self, query: str, f: Filters, k: int) -> list[int]:
        words = terms(query)
        if not words or len(words) > FUZZY_MAX_WORDS or len(query.strip()) < 3:
            return []
        q = f"""
            SELECT m.id, word_similarity(%(fq)s, m.body) AS score
            {FROM_MESSAGES}
            WHERE %(fq)s <%% m.body AND {f.sql}
            ORDER BY score DESC, m.sent_at DESC NULLS LAST, m.id DESC
            LIMIT %(k)s"""
        with self.conn.transaction():
            self.conn.execute(
                sql.SQL("SET LOCAL pg_trgm.word_similarity_threshold = {}").format(sql.Literal(FUZZY_THRESHOLD))
            )
            rows = self.conn.execute(q, {**f.params, "fq": " ".join(words), "k": k}).fetchall()
        return [r["id"] for r in rows]

    # -- public API ---------------------------------------------------------

    def search(self, req: SearchRequest) -> dict[str, Any]:
        req = req.normalized()
        if not req.query:
            return self.browse(req)
        cursor = decode_cursor(req.cursor) if req.cursor else {}
        if cursor and cursor.get("f") != req.fingerprint():
            raise ValueError("cursor belongs to a different search; repeat the original query and filters")
        offset = int(cursor.get("o", 0))
        pool = int(cursor.get("k", 0)) or min(MAX_POOL, max(self.pool, 3 * req.limit))
        f = self.filters(req)
        notes: list[str] = []

        dims = self._model_dims()
        mode = req.mode
        if mode in ("hybrid", "semantic") and not dims:
            notes.append("no embeddings for the configured model yet; ran keyword search instead")
            mode = "keyword"

        ranked: dict[str, list[int]] = {}
        every: set[int] = set()
        if mode in ("hybrid", "semantic") and dims:
            ranked["semantic"] = self._semantic(req.query, f, pool, dims)
        if mode in ("hybrid", "keyword"):
            ranked["keyword"], every = self._keyword(req.query, f, pool)
            if len(every) < FUZZY_WHEN_FEWER_THAN:
                ranked["fuzzy"] = self._fuzzy(req.query, f, pool)

        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, list[str]] = defaultdict(list)
        for name, ids in ranked.items():
            for rank, message_id in enumerate(ids):
                scores[message_id] += WEIGHTS[name] / (RRF_K + rank + 1)
                matched[message_id].append(name)
        for message_id in every:
            scores[message_id] += ALL_TERMS_BONUS
        fused = sorted(scores, key=lambda i: (-scores[i], -i))
        page = fused[offset : offset + req.limit]
        has_more = len(fused) > offset + req.limit
        hits = [
            {"rank": offset + n + 1, "id": i, "score": round(scores[i], 5), "matched_by": matched[i]}
            for n, i in enumerate(page)
        ]
        if not has_more and any(len(ids) >= pool for ids in ranked.values()):
            notes.append("reached the end of the relevance pool; drop the query text to page through every match")
        return {
            "query": req.query,
            "mode": mode,
            "filters": f.applied,
            "total_candidates": len(fused),
            "offset": offset,
            "hits": hits,
            "excerpts": self.excerpts(page, req.context, {h["id"]: h["rank"] for h in hits}),
            "next_cursor": encode_cursor({"o": offset + req.limit, "k": pool, "f": req.fingerprint()})
            if has_more
            else None,
            "notes": notes,
        }

    def browse(self, req: SearchRequest) -> dict[str, Any]:
        """No query text: every message matching the filters, newest first, keyset-paginated."""
        cursor = decode_cursor(req.cursor) if req.cursor else {}
        if cursor and cursor.get("f") != req.fingerprint():
            raise ValueError("cursor belongs to a different search; repeat the original filters")
        f = self.filters(req)
        f.add("m.sent_at IS NOT NULL")
        if cursor.get("b"):
            f.add(
                "(m.sent_at, m.id) < (%(b_at)s::timestamptz, %(b_id)s)",
                b_at=cursor["b"][0],
                b_id=int(cursor["b"][1]),
            )
        rows = self.conn.execute(
            f"SELECT m.id, m.sent_at {FROM_MESSAGES} WHERE {f.sql} ORDER BY m.sent_at DESC, m.id DESC LIMIT %(k)s",
            {**f.params, "k": req.limit + 1},
        ).fetchall()
        has_more = len(rows) > req.limit
        rows = rows[: req.limit]
        ids = [r["id"] for r in rows]
        last = rows[-1] if rows else None
        return {
            "query": "",
            "mode": "browse",
            "filters": f.applied,
            "total_candidates": None,
            "offset": None,
            "hits": [{"rank": n + 1, "id": i, "score": None, "matched_by": ["filters"]} for n, i in enumerate(ids)],
            "excerpts": self.excerpts(ids, req.context, {i: n + 1 for n, i in enumerate(ids)}),
            "next_cursor": encode_cursor({"b": [last["sent_at"].isoformat(), last["id"]], "f": req.fingerprint()})
            if has_more and last
            else None,
            "notes": [],
        }

    # -- hydration ----------------------------------------------------------

    def fetch(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        rows = self.conn.execute(
            f"SELECT {MESSAGE_COLUMNS} {FROM_MESSAGES} WHERE m.id = ANY(%s)", (list(set(ids)),)
        ).fetchall()
        return {r["id"]: r for r in rows}

    def message_view(self, row: dict[str, Any]) -> dict[str, Any]:
        sent = row["sent_at"]
        view: dict[str, Any] = {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "chat": row["chat"],
            "network": row["network"],
            "sender": "me" if row["direction"] == "outgoing" else (row["sender"] or "unknown"),
            "direction": row["direction"],
            "sent_at": sent.astimezone(self.tz).isoformat(timespec="minutes") if sent else None,
            "text": row["body"],
        }
        if row["direction"] == "outgoing" and row["sender"]:
            view["sender_name"] = row["sender"]
        if row["time_precision"] != "exact":
            view["time_precision"] = row["time_precision"]
        if row["kind"] != "text":
            view["kind"] = row["kind"]
        if row["edited_at"]:
            view["edited"] = True
        if row["reply_to_message_id"]:
            view["reply_to"] = row["reply_to_message_id"]
        if row["attachments"]:
            view["attachments"] = [a.get("fileName") or a.get("type") or "file" for a in row["attachments"]]
        if row["reactions"]:
            view["reactions"] = [r.get("reaction") for r in row["reactions"] if r.get("reaction")]
        return view

    def chat_view(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"id": row["conversation_id"], "title": row["chat"], "network": row["network"], "kind": row["chat_kind"]}

    def window_ids(self, hit_ids: list[int], n: int) -> dict[int, list[int]]:
        windows = {i: [i] for i in hit_ids}
        if not hit_ids or n <= 0:
            return windows
        rows = self.conn.execute(
            """
            SELECT h.id AS hit, w.id
            FROM unnest(%(ids)s::bigint[]) AS h(id)
            JOIN messages hm ON hm.id = h.id
            CROSS JOIN LATERAL (
                (SELECT m.id FROM messages m
                 WHERE m.conversation_id = hm.conversation_id AND hm.sent_at IS NOT NULL
                   AND (m.sent_at, m.id) < (hm.sent_at, hm.id) AND NOT m.deleted
                 ORDER BY m.sent_at DESC, m.id DESC LIMIT %(n)s)
                UNION ALL
                (SELECT m.id FROM messages m
                 WHERE m.conversation_id = hm.conversation_id AND hm.sent_at IS NOT NULL
                   AND (m.sent_at, m.id) > (hm.sent_at, hm.id) AND NOT m.deleted
                 ORDER BY m.sent_at, m.id LIMIT %(n)s)
            ) w
            """,
            {"ids": hit_ids, "n": n},
        ).fetchall()
        for r in rows:
            windows[r["hit"]].append(r["id"])
        return windows

    def excerpts(self, hit_ids: list[int], context: int, ranks: dict[int, int]) -> list[dict[str, Any]]:
        """Group hits and their context windows into per-conversation excerpts."""
        windows = self.window_ids(hit_ids, context)
        rows = self.fetch([i for w in windows.values() for i in w])
        parents = {
            r["reply_to_message_id"]
            for r in rows.values()
            if r["reply_to_message_id"] and r["reply_to_message_id"] not in rows
        }
        rows.update(self.fetch(list(parents)))

        # Union windows that share a message (same chat, overlapping context).
        parent_of: dict[int, int] = {}

        def find(x: int) -> int:
            while parent_of.setdefault(x, x) != x:
                x = parent_of[x]
            return x

        owner: dict[int, int] = {}
        for hit, ids in windows.items():
            for i in ids:
                if i in owner:
                    parent_of[find(hit)] = find(owner[i])
                else:
                    owner[i] = hit
        groups: dict[int, set[int]] = defaultdict(set)
        group_hits: dict[int, list[int]] = defaultdict(list)
        for hit, ids in windows.items():
            if hit not in rows:
                continue
            root = find(hit)
            groups[root].update(i for i in ids if i in rows)
            group_hits[root].append(hit)

        def order(i: int) -> tuple:
            sent = rows[i]["sent_at"]
            return (sent is None, sent or datetime.min.replace(tzinfo=UTC), i)

        out = []
        for root, ids in groups.items():
            hits = sorted(group_hits[root], key=lambda h: ranks.get(h, 0))
            messages = []
            for i in sorted(ids, key=order):
                view = self.message_view(rows[i])
                if i in ranks and i in group_hits[root]:
                    view["hit"] = ranks[i]
                parent = rows[i]["reply_to_message_id"]
                if parent and parent not in ids and parent in rows:
                    p = self.message_view(rows[parent])
                    view["in_reply_to"] = {k: p[k] for k in ("id", "sender", "sent_at", "text")}
                messages.append(view)
            out.append(
                {
                    "conversation": self.chat_view(rows[hits[0]]),
                    "hit_ids": hits,
                    "best_rank": ranks.get(hits[0], 0),
                    "messages": messages,
                }
            )
        out.sort(key=lambda e: e["best_rank"])
        return out

    # -- other tools --------------------------------------------------------

    def get_messages(self, ids: list[int], context: int = 0) -> dict[str, Any]:
        ids = [int(i) for i in ids][:100]
        found = self.fetch(ids)
        missing = [i for i in ids if i not in found]
        present = [i for i in ids if i in found]
        return {
            "excerpts": self.excerpts(present, context, {i: n + 1 for n, i in enumerate(present)}),
            "missing": missing,
        }

    def conversation(
        self,
        conversation_id: int | None = None,
        message_id: int | None = None,
        before: int = 10,
        after: int = 10,
        when: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Reconstruct a thread: around a message, over a time range, or the latest messages."""
        limit = max(1, min(200, int(limit)))
        before, after = max(0, min(200, int(before))), max(0, min(200, int(after)))
        center = None
        if cursor:
            c = decode_cursor(cursor)
            anchor = self.fetch([int(c["a"])]).get(int(c["a"]))
            if not anchor:
                raise ValueError("cursor anchor no longer exists")
            conversation_id = anchor["conversation_id"]
            older, newer = (limit, 0) if c["d"] == "older" else (0, limit)
            ids = self._neighbors(anchor, older, newer)
            ids.remove(anchor["id"])
        elif message_id is not None:
            anchor = self.fetch([int(message_id)]).get(int(message_id))
            if not anchor:
                raise ValueError(f"no message with id {message_id}")
            conversation_id, center = anchor["conversation_id"], anchor["id"]
            ids = self._neighbors(anchor, before, after)
        elif conversation_id is not None:
            rng = self.time_range(when, None, None)
            clauses = ["m.conversation_id = %(c)s", "NOT m.deleted", "m.sent_at IS NOT NULL"]
            params: dict[str, Any] = {"c": int(conversation_id), "k": limit}
            if rng.start:
                clauses.append("m.sent_at >= %(s)s")
                params["s"] = rng.start
            if rng.end:
                clauses.append("m.sent_at < %(e)s")
                params["e"] = rng.end
            # With a range, read forward from its start; otherwise show the latest messages.
            direction = "ASC" if rng.start else "DESC"
            rows = self.conn.execute(
                f"SELECT m.id FROM messages m WHERE {' AND '.join(clauses)} "
                f"ORDER BY m.sent_at {direction}, m.id {direction} LIMIT %(k)s",
                params,
            ).fetchall()
            ids = [r["id"] for r in rows]
        else:
            raise ValueError("pass conversation_id, message_id or cursor")

        rows = self.fetch(ids)
        ordered = sorted(rows.values(), key=lambda r: (r["sent_at"] or datetime.min.replace(tzinfo=UTC), r["id"]))
        info = self.conn.execute(
            "SELECT id AS conversation_id, title AS chat, network, kind AS chat_kind FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        if not info:
            raise ValueError(f"no conversation with id {conversation_id}")
        messages = [self.message_view(r) for r in ordered]
        for m in messages:
            if m["id"] == center:
                m["hit"] = 1
        first, last = (ordered[0], ordered[-1]) if ordered else (None, None)
        return {
            "conversation": self.chat_view(info),
            "messages": messages,
            "older_cursor": encode_cursor({"a": first["id"], "d": "older"})
            if first and self._has_neighbor(first, older=True)
            else None,
            "newer_cursor": encode_cursor({"a": last["id"], "d": "newer"})
            if last and self._has_neighbor(last, older=False)
            else None,
        }

    def _neighbors(self, anchor: dict[str, Any], older: int, newer: int) -> list[int]:
        if anchor["sent_at"] is None:
            return [anchor["id"]]
        windows = self.window_ids([anchor["id"]], max(older, newer))
        rows = self.fetch(windows[anchor["id"]])
        key = (anchor["sent_at"], anchor["id"])
        earlier = sorted(
            (r for r in rows.values() if (r["sent_at"], r["id"]) < key), key=lambda r: (r["sent_at"], r["id"])
        )
        later = sorted(
            (r for r in rows.values() if (r["sent_at"], r["id"]) > key), key=lambda r: (r["sent_at"], r["id"])
        )
        picked = earlier[len(earlier) - older :] if older else []
        return [r["id"] for r in picked] + [anchor["id"]] + [r["id"] for r in later[:newer]]

    def _has_neighbor(self, row: dict[str, Any], older: bool) -> bool:
        if row["sent_at"] is None:
            return False
        op = "<" if older else ">"
        return (
            self.conn.execute(
                f"SELECT 1 FROM messages m WHERE m.conversation_id = %s AND NOT m.deleted "
                f"AND (m.sent_at, m.id) {op} (%s, %s) LIMIT 1",
                (row["conversation_id"], row["sent_at"], row["id"]),
            ).fetchone()
            is not None
        )

    def conversations(
        self,
        query: str = "",
        network: list[str] | None = None,
        person: list[str] | None = None,
        when: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(100, int(limit)))
        offset = int(decode_cursor(cursor).get("o", 0)) if cursor else 0
        clauses, params = ["true"], {"k": limit + 1, "o": offset}
        if query:
            clauses.append("c.title ILIKE %(t)s")
            params["t"] = like(query)
        if network:
            clauses.append("c.network = ANY(%(n)s)")
            params["n"] = [x.lower() for x in network]
        if person:
            clauses.append("c.id = ANY(%(pc)s)")
            params["pc"] = self._person_conversations(person)
        having = ""
        rng = self.time_range(when, None, None)
        if rng.start or rng.end:
            having = "HAVING bool_or(m.sent_at >= %(s)s AND m.sent_at < %(e)s)"
            params["s"] = rng.start or datetime.min.replace(tzinfo=UTC)
            params["e"] = rng.end or datetime.max.replace(tzinfo=UTC)
        rows = self.conn.execute(
            f"""SELECT c.id, c.title, c.network, c.kind, count(m.id) AS messages,
                       min(m.sent_at) AS first_message, max(m.sent_at) AS last_message
                FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id AND NOT m.deleted
                WHERE {" AND ".join(clauses)}
                GROUP BY c.id {having}
                ORDER BY max(m.sent_at) DESC NULLS LAST, c.id DESC
                LIMIT %(k)s OFFSET %(o)s""",
            params,
        ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        members = self._members([r["id"] for r in rows])
        chats = [
            {
                "id": r["id"],
                "title": r["title"],
                "network": r["network"],
                "kind": r["kind"],
                "messages": r["messages"],
                "first_message": self._iso(r["first_message"]),
                "last_message": self._iso(r["last_message"]),
                "people": members.get(r["id"], []),
            }
            for r in rows
        ]
        return {"conversations": chats, "next_cursor": encode_cursor({"o": offset + limit}) if has_more else None}

    def _members(self, conversation_ids: list[int], top: int = 6) -> dict[int, list[str]]:
        if not conversation_ids:
            return {}
        rows = self.conn.execute(
            """SELECT m.conversation_id, coalesce(p.display_name, m.sender_name) AS name, count(*) AS n
               FROM messages m LEFT JOIN participants p ON p.id = m.sender_id
               WHERE m.conversation_id = ANY(%s) AND m.direction <> 'outgoing'
               GROUP BY 1, 2 ORDER BY 1, 3 DESC""",
            (conversation_ids,),
        ).fetchall()
        out: dict[int, list[str]] = defaultdict(list)
        for r in rows:
            if r["name"] and len(out[r["conversation_id"]]) < top:
                out[r["conversation_id"]].append(r["name"])
        return out

    def people(self, query: str = "", network: list[str] | None = None, limit: int = 20) -> dict[str, Any]:
        """People across every network, merged by display name."""
        limit = max(1, min(100, int(limit)))
        clauses, params = ["m.direction <> 'outgoing'", "NOT m.deleted"], {}
        if query:
            ids = [
                r["id"]
                for r in self.conn.execute(
                    "SELECT id FROM participants WHERE display_name ILIKE %(q)s OR handle ILIKE %(q)s",
                    {"q": like(query)},
                )
            ]
            clauses.append("(m.sender_id = ANY(%(ids)s) OR m.sender_name ILIKE %(q)s)")
            params.update(ids=ids, q=like(query))
        if network:
            clauses.append("c.network = ANY(%(n)s)")
            params["n"] = [x.lower() for x in network]
        rows = self.conn.execute(
            f"""SELECT coalesce(p.display_name, m.sender_name) AS name, p.handle, c.network,
                       count(*) AS messages, count(DISTINCT m.conversation_id) AS chats,
                       min(m.sent_at) AS first_message, max(m.sent_at) AS last_message
                {FROM_MESSAGES}
                WHERE {" AND ".join(clauses)}
                GROUP BY 1, 2, 3 ORDER BY messages DESC LIMIT 500""",
            params,
        ).fetchall()
        merged: dict[str, dict[str, Any]] = {}
        for r in rows:
            if not r["name"]:
                continue
            person = merged.setdefault(
                r["name"].casefold(),
                {"name": r["name"], "messages": 0, "networks": {}, "handles": [], "first": None, "last": None},
            )
            person["messages"] += r["messages"]
            person["networks"][r["network"]] = person["networks"].get(r["network"], 0) + r["messages"]
            if r["handle"] and r["handle"] not in person["handles"]:
                person["handles"].append(r["handle"])
            for key, value, pick in (("first", r["first_message"], min), ("last", r["last_message"], max)):
                if value:
                    person[key] = pick(person[key], value) if person[key] else value
        people = sorted(merged.values(), key=lambda p: -p["messages"])[:limit]
        for p in people:
            p["first_message"], p["last_message"] = self._iso(p.pop("first")), self._iso(p.pop("last"))
        return {"people": people}

    def stats(self) -> dict[str, Any]:
        networks = self.conn.execute(
            """SELECT c.network, count(m.id) AS messages, count(DISTINCT c.id) AS conversations,
                      min(m.sent_at) AS first_message, max(m.sent_at) AS last_message
               FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id AND NOT m.deleted
               GROUP BY c.network ORDER BY messages DESC"""
        ).fetchall()
        searchable = self.conn.execute(
            "SELECT count(*) AS n FROM messages WHERE NOT deleted AND body <> ''"
        ).fetchone()["n"]
        models = self.conn.execute(
            """SELECT em.model, em.provider, em.dims, count(e.message_id) AS embedded, max(e.embedded_at) AS last
               FROM embedding_models em LEFT JOIN message_embeddings e ON e.model = em.model
               GROUP BY em.model, em.provider, em.dims ORDER BY em.model"""
        ).fetchall()
        return {
            "networks": [
                {
                    **{k: r[k] for k in ("network", "messages", "conversations")},
                    "first_message": self._iso(r["first_message"]),
                    "last_message": self._iso(r["last_message"]),
                }
                for r in networks
            ],
            "searchable_messages": searchable,
            "embedding_models": [
                {
                    "model": r["model"],
                    "provider": r["provider"],
                    "dims": r["dims"],
                    "embedded": r["embedded"],
                    "coverage": round(r["embedded"] / searchable, 4) if searchable else 0,
                    "active": bool(self.embedder and self.embedder.name == r["model"]),
                    "last_embedded": self._iso(r["last"]),
                }
                for r in models
            ],
            "active_model": self.embedder.name if self.embedder else None,
            "timezone": str(self.tz),
            "now": self.now().isoformat(timespec="minutes"),
        }

    def _iso(self, value: datetime | None) -> str | None:
        return value.astimezone(self.tz).isoformat(timespec="minutes") if value else None
