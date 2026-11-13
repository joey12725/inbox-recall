"""Migrations and per-model vector indexes."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

import psycopg
from psycopg import sql

from beeper_pg_sync.db import migrate as core_migrate

MAX_INDEXED_DIMS = 2000  # pgvector's HNSW limit for the `vector` type


def migrations_dir() -> Path:
    return Path(str(resources.files("inbox_recall") / "migrations"))


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply the core sync schema plus the search layer (1000+)."""
    return core_migrate(conn, extra_dirs=[migrations_dir()])


def index_name(model: str) -> str:
    return "message_embeddings_hnsw_" + hashlib.sha1(model.encode()).hexdigest()[:12]


def ensure_model(conn: psycopg.Connection, model: str, provider: str, dims: int) -> None:
    """Register an embedding model and build its partial HNSW index once.

    Each model's vectors live in the shared `message_embeddings` table; the
    index covers only that model's rows, cast to its fixed width, so cosine
    ANN queries for one model never touch another model's data.
    """
    if not 0 < dims <= MAX_INDEXED_DIMS:
        raise ValueError(f"{model} returns {dims} dimensions; HNSW indexes support up to {MAX_INDEXED_DIMS}")
    row = conn.execute("SELECT dims FROM embedding_models WHERE model = %s", (model,)).fetchone()
    if row:
        if row["dims"] != dims:
            raise ValueError(
                f"{model} is registered with {row['dims']} dimensions but returned {dims}; "
                "use a different EMBED_MODEL name or `inbox-recall embed --drop-model`"
            )
        return
    name = index_name(model)
    with conn.transaction():
        conn.execute(
            """INSERT INTO embedding_models (model, provider, dims, index_name)
               VALUES (%s, %s, %s, %s) ON CONFLICT (model) DO NOTHING""",
            (model, provider, dims, name),
        )
        conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON message_embeddings "
                "USING hnsw ((embedding::vector({dims})) vector_cosine_ops) WHERE model = {model}"
            ).format(name=sql.Identifier(name), dims=sql.Literal(dims), model=sql.Literal(model))
        )


def drop_model(conn: psycopg.Connection, model: str) -> bool:
    row = conn.execute("SELECT index_name FROM embedding_models WHERE model = %s", (model,)).fetchone()
    if not row:
        return False
    with conn.transaction():
        conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(row["index_name"])))
        conn.execute("DELETE FROM embedding_models WHERE model = %s", (model,))
    return True


def vector_version(conn: psycopg.Connection) -> tuple[int, ...]:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not row:
        return ()
    return tuple(int(part) for part in row["extversion"].split(".") if part.isdigit())
