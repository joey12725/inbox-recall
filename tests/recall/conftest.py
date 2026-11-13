"""Fixtures for the search layer. Every message is fabricated (see inbox_recall.demo)."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from beeper_pg_sync.db import connect
from inbox_recall import demo
from inbox_recall.embedders import HashEmbedder
from inbox_recall.pipeline import EmbedPipeline
from inbox_recall.schema import migrate
from inbox_recall.search import Searcher

DSN = os.environ.get("TEST_DATABASE_URL")
TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=TZ)


@pytest.fixture
def dsn():
    if not DSN:
        pytest.skip("set TEST_DATABASE_URL (Postgres with pgvector) to run database tests")
    return DSN


@pytest.fixture
def conn(dsn):
    with connect(dsn) as c:
        c.execute("DROP SCHEMA public CASCADE")
        c.execute("CREATE SCHEMA public")
        migrate(c)
        yield c


@pytest.fixture
def embedder():
    return HashEmbedder(256)


@pytest.fixture
def archive(conn, embedder):
    """The demo archive with a small amount of filler, fully embedded."""
    demo.seed(conn, NOW, filler=400)
    EmbedPipeline(conn, embedder, batch_size=100).run()
    return conn


@pytest.fixture
def searcher(archive, embedder):
    return Searcher(archive, embedder, tz=TZ, now=NOW)


@pytest.fixture
def ids(archive):
    def lookup(*keys):
        found = demo.message_ids(archive, list(keys))
        return [found[k] for k in keys]

    return lookup
