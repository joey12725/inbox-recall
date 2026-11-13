"""Settings for the search layer, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


def local_timezone() -> tzinfo:
    """RECALL_TIMEZONE (an IANA name), else the system's local zone."""
    name = _env("RECALL_TIMEZONE") or _env("TZ")
    if name:
        return ZoneInfo(name)
    return datetime.now().astimezone().tzinfo or ZoneInfo("UTC")


@dataclass
class RecallSettings:
    database_url: str | None = field(default_factory=lambda: _env("DATABASE_URL") or None)

    # Embeddings. Provider is one of: none, hash, openai, ollama, lmstudio,
    # sentence-transformers. "openai" means any OpenAI-compatible /embeddings API.
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "none").lower())
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL"))
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY") or _env("OPENAI_API_KEY"))
    embed_dimensions: int | None = field(default_factory=lambda: _int("EMBED_DIMENSIONS", 0) or None)
    embed_batch_size: int = field(default_factory=lambda: _int("EMBED_BATCH_SIZE", 64))
    embed_context_messages: int = field(default_factory=lambda: _int("EMBED_CONTEXT_MESSAGES", 2))
    embed_query_prefix: str = field(default_factory=lambda: os.environ.get("EMBED_QUERY_PREFIX", ""))
    embed_document_prefix: str = field(default_factory=lambda: os.environ.get("EMBED_DOCUMENT_PREFIX", ""))

    # Search API / MCP over HTTP.
    token: str = field(default_factory=lambda: _env("RECALL_TOKEN"))
    host: str = field(default_factory=lambda: _env("RECALL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("RECALL_PORT", 8766))
    timezone: tzinfo = field(default_factory=local_timezone)
