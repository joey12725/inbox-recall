"""Embedding providers.

Every provider turns a list of strings into a list of float vectors and has a
stable `name` that identifies its vector space in the database. Switching
providers or models means a new name, so old and new vectors never mix.

- `OpenAICompatibleEmbedder`: any `/v1/embeddings` API. OpenAI, Ollama,
  LM Studio, vLLM, llama.cpp server, Text Embeddings Inference, LiteLLM...
- `SentenceTransformerEmbedder`: in-process models via sentence-transformers
  (install it separately: `uv pip install sentence-transformers`).
- `HashEmbedder`: deterministic feature hashing, no model and no network.
  Used by the test suite and the demo; it captures word and spelling overlap,
  not meaning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from .config import RecallSettings

log = logging.getLogger(__name__)

Vector = list[float]


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    provider: str
    query_prefix: str
    document_prefix: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


# -- deterministic hashing ----------------------------------------------------

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her his how i i'm if in is it it's "
    "me my of on or our she so that the their them they this to us was we were what when where which who "
    "why will with you your".split()
)


class HashEmbedder:
    """Signed feature hashing over stemmed words and character trigrams.

    Documents from the pipeline put the message on the last line and its chat
    context above it; context lines count for less, like a real model would
    weigh them.
    """

    context_weight = 0.35

    provider = "hash"

    def __init__(self, dims: int = 256, query_prefix: str = "", document_prefix: str = ""):
        self.dims = dims
        self.name = f"hash-v1@{dims}"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _features(self, text: str) -> dict[str, float]:
        features: Counter[str] = Counter()
        lines = text.lower().split("\n")
        for n, line in enumerate(lines):
            weight = 1.0 if n == len(lines) - 1 else self.context_weight
            for raw in _WORD.findall(line):
                if raw in STOPWORDS:
                    continue
                word = _stem(raw.strip("'"))
                if not word:
                    continue
                features["w:" + word] += 2 * weight
                padded = f"<{word}>"
                for i in range(len(padded) - 2):
                    features["t:" + padded[i : i + 3]] += weight
        return features

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for feature, count in self._features(text).items():
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dims
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[index] += sign * math.log1p(count)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# -- OpenAI-compatible HTTP ---------------------------------------------------

PRESETS = {
    "openai": ("https://api.openai.com/v1", "text-embedding-3-small"),
    "ollama": ("http://127.0.0.1:11434/v1", "nomic-embed-text"),
    "lmstudio": ("http://127.0.0.1:1234/v1", "text-embedding-nomic-embed-text-v1.5"),
}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        dimensions: int | None = None,
        provider: str = "openai",
        query_prefix: str = "",
        document_prefix: str = "",
        timeout: float = 60.0,
        max_retries: int = 5,
        max_chars: int = 8000,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dimensions = dimensions
        self.provider = provider
        self.name = f"{provider}:{model}" + (f"@{dimensions}" if dimensions else "")
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_chars = max_chars

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "inbox-recall"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/embeddings", data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read()[:300].decode("utf-8", "replace")
                if error.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise EmbeddingError(f"{self.base_url} returned HTTP {error.code}: {detail}") from error
                retry_after = error.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == self.max_retries:
                    raise EmbeddingError(f"cannot reach {self.base_url}: {error}") from error
                wait = delay
            log.warning("embedding request failed, retrying in %.1fs", wait)
            time.sleep(min(wait, 60.0))
            delay = min(delay * 2, 30.0)
        raise EmbeddingError("unreachable")

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        payload: dict = {
            "model": self.model,
            "input": [t[: self.max_chars] or " " for t in texts],
            "encoding_format": "float",
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        body = self._post(payload)
        data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(data)}")
        return [[float(x) for x in item["embedding"]] for item in data]


# -- sentence-transformers (optional) ------------------------------------------


class SentenceTransformerEmbedder:
    provider = "sentence-transformers"

    def __init__(self, model: str, query_prefix: str = "", document_prefix: str = "", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise EmbeddingError(
                "EMBED_PROVIDER=sentence-transformers needs the package: uv pip install sentence-transformers"
            ) from error
        self.model = SentenceTransformer(model, device=device)
        self.name = f"st:{model}"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]


# -- factory --------------------------------------------------------------------


def from_settings(settings: RecallSettings) -> Embedder | None:
    provider = settings.embed_provider
    prefixes = {"query_prefix": settings.embed_query_prefix, "document_prefix": settings.embed_document_prefix}
    if provider in ("", "none", "off"):
        return None
    if provider == "hash":
        return HashEmbedder(settings.embed_dimensions or 256, **prefixes)
    if provider in PRESETS:
        default_url, default_model = PRESETS[provider]
        return OpenAICompatibleEmbedder(
            model=settings.embed_model or default_model,
            base_url=settings.embed_base_url or default_url,
            api_key=settings.embed_api_key,
            dimensions=settings.embed_dimensions,
            provider=provider,
            **prefixes,
        )
    if provider in ("sentence-transformers", "st", "local"):
        return SentenceTransformerEmbedder(settings.embed_model or "BAAI/bge-small-en-v1.5", **prefixes)
    raise ValueError(
        f"unknown EMBED_PROVIDER {provider!r} (use openai, ollama, lmstudio, sentence-transformers or hash)"
    )


def to_pgvector(vector: Vector) -> str:
    return "[" + ",".join(format(x, ".7g") for x in vector) + "]"
