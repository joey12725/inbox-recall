"""Unit tests that need no database."""

import json
import math
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inbox_recall import render
from inbox_recall.config import RecallSettings
from inbox_recall.embedders import (
    EmbeddingError,
    HashEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    from_settings,
    to_pgvector,
)
from inbox_recall.pipeline import document_text
from inbox_recall.query import SearchRequest, decode_cursor, encode_cursor, parse_operators


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_deterministic_normalized_and_lexical():
    e = HashEmbedder(128)
    a, b, c = e.embed(["booked the flights to Lisbon", "booked the flights to Lisbon", "my bowl came out lopsided"])
    assert a == b and len(a) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    query = e.embed(["lisbon flight"])[0]
    assert cosine(query, a) > cosine(query, c)
    assert e.name == "hash-v1@128"


def test_hash_embedder_weighs_the_message_line_over_its_context():
    e = HashEmbedder()
    doc = e.embed(["whatsapp chat: Maya\nMaya: lisbon trip planning\nme: the pottery class was great"])[0]
    pottery, lisbon = e.embed(["pottery class", "lisbon trip planning"])
    assert cosine(pottery, doc) > cosine(lisbon, doc)


def test_document_text_includes_chat_and_context():
    row = {"title": "Maya Chen", "network": "whatsapp", "sender": None, "direction": "outgoing", "body": "booked!"}
    context = [{"sender": "Maya Chen", "direction": "incoming", "body": "did you get   the Lisbon\nflights?"}]
    assert document_text(row, context) == (
        "whatsapp chat: Maya Chen\nMaya Chen: did you get the Lisbon flights?\nme: booked!"
    )


def test_operators_split_from_query_text():
    text, filters = parse_operators('from:maya in:"book club" when:"last spring" is:received lisbon https://x.test/a')
    assert text == "lisbon https://x.test/a"
    assert filters == {"sender": ["maya"], "chat": ["book club"], "when": "last spring", "direction": "incoming"}


def test_cursor_round_trip_and_tampering():
    token = encode_cursor({"o": 20, "k": 200, "f": "abc"})
    assert decode_cursor(token) == {"o": 20, "k": 200, "f": "abc"}
    with pytest.raises(ValueError):
        decode_cursor("not-a-cursor!!")


def test_request_normalization_clamps_and_validates():
    req = SearchRequest(query="  hi ", limit=1000, context=-3, network=["WhatsApp "], sender=["", "Maya"]).normalized()
    assert (req.query, req.limit, req.context, req.network, req.sender) == ("hi", 100, 0, ["whatsapp"], ["Maya"])
    with pytest.raises(ValueError):
        SearchRequest(direction="sideways").normalized()
    assert SearchRequest(query="a", limit=5).fingerprint() == SearchRequest(query="a", limit=50).fingerprint()
    assert SearchRequest(query="a").fingerprint() != SearchRequest(query="b").fingerprint()


def test_to_pgvector_format():
    assert to_pgvector([1.0, -0.25, 1e-9]) == "[1,-0.25,1e-09]"


class FakeEmbeddingsAPI(BaseHTTPRequestHandler):
    calls: list = []
    fail_first = 0

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append({"body": body, "auth": self.headers.get("Authorization")})
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        dims = body.get("dimensions", 3)
        data = [{"index": i, "embedding": [float(len(t))] + [0.0] * (dims - 1)} for i, t in enumerate(body["input"])]
        payload = json.dumps({"data": list(reversed(data))}).encode()  # out of order on purpose
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_api():
    FakeEmbeddingsAPI.calls = []
    FakeEmbeddingsAPI.fail_first = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddingsAPI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_openai_compatible_embedder_batches_orders_and_retries(fake_api):
    FakeEmbeddingsAPI.fail_first = 1
    e = OpenAICompatibleEmbedder("demo-embed", fake_api, api_key="demo-key", dimensions=4, provider="ollama")
    vectors = e.embed(["a", "bbb"])
    assert vectors == [[1.0, 0, 0, 0], [3.0, 0, 0, 0]]
    assert e.name == "ollama:demo-embed@4"
    assert len(FakeEmbeddingsAPI.calls) == 2  # one 429, one success
    sent = FakeEmbeddingsAPI.calls[-1]
    assert sent["auth"] == "Bearer demo-key"
    assert sent["body"] == {"model": "demo-embed", "input": ["a", "bbb"], "encoding_format": "float", "dimensions": 4}


def test_openai_compatible_embedder_gives_up_with_a_clear_error():
    e = OpenAICompatibleEmbedder("m", "http://127.0.0.1:9/v1", max_retries=0, timeout=1)
    with pytest.raises(EmbeddingError, match="cannot reach"):
        e.embed(["x"])


def test_sentence_transformers_adapter(monkeypatch):
    class FakeModel:
        def __init__(self, name, device=None):
            self.name = name

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            assert normalize_embeddings
            return [[1.0, float(len(t))] for t in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    e = SentenceTransformerEmbedder("demo/mini", query_prefix="query: ")
    assert e.name == "st:demo/mini" and e.query_prefix == "query: "
    assert e.embed(["ab"]) == [[1.0, 2.0]]


def test_provider_presets(monkeypatch):
    for key in ("EMBED_MODEL", "EMBED_BASE_URL", "EMBED_DIMENSIONS", "EMBED_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    e = from_settings(RecallSettings())
    assert e.base_url == "http://127.0.0.1:11434/v1" and e.name == "ollama:nomic-embed-text"
    monkeypatch.setenv("EMBED_PROVIDER", "none")
    assert from_settings(RecallSettings()) is None
    monkeypatch.setenv("EMBED_PROVIDER", "carrier-pigeon")
    with pytest.raises(ValueError):
        from_settings(RecallSettings())


def test_render_search_marks_hits_and_cites_ids():
    result = {
        "query": "lisbon",
        "mode": "hybrid",
        "filters": {"sender": ["Maya"]},
        "total_candidates": 1,
        "offset": 0,
        "hits": [{"rank": 1, "id": 2}],
        "notes": [],
        "next_cursor": None,
        "excerpts": [
            {
                "conversation": {"id": 7, "title": "Maya Chen", "network": "whatsapp", "kind": "dm"},
                "messages": [
                    {"id": 1, "sender": "me", "sent_at": "2026-03-25T19:00-05:00", "text": "trip?"},
                    {
                        "id": 2,
                        "sender": "Maya Chen",
                        "sent_at": "2026-03-25T19:02-05:00",
                        "text": "Lisbon in May!",
                        "hit": 1,
                        "reactions": ["❤️"],
                        "in_reply_to": {"id": 1, "sender": "me", "sent_at": "x", "text": "trip?"},
                    },
                ],
            }
        ],
    }
    text = render.search(result)
    assert '# Results 1-1 of 1 for "lisbon" (hybrid search)' in text
    assert "filters: sender=Maya" in text
    assert "## Maya Chen · whatsapp dm · conversation 7" in text
    assert "  [#1] 2026-03-25 19:00 me: trip?" in text
    assert "→ [#2] 2026-03-25 19:02 Maya Chen: Lisbon in May!  (reactions: ❤️)" in text
    assert "↳ replying to [#1] me: trip?" in text
