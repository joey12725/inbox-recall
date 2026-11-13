"""MCP (stdio + HTTP), the REST API and the CLI, end to end on the fabricated archive."""

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from conftest import NOW, TZ

from inbox_recall import eval as bench
from inbox_recall.api import serve
from inbox_recall.cli import main
from inbox_recall.mcp import McpServer
from inbox_recall.search import Searcher
from inbox_recall.tools import TOOLS


@pytest.fixture
def factory(archive, embedder):
    @contextmanager
    def open_searcher():
        yield Searcher(archive, embedder, tz=TZ, now=NOW)

    return open_searcher


def rpc(server, method, params=None, msg_id=1):
    message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if msg_id is not None:
        message["id"] = msg_id
    return server.handle(message)


def test_mcp_handshake_and_tool_listing(factory):
    server = McpServer(factory)
    init = rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert "cite" in init["result"]["instructions"].lower()
    assert rpc(server, "notifications/initialized", msg_id=None) is None
    old = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert old["result"]["protocolVersion"] == "2025-06-18"

    tools = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == [t.name for t in TOOLS]
    search = tools[0]
    assert search["annotations"]["readOnlyHint"] is True
    assert {"query", "sender", "when", "cursor", "format"} <= set(search["inputSchema"]["properties"])

    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "nope")["error"]["code"] == -32601
    assert rpc(server, "tools/call", {"name": "drop_tables"})["error"]["code"] == -32602


def test_mcp_tool_calls_return_citable_text_and_json(factory, ids):
    server = McpServer(factory)
    reply = rpc(server, "tools/call", {"name": "search_messages", "arguments": {"query": "timing belt", "limit": 1}})
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is False
    assert f"→ [#{ids('im-sam:car-1')[0]}]" in text and "Sam Rivera: mechanic called" in text

    as_json = rpc(
        server,
        "tools/call",
        {"name": "find_people", "arguments": {"query": "maya", "format": "json"}},
    )
    people = json.loads(as_json["result"]["content"][0]["text"])["people"]
    assert people[0]["name"] == "Maya Chen"

    thread = rpc(
        server, "tools/call", {"name": "get_conversation", "arguments": {"message_id": ids("im-sam:car-1")[0]}}
    )
    assert "car will be ready friday afternoon" in thread["result"]["content"][0]["text"]

    bad = rpc(server, "tools/call", {"name": "search_messages", "arguments": {"query": "x", "when": "whenever"}})
    assert bad["result"]["isError"] is True and "can't read" in bad["result"]["content"][0]["text"]

    batch = server.handle(
        [{"jsonrpc": "2.0", "id": 7, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/x"}]
    )
    assert batch == [{"jsonrpc": "2.0", "id": 7, "result": {}}]
    assert json.loads(server.handle_text("{oops"))["error"]["code"] == -32700


def test_mcp_over_stdio_subprocess(archive, dsn):
    env = {
        **os.environ,
        "DATABASE_URL": dsn,
        "EMBED_PROVIDER": "hash",
        "RECALL_TIMEZONE": "America/Chicago",
    }
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "archive_stats", "arguments": {}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "inbox_recall", "mcp"],
        input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert "embeddings hash-v1@256 (active)" in replies[1]["result"]["content"][0]["text"]


@contextmanager
def running(factory, token):
    httpd = serve(factory, token, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def request(url, data=None, token=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            return resp.status, body
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_http_api_and_mcp_endpoint(factory, ids):
    with running(factory, "demo-recall-token") as base:
        assert request(f"{base}/healthz")[0] == 200
        assert request(f"{base}/v1/stats")[0] == 401
        assert request(f"{base}/v1/stats", token="wrong")[0] == 401

        status, body = request(
            f"{base}/v1/search?q=lisbon+flights&sender=Maya&when=last+spring&limit=3", token="demo-recall-token"
        )
        assert status == 200
        result = json.loads(body)
        assert ids("wa-maya:lisbon-1")[0] in [h["id"] for h in result["hits"]]

        status, text = request(f"{base}/v1/search?q=timing+belt&format=text&limit=1", token="demo-recall-token")
        assert status == 200 and text.startswith("# Results 1-1")

        car = ids("im-sam:car-1")[0]
        status, body = request(f"{base}/v1/messages/{car}/thread?before=0&after=2", token="demo-recall-token")
        assert [m["text"] for m in json.loads(body)["messages"]][-1] == "car will be ready friday afternoon"
        status, body = request(f"{base}/v1/messages?ids={car},999999", token="demo-recall-token")
        assert json.loads(body)["missing"] == [999999]
        status, body = request(f"{base}/v1/people?q=priya", token="demo-recall-token")
        assert json.loads(body)["people"][0]["name"] == "Priya Nair"
        status, body = request(f"{base}/v1/conversations?network=discord", token="demo-recall-token")
        conv = json.loads(body)["conversations"][0]
        status, body = request(f"{base}/v1/conversations/{conv['id']}?limit=3", token="demo-recall-token")
        assert len(json.loads(body)["messages"]) == 3
        assert request(f"{base}/v1/search?q=x&when=whenever", token="demo-recall-token")[0] == 400

        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
        assert request(f"{base}/mcp", init)[0] == 401
        status, body = request(f"{base}/mcp", init, token="demo-recall-token")
        assert status == 200 and json.loads(body)["result"]["serverInfo"]["name"] == "inbox-recall"
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert request(f"{base}/mcp", note, token="demo-recall-token")[0] == 202
        assert request(f"{base}/mcp", token="demo-recall-token")[0] == 405


def test_http_without_token_is_loopback_only(factory):
    with pytest.raises(RuntimeError, match="RECALL_TOKEN"):
        serve(factory, "", "0.0.0.0", 0)
    with running(factory, "") as base:
        assert request(f"{base}/v1/stats")[0] == 200
        assert request(f"{base}/v1/stats", headers={"Origin": "https://evil.example"})[0] == 401
        assert request(f"{base}/v1/stats", headers={"Origin": "http://localhost:3000"})[0] == 200


def test_cli_search_and_eval(archive, dsn, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("RECALL_TIMEZONE", "America/Chicago")
    assert main(["search", 'from:Sam "timing belt"', "--limit", "1", "--context", "0"]) == 0
    out = capsys.readouterr().out
    assert "filters: sender=Sam" in out and "mechanic called" in out

    assert main(["eval"]) == 2  # no demo seed marker in this database
    assert "inbox-recall demo" in capsys.readouterr().err


def test_benchmark_hybrid_beats_single_rankers(archive, embedder):
    report = bench.run(archive, embedder, NOW)
    results = report["results"]
    assert report["cases"] == len(bench.CASES)
    assert results["hybrid"]["all"]["recall@10"] >= results["keyword"]["all"]["recall@10"]
    assert results["hybrid"]["all"]["recall@10"] >= results["semantic"]["all"]["recall@10"]
    assert results["hybrid"]["filtered"]["recall@10"] == 1.0
    assert results["hybrid"]["keyword"]["recall@10"] == 1.0
    assert results["hybrid"]["fuzzy"]["recall@10"] == 1.0
    assert "| hybrid |" in bench.table(report)
