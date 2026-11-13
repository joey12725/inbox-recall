"""HTTP API and MCP-over-HTTP (stdlib only).

GET  /v1/search                    search_messages (q, sender, person, chat, network, direction,
                                   when, after, before, mode, context, limit, cursor)
GET  /v1/conversations             list_conversations (q, network, person, when, limit, cursor)
GET  /v1/conversations/{id}        get_conversation by conversation (when, limit, cursor)
GET  /v1/messages/{id}/thread      get_conversation around a message (before, after)
GET  /v1/messages?ids=1,2,3        get_messages (context)
GET  /v1/people                    find_people (q, network, limit)
GET  /v1/stats                     archive_stats
POST /mcp                          MCP Streamable HTTP (JSON responses)
GET  /healthz                      liveness, no auth

Add `format=text` to any /v1 call for the agent-ready transcript instead of JSON.
With RECALL_TOKEN set, every route but /healthz needs `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .mcp import McpServer
from .search import Searcher
from .timeparse import TimeParseError
from .tools import BY_NAME

log = logging.getLogger(__name__)
MAX_BODY = 1024 * 1024
LIST_PARAMS = {"sender", "person", "chat", "network", "ids"}
INT_PARAMS = {"context", "limit", "before", "after", "conversation_id", "message_id"}

ROUTES: list[tuple[re.Pattern[str], str, Callable[[re.Match[str]], dict[str, Any]]]] = [
    (re.compile(r"/v1/search"), "search_messages", lambda m: {}),
    (re.compile(r"/v1/conversations"), "list_conversations", lambda m: {}),
    (re.compile(r"/v1/conversations/(\d+)"), "get_conversation", lambda m: {"conversation_id": int(m.group(1))}),
    (re.compile(r"/v1/messages/(\d+)/thread"), "get_conversation", lambda m: {"message_id": int(m.group(1))}),
    (re.compile(r"/v1/messages"), "get_messages", lambda m: {}),
    (re.compile(r"/v1/people"), "find_people", lambda m: {}),
    (re.compile(r"/v1/stats"), "archive_stats", lambda m: {}),
]


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def query_arguments(tool: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key, values in qs.items():
        name = "query" if key == "q" else key
        if name in LIST_PARAMS:
            items = [v.strip() for value in values for v in value.split(",") if v.strip()]
            args[name] = [int(i) for i in items] if name == "ids" else items
        elif name in INT_PARAMS and tool != "search_messages" or name in ("context", "limit"):
            args[name] = int(values[-1])
        else:
            args[name] = values[-1]
    return args


def make_handler(searchers: Callable[[], AbstractContextManager[Searcher]], token: str):
    mcp = McpServer(searchers)

    class Handler(BaseHTTPRequestHandler):
        server_version = "inbox-recall"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: Any, content_type: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not token:
                origin = self.headers.get("Origin")
                return not origin or is_loopback(urlsplit(origin).hostname or "")
            header = self.headers.get("Authorization", "")
            return header.startswith("Bearer ") and hmac.compare_digest(header[7:].encode(), token.encode())

        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            if url.path == "/healthz":
                return self._send(HTTPStatus.OK, {"ok": True})
            if not self._authorized():
                return self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            if url.path == "/mcp":
                self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
                self.send_header("Allow", "POST")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            for pattern, name, fixed in ROUTES:
                match = pattern.fullmatch(url.path)
                if match:
                    return self._tool(name, url.query, fixed(match))
            return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _tool(self, name: str, query: str, fixed: dict[str, Any]) -> None:
            tool = BY_NAME[name]
            try:
                args = {**query_arguments(name, parse_qs(query)), **fixed}
                output = args.pop("format", "json")
                with searchers() as searcher:
                    result = tool.run(searcher, args)
            except (ValueError, TypeError, TimeParseError) as error:
                return self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            if output == "text":
                return self._send(HTTPStatus.OK, tool.text(result), "text/plain")
            return self._send(HTTPStatus.OK, result)

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/mcp":
                return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if not self._authorized():
                return self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
            reply = mcp.handle_text(self.rfile.read(length).decode("utf-8", "replace"))
            if reply is None:
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            return self._send(HTTPStatus.OK, reply)

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "sessions are not used"})

    return Handler


def serve(
    searchers: Callable[[], AbstractContextManager[Searcher]], token: str, host: str, port: int
) -> ThreadingHTTPServer:
    if not token and not is_loopback(host):
        raise RuntimeError(f"refusing to listen on {host} without RECALL_TOKEN; set one or bind to 127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), make_handler(searchers, token))
    log.info("inbox-recall API on http://%s:%s (MCP at /mcp)", host, httpd.server_address[1])
    return httpd
