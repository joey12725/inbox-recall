"""A dependency-free Model Context Protocol server.

Implements the tool subset of MCP over JSON-RPC 2.0:
`initialize`, `ping`, `tools/list` and `tools/call`. Two transports share it:

- stdio (`inbox-recall mcp`): newline-delimited JSON on stdin/stdout, which is
  what Claude Code, Claude Desktop, Cursor and most agent runtimes launch.
- Streamable HTTP (`POST /mcp` on `inbox-recall serve`), answering with plain
  JSON responses, for agents that run on another machine.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import IO, Any

from . import __version__
from .search import Searcher
from .timeparse import TimeParseError
from .tools import BY_NAME, INSTRUCTIONS, TOOLS

log = logging.getLogger(__name__)

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SearcherFactory = Callable[[], AbstractContextManager[Searcher]]

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class McpServer:
    def __init__(self, searchers: SearcherFactory):
        self.searchers = searchers

    # -- JSON-RPC -------------------------------------------------------------

    def handle(self, message: Any) -> Any:
        """Handle one decoded JSON-RPC message or batch. Returns the reply, or None for notifications."""
        if isinstance(message, list):
            replies = [r for r in (self._one(m) for m in message) if r is not None]
            return replies or None
        return self._one(message)

    def handle_text(self, text: str) -> str | None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps(_error(None, PARSE_ERROR, "parse error"))
        reply = self.handle(message)
        return json.dumps(reply, default=str, ensure_ascii=False) if reply is not None else None

    def _one(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "invalid request")
        msg_id = message.get("id")
        method = message.get("method")
        if method is None:
            return None  # a response from the client; we never send requests
        is_notification = "id" not in message
        try:
            result = self._dispatch(method, message.get("params") or {})
        except RpcError as error:
            return None if is_notification else _error(msg_id, error.code, str(error))
        except Exception:  # noqa: BLE001 - report, never crash the transport
            log.exception("MCP %s failed", method)
            return None if is_notification else _error(msg_id, INTERNAL_ERROR, "internal error")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "inbox-recall", "title": "inbox-recall", "version": __version__},
                "instructions": INSTRUCTIONS,
            }
        if method == "ping":
            return {}
        if method.startswith("notifications/"):
            return None
        if method == "tools/list":
            return {"tools": [tool.spec() for tool in TOOLS]}
        if method == "tools/call":
            return self.call_tool(params.get("name"), params.get("arguments") or {})
        raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")

    # -- tools ----------------------------------------------------------------

    def call_tool(self, name: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = BY_NAME.get(name) if isinstance(name, str) else None
        if not tool:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise RpcError(INVALID_PARAMS, "arguments must be an object")
        output_format = arguments.pop("format", "text")
        try:
            with self.searchers() as searcher:
                result = tool.run(searcher, arguments)
        except (ValueError, TypeError, TimeParseError) as error:
            return {"content": [{"type": "text", "text": f"error: {error}"}], "isError": True}
        text = json.dumps(result, default=str, ensure_ascii=False) if output_format == "json" else tool.text(result)
        return {"content": [{"type": "text", "text": text}], "isError": False}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _lines(stream: IO[str]) -> Iterator[str]:
    for line in stream:
        line = line.strip()
        if line:
            yield line


def serve_stdio(server: McpServer, stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    for line in _lines(stdin):
        reply = server.handle_text(line)
        if reply is not None:
            stdout.write(reply + "\n")
            stdout.flush()
