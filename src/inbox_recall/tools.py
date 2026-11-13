"""The agent-facing tools, shared by the MCP server, the HTTP API and the CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from . import render
from .query import SearchRequest
from .search import Searcher

INSTRUCTIONS = """\
These tools search the user's own message history: every chat synced from Beeper \
(WhatsApp, Signal, Telegram, iMessage, Discord, Instagram and more) plus Instagram DMs \
from the browser extension, in one Postgres archive.

How to use them well:
- Start with search_messages. Put the topic in `query` and move people, chats, networks \
and dates into filters: "what did Maya say about the lease last spring" becomes \
query="lease", sender=["Maya"], when="last spring".
- `sender` is who wrote the message ("me" = the user). `person` limits to conversations \
that include someone. `when` takes plain English ("last spring", "march 2024", "past 2 weeks").
- Results are conversation excerpts. Lines marked → are matches; the rest is context. \
Use get_conversation(message_id=...) to read more of a thread.
- Use find_people or list_conversations when a name or chat is ambiguous.
- Always cite the messages you rely on by id, like [#4812], with sender and date.
- Leave `query` empty to list every message matching the filters, newest first."""

FORMAT = {
    "type": "string",
    "enum": ["text", "json"],
    "default": "text",
    "description": "text: compact, citation-tagged transcript (default). json: the full structured result.",
}
STRING_LIST = {"type": "array", "items": {"type": "string"}}


@dataclass
class Tool:
    name: str
    title: str
    description: str
    schema: dict[str, Any]
    run: Callable[[Searcher, dict[str, Any]], dict[str, Any]]
    text: Callable[[dict[str, Any]], str]

    def spec(self) -> dict[str, Any]:
        schema = {**self.schema, "properties": {**self.schema["properties"], "format": FORMAT}}
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": schema,
            "annotations": {"title": self.title, "readOnlyHint": True, "openWorldHint": False},
        }


def _list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _search(searcher: Searcher, a: dict[str, Any]) -> dict[str, Any]:
    req = SearchRequest(
        query=str(a.get("query") or ""),
        sender=_list(a.get("sender")),
        person=_list(a.get("person")),
        chat=_list(a.get("chat")),
        network=_list(a.get("network")),
        direction=a.get("direction"),
        when=a.get("when"),
        after=_date(a.get("after")),
        before=_date(a.get("before")),
        mode=a.get("mode") or "hybrid",
        context=int(a.get("context", 2)),
        limit=int(a.get("limit", 10)),
        cursor=a.get("cursor"),
    )
    return searcher.search(req)


TOOLS = [
    Tool(
        name="search_messages",
        title="Search messages",
        description=(
            "Hybrid semantic + keyword + fuzzy search across every synced chat and network. Returns ranked "
            "matches as conversation excerpts with surrounding messages, each line tagged with a citable "
            "message id. Filter by sender, person, chat, network, direction and time (`when` accepts plain "
            "English such as 'last spring' or 'march 2024'). Leave query empty to list messages matching the "
            "filters, newest first. Paginate with next_cursor."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for: a topic, phrase or question."},
                "sender": {**STRING_LIST, "description": "Who wrote the message (name or handle). 'me' = the user."},
                "person": {**STRING_LIST, "description": "Only conversations that include this person."},
                "chat": {**STRING_LIST, "description": "Chat title fragment or numeric conversation id."},
                "network": {**STRING_LIST, "description": "e.g. whatsapp, signal, instagram, imessage, telegram."},
                "direction": {"type": "string", "enum": ["incoming", "outgoing"]},
                "when": {"type": "string", "description": "Time range in plain English, e.g. 'last summer'."},
                "after": {"type": "string", "description": "ISO date or datetime (inclusive)."},
                "before": {"type": "string", "description": "ISO date or datetime (exclusive)."},
                "mode": {"type": "string", "enum": ["hybrid", "semantic", "keyword"], "default": "hybrid"},
                "context": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 25,
                    "default": 2,
                    "description": "Messages of context before and after each match.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "cursor": {"type": "string", "description": "next_cursor from a previous call with the same query."},
            },
            "additionalProperties": False,
        },
        run=_search,
        text=render.search,
    ),
    Tool(
        name="get_conversation",
        title="Read a conversation",
        description=(
            "Reconstruct a thread. Pass message_id to read the messages around it (before/after counts), "
            "conversation_id with `when` to read a time range, or conversation_id alone for the latest "
            "messages. Returns cursors to keep reading older or newer messages."
        ),
        schema={
            "type": "object",
            "properties": {
                "message_id": {"type": "integer"},
                "conversation_id": {"type": "integer"},
                "before": {"type": "integer", "minimum": 0, "maximum": 200, "default": 10},
                "after": {"type": "integer", "minimum": 0, "maximum": 200, "default": 10},
                "when": {"type": "string", "description": "With conversation_id: plain-English time range."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                "cursor": {"type": "string"},
            },
            "additionalProperties": False,
        },
        run=lambda s, a: s.conversation(
            conversation_id=a.get("conversation_id"),
            message_id=a.get("message_id"),
            before=a.get("before", 10),
            after=a.get("after", 10),
            when=a.get("when"),
            limit=a.get("limit", 50),
            cursor=a.get("cursor"),
        ),
        text=render.conversation,
    ),
    Tool(
        name="get_messages",
        title="Get messages by id",
        description="Fetch specific messages by id (to verify a citation), optionally with context.",
        schema={
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 100},
                "context": {"type": "integer", "minimum": 0, "maximum": 25, "default": 0},
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
        run=lambda s, a: s.get_messages(a.get("ids") or [], int(a.get("context", 0))),
        text=render.messages,
    ),
    Tool(
        name="list_conversations",
        title="List conversations",
        description=(
            "List chats, most recently active first, with message counts, date span and the people in them. "
            "Filter by title, network, a person in the chat, or activity in a time range."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Chat title fragment."},
                "network": STRING_LIST,
                "person": STRING_LIST,
                "when": {"type": "string", "description": "Only chats with messages in this time range."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                "cursor": {"type": "string"},
            },
            "additionalProperties": False,
        },
        run=lambda s, a: s.conversations(
            query=a.get("query") or "",
            network=_list(a.get("network")),
            person=_list(a.get("person")),
            when=a.get("when"),
            limit=a.get("limit", 20),
            cursor=a.get("cursor"),
        ),
        text=render.conversations,
    ),
    Tool(
        name="find_people",
        title="Find people",
        description=(
            "Look up people by name or handle across every network. One person on WhatsApp and Instagram "
            "shows up once, with message counts per network. Use it to resolve who someone is before filtering."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "network": STRING_LIST,
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "additionalProperties": False,
        },
        run=lambda s, a: s.people(a.get("query") or "", _list(a.get("network")), a.get("limit", 20)),
        text=render.people,
    ),
    Tool(
        name="archive_stats",
        title="Archive overview",
        description="Networks, message counts, date coverage and embedding coverage of the archive.",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        run=lambda s, a: s.stats(),
        text=render.stats,
    ),
]

BY_NAME = {tool.name: tool for tool in TOOLS}
