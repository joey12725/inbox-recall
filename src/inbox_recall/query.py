"""Search requests, inline operators and pagination cursors."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

Mode = Literal["hybrid", "semantic", "keyword"]
MAX_LIMIT = 100
MAX_CONTEXT = 25


@dataclass
class SearchRequest:
    query: str = ""
    sender: list[str] = field(default_factory=list)  # who wrote it; "me" = messages you sent
    person: list[str] = field(default_factory=list)  # conversations that include this person
    chat: list[str] = field(default_factory=list)  # chat title fragments or numeric conversation ids
    network: list[str] = field(default_factory=list)
    direction: str | None = None  # incoming | outgoing
    when: str | None = None  # natural language, see timeparse
    after: datetime | None = None
    before: datetime | None = None
    mode: Mode = "hybrid"
    context: int = 2
    limit: int = 10
    cursor: str | None = None

    def normalized(self) -> SearchRequest:
        req = replace(
            self,
            query=self.query.strip(),
            sender=[s.strip() for s in self.sender if s and s.strip()],
            person=[s.strip() for s in self.person if s and s.strip()],
            chat=[s.strip() for s in self.chat if s and s.strip()],
            network=[s.strip().lower() for s in self.network if s and s.strip()],
            direction=(self.direction or "").strip().lower() or None,
            limit=max(1, min(MAX_LIMIT, int(self.limit))),
            context=max(0, min(MAX_CONTEXT, int(self.context))),
        )
        if req.direction not in (None, "incoming", "outgoing"):
            raise ValueError("direction must be 'incoming' or 'outgoing'")
        if req.mode not in ("hybrid", "semantic", "keyword"):
            raise ValueError("mode must be hybrid, semantic or keyword")
        return req

    def fingerprint(self) -> str:
        data = asdict(self)
        data.pop("cursor")
        data.pop("limit")
        data.pop("context")
        return hashlib.sha1(json.dumps(data, default=str, sort_keys=True).encode()).hexdigest()[:12]


OPERATORS = {
    "from": "sender",
    "sender": "sender",
    "with": "person",
    "person": "person",
    "in": "chat",
    "chat": "chat",
    "network": "network",
    "on": "network",
    "when": "when",
    "after": "after",
    "before": "before",
    "is": "is",
}


def parse_operators(text: str) -> tuple[str, dict[str, Any]]:
    """Split `from:maya in:"book club" when:"last spring" lisbon` into text and filters.

    Unknown `word:` tokens stay in the query text, so URLs and times survive.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    words: list[str] = []
    filters: dict[str, Any] = {}
    for token in tokens:
        m = re.fullmatch(r"([a-z]+):(.+)", token, re.I)
        key = OPERATORS.get(m.group(1).lower()) if m else None
        if not m or not key:
            words.append(token)
            continue
        value = m.group(2)
        if key == "is":
            if value.lower() in ("sent", "outgoing", "mine"):
                filters["direction"] = "outgoing"
            elif value.lower() in ("received", "incoming"):
                filters["direction"] = "incoming"
            else:
                words.append(token)
        elif key in ("sender", "person", "chat", "network"):
            filters.setdefault(key, []).append(value)
        else:
            filters[key] = value
    return " ".join(words), filters


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid cursor") from error
    if not isinstance(data, dict):
        raise ValueError("invalid cursor")
    return data
