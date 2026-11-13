"""Compact, citation-first text for agents (and humans in a terminal).

Every line that quotes a message starts with its id, `[#4812]`, so an agent can
cite it, fetch it again with get_messages, or open the thread around it.
Matches are marked with `→`; surrounding lines are context.
"""

from __future__ import annotations

from typing import Any

CONTEXT_CLIP = 400


def _when(iso: str | None) -> str:
    if not iso:
        return "????-??-?? ??:??"
    return iso[:16].replace("T", " ")


def _clip(text: str, limit: int | None) -> str:
    text = text.replace("\r", "")
    if limit and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("\n", "\n      ")


def message_line(m: dict[str, Any], clip: int | None = None) -> str:
    marker = "→" if m.get("hit") else " "
    who = m["sender"]
    extras = []
    if m.get("attachments"):
        extras.append("attachments: " + ", ".join(m["attachments"]))
    if m.get("reactions"):
        extras.append("reactions: " + " ".join(m["reactions"]))
    if m.get("edited"):
        extras.append("edited")
    if m.get("time_precision") == "approximate":
        extras.append("approx. time")
    tail = f"  ({'; '.join(extras)})" if extras else ""
    line = (
        f"{marker} [#{m['id']}] {_when(m['sent_at'])} {who}: {_clip(m['text'], None if m.get('hit') else clip)}{tail}"
    )
    reply = m.get("in_reply_to")
    if reply:
        quoted = _clip(reply["text"], 160)
        line += f"\n      ↳ replying to [#{reply['id']}] {reply['sender']}: {quoted}"
    return line


def chat_heading(chat: dict[str, Any]) -> str:
    title = chat.get("title") or "untitled chat"
    return f"## {title} · {chat['network']} {chat.get('kind') or ''} · conversation {chat['id']}".replace("  ", " ")


def search(result: dict[str, Any]) -> str:
    lines = []
    count = len(result["hits"])
    if result["mode"] == "browse":
        lines.append(f"# {count} messages, newest first")
    else:
        start = (result.get("offset") or 0) + 1
        span = f"{start}-{start + count - 1}" if count else "0"
        lines.append(
            f'# Results {span} of {result["total_candidates"]} for "{result["query"]}" ({result["mode"]} search)'
        )
    if result["filters"]:
        lines.append("filters: " + " · ".join(f"{k}={_fmt(v)}" for k, v in result["filters"].items()))
    for note in result.get("notes", []):
        lines.append(f"note: {note}")
    if not result["excerpts"]:
        lines.append("\nNo matching messages.")
    for excerpt in result["excerpts"]:
        lines.append("")
        lines.append(chat_heading(excerpt["conversation"]))
        lines.extend(message_line(m, CONTEXT_CLIP) for m in excerpt["messages"])
    lines.append("")
    if result.get("next_cursor"):
        lines.append(f"More results: pass cursor={result['next_cursor']}")
    lines.append("Cite messages by id, e.g. [#123]. Open a thread with get_conversation(message_id=...).")
    return "\n".join(lines)


def conversation(result: dict[str, Any]) -> str:
    lines = [chat_heading(result["conversation"])]
    lines.extend(message_line(m) for m in result["messages"])
    if not result["messages"]:
        lines.append("(no messages in this range)")
    if result.get("older_cursor"):
        lines.append(f"\nOlder messages: cursor={result['older_cursor']}")
    if result.get("newer_cursor"):
        lines.append(f"Newer messages: cursor={result['newer_cursor']}")
    return "\n".join(lines)


def messages(result: dict[str, Any]) -> str:
    lines = []
    for excerpt in result["excerpts"]:
        lines.append(chat_heading(excerpt["conversation"]))
        lines.extend(message_line(m) for m in excerpt["messages"])
        lines.append("")
    if result.get("missing"):
        lines.append("not found: " + ", ".join(f"#{i}" for i in result["missing"]))
    return "\n".join(lines).strip()


def conversations(result: dict[str, Any]) -> str:
    lines = []
    for c in result["conversations"]:
        people = ", ".join(c["people"]) if c["people"] else "-"
        lines.append(
            f"- conversation {c['id']}: {c['title'] or 'untitled'} ({c['network']} {c['kind']}) · "
            f"{c['messages']} messages · {_when(c['first_message'])[:10]} to {_when(c['last_message'])[:10]} · "
            f"people: {people}"
        )
    if not lines:
        lines.append("No matching conversations.")
    if result.get("next_cursor"):
        lines.append(f"More: cursor={result['next_cursor']}")
    return "\n".join(lines)


def people(result: dict[str, Any]) -> str:
    lines = []
    for p in result["people"]:
        nets = ", ".join(f"{n} {c}" for n, c in p["networks"].items())
        handles = f" ({', '.join(p['handles'])})" if p["handles"] else ""
        lines.append(
            f"- {p['name']}{handles}: {p['messages']} messages · {nets} · "
            f"{_when(p['first_message'])[:10]} to {_when(p['last_message'])[:10]}"
        )
    return "\n".join(lines) if lines else "No matching people."


def stats(result: dict[str, Any]) -> str:
    lines = [f"Archive as of {result['now']} ({result['timezone']})"]
    for n in result["networks"]:
        lines.append(
            f"- {n['network']}: {n['messages']} messages in {n['conversations']} conversations, "
            f"{_when(n['first_message'])[:10]} to {_when(n['last_message'])[:10]}"
        )
    lines.append(f"Searchable messages: {result['searchable_messages']}")
    for m in result["embedding_models"]:
        flag = " (active)" if m["active"] else ""
        lines.append(f"- embeddings {m['model']}{flag}: {m['embedded']} vectors, {m['coverage']:.1%} coverage")
    if not result["active_model"]:
        lines.append("Semantic search is off (EMBED_PROVIDER not set); keyword and fuzzy search still work.")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)
