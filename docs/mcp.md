# Connecting agents

inbox-recall speaks the [Model Context Protocol](https://modelcontextprotocol.io/) two ways:

- **stdio**: `inbox-recall mcp`. The client launches the process and talks over stdin/stdout. Best when the agent runs on the same machine as the database (or can reach it).
- **Streamable HTTP**: `POST /mcp` on `inbox-recall serve`. Best for agents on another box, in a container, or on a home server. Responses are plain JSON; no session state is kept.

Both expose the same six read-only tools and the same server instructions, which tell the model how to split a question into query + filters and to cite `[#id]`s.

## Claude Code

```bash
claude mcp add inbox-recall \
  -e DATABASE_URL=postgresql://sync:sync@localhost:5432/messages \
  -e EMBED_PROVIDER=ollama \
  -e EMBED_QUERY_PREFIX="search_query: " -e EMBED_DOCUMENT_PREFIX="search_document: " \
  -e RECALL_TIMEZONE=America/New_York \
  -- uv --directory /path/to/inbox-recall run inbox-recall mcp
```

Over HTTP instead:

```bash
claude mcp add --transport http inbox-recall http://home-server.local:8766/mcp \
  --header "Authorization: Bearer $RECALL_TOKEN"
```

## Claude Desktop, Cursor, Windsurf, Zed and other JSON configs

```json
{
  "mcpServers": {
    "inbox-recall": {
      "command": "uv",
      "args": ["--directory", "/path/to/inbox-recall", "run", "inbox-recall", "mcp"],
      "env": {
        "DATABASE_URL": "postgresql://sync:sync@localhost:5432/messages",
        "EMBED_PROVIDER": "ollama",
        "EMBED_QUERY_PREFIX": "search_query: ",
        "EMBED_DOCUMENT_PREFIX": "search_document: "
      }
    }
  }
}
```

The query-time embedder has to match the one that built the index: same `EMBED_PROVIDER`, `EMBED_MODEL`, `EMBED_DIMENSIONS` and prefixes. If they don't match, search still works but runs keyword-only and says so in a note.

## OpenClaw, Hermes Agent and other self-hosted runtimes

Most agent runtimes accept either a command (stdio) or a URL (HTTP) for an MCP server. For a long-running agent on a server, run the HTTP transport next to the database:

```bash
docker compose --profile daemon up -d    # db + sync daemon + `inbox-recall serve --embed`
```

Set `RECALL_TOKEN` in `.env` first; the server refuses to listen beyond loopback without one. Point the agent at `http://<host>:8766/mcp` with the header `Authorization: Bearer <token>`. If the runtime only speaks plain HTTP tools, the same calls are on the REST API:

```bash
curl -H "Authorization: Bearer $RECALL_TOKEN" \
  "http://localhost:8766/v1/search?q=lease+renewal&person=priya&when=this+year&format=text"
```

## Tools

| Tool | Key arguments |
| --- | --- |
| `search_messages` | `query`, `sender[]` (`me` = you), `person[]`, `chat[]`, `network[]`, `direction`, `when`, `after`, `before`, `mode` (`hybrid`/`semantic`/`keyword`), `context` (0-25), `limit` (1-100), `cursor` |
| `get_conversation` | `message_id` + `before`/`after`, or `conversation_id` + `when`, or `cursor`; `limit` |
| `get_messages` | `ids[]`, `context` |
| `list_conversations` | `query` (title), `network[]`, `person[]`, `when`, `limit`, `cursor` |
| `find_people` | `query`, `network[]`, `limit` |
| `archive_stats` | none |

Every tool also takes `format`: `text` (default) returns the compact transcript below, `json` returns the full structured result.

```text
# Results 1-1 of 14 for "lease renewal" (hybrid search)
filters: person=priya

## Priya Nair · signal dm · conversation 3
  [#1102] 2026-09-04 18:40 Priya Nair: got an email from Mr. Alvarez, rent goes up to 2,150 when we renew
  [#1103] 2026-09-04 18:42 me: ugh that's almost 9% more
→ [#1104] 2026-09-04 18:45 Priya Nair: renewal paperwork is due by the 30th or we go month to month
  [#1105] 2026-09-04 18:47 Priya Nair: I think we should sign, moving would cost more than the difference
```

Lines marked `→` are matches; the rest is context. Overlapping windows from the same chat are merged into one excerpt.

## Prompting tips

The server sends these instructions on `initialize`, and they're worth repeating in a system prompt if your client drops them:

- Put the topic in `query` and move names, chats, networks and dates into filters. "What did Maya say about the lease last spring" is `query: "lease", sender: ["Maya"], when: "last spring"`.
- Use `person` for "conversations with X" and `sender` for "things X said".
- When a hit looks promising, call `get_conversation(message_id=...)` to read what came before and after.
- Resolve ambiguous names with `find_people` before filtering.
- Cite message ids with sender and date in the final answer.

## Privacy

The archive and the index stay in your Postgres. What leaves the machine is up to you: with a local embedding model (Ollama, LM Studio, sentence-transformers) and a local LLM, nothing does. With a hosted model, the agent sees only the excerpts its tool calls return. Every tool is read-only; nothing can send, edit or delete messages.
