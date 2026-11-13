<img src="assets/logo.svg" width="88" height="88" alt="Chat bubble with a magnifying glass">

# inbox-recall

**Every conversation you've ever had, searchable by your own agents.**

inbox-recall gives self-hosted AI agents a memory of your messages. It syncs WhatsApp, Signal, Telegram, iMessage, Discord, Slack, Instagram and the rest of the networks [Beeper](https://www.beeper.com/) bridges into one Postgres database on your machine, indexes them for hybrid semantic + keyword search with pgvector, and serves that to Claude Code, Claude Desktop, Cursor, OpenClaw, Hermes Agent or any other MCP client.

Ask *"what did Priya say about the lease renewal?"* and your agent gets back the actual thread: ranked excerpts with the messages around each hit, who said what, when, and a message id to cite. Assistants are starting to search your texts for you. inbox-recall is that idea built for self-hosted agents and for archives that span every network and many years.

<p align="center"><img src="assets/architecture.svg" alt="Beeper and the Chrome extension sync into Postgres; an embedding worker adds pgvector vectors; inbox-recall serves hybrid search to agents over MCP" width="900"></p>

## Ask your agent

A Claude Code session against the fabricated demo archive that ships with the repo (every person and message in it is made up):

```text
> What did Maya say about the Lisbon trip last spring? Where did we stay, and do I still owe her?

● inbox-recall · search_messages (query: "lisbon trip", sender: ["Maya"], when: "last spring", limit: 3, context: 1)
  ⎿ # Results 1-3 of 37 for "lisbon trip" (hybrid search)
    filters: sender=Maya · time=2026-03-01 to 2026-05-31 ("last spring")

    ## Maya Chen · whatsapp dm · conversation 1
      [#266] 2026-03-20 10:39 me: fair enough
    → [#267] 2026-03-25 19:02 Maya Chen: ok I'm officially in for Lisbon in May!! flights are cheap from here right now
      [#268] 2026-03-25 19:05 me: amazing. I can do May 14 to 20, does that work?

    ## Maya Chen · whatsapp dm · conversation 1
      [#288] 2026-05-19 03:53 me: I forgot my charger again
    → [#289] 2026-05-20 09:30 Maya Chen: still thinking about that custard tart place. best trip we've taken
      [#290] 2026-05-21 12:46 Maya Chen: what time works for you

    ## Maya Chen · whatsapp dm · conversation 1
      [#277] 2026-04-24 19:02 Maya Chen: my coworker says Manteigaria has the best pastel de nata, way better than Belem
    → [#278] 2026-04-24 19:06 Maya Chen: also we have to do the day trip to Sintra, she said go early before the tour buses
      [#279] 2026-04-24 19:09 me: adding both to the list

● inbox-recall · get_conversation (message_id: 270, before: 1, after: 4)
  ⎿ ## Maya Chen · whatsapp dm · conversation 1
      [#269] 2026-03-25 19:07 Maya Chen: perfect. I'll book the flights tonight, TAP has a direct one
    → [#270] 2026-03-25 19:43 Maya Chen: booked! confirmation code is in your email. we land 7:50am
      [#271] 2026-03-28 19:02 Maya Chen: found a place in Alfama, the one with the blue tiled balcony and the view of the river
      [#272] 2026-03-28 19:04 me: omg yes book it before someone else does
      [#273] 2026-03-28 19:11 Maya Chen: done, 6 nights. you owe me 410 for your half
      [#274] 2026-03-28 19:14 me: sending it now

● Maya signed on for Lisbon on March 25 and booked a direct TAP flight that same evening,
  landing at 7:50am [#267, #269, #270]. Three days later she found a place in Alfama with a
  blue tiled balcony and a river view and booked it for 6 nights [#271, #273].

  She said you owed her 410 for your half, and you replied "sending it now" on March 28
  [#273, #274]. As far as the chat shows, that's settled.

  She also pushed Manteigaria for pastel de nata and an early-morning day trip to Sintra
  [#277, #278], and in May called it the best trip you've taken [#289].
```

The agent split the question into a topic, a person and a time range, then opened the thread around a promising hit to read what happened next. Every claim carries the ids it came from.

## Why it's different

| | Built-in assistant message search | inbox-recall |
| --- | --- | --- |
| **Networks** | One app's messages | Every network Beeper bridges (WhatsApp, Signal, Telegram, iMessage, Google Messages, Discord, Slack, Messenger, LinkedIn, X...) plus Instagram, deduplicated into one index |
| **Where your archive lives** | The vendor decides | Your Postgres, on your hardware. Nothing is uploaded unless you pick a hosted embedding API |
| **Which agent can use it** | That vendor's assistant | Any MCP client over stdio or HTTP, plus a REST API and a CLI |
| **Retrieval** | Whatever the vendor ships | Semantic ANN (pgvector HNSW) + Postgres full-text + trigram fuzzy matching, fused with reciprocal rank fusion. You can read it, tune it and benchmark it |
| **Answers come as** | Depends on the integration | Conversation excerpts: each hit plus surrounding turns, overlapping windows merged, replies quoted, `[#id]` on every line |
| **Time** | Depends on the integration | `when: "last spring"`, `"march 2024"`, `"past 2 weeks"`, `"over the holidays"`, resolved in your time zone |
| **Scale** | Bounded by the integration | Years of history: HNSW and GIN indexes, keyset pagination, and an embedder that only touches new or edited messages |
| **Your data afterwards** | Inside the product | Plain SQL tables with stable ids. Query them, chart them, build on them |

## Features

- **Hybrid retrieval.** Three rankers run under the same filters: cosine ANN over the active model's HNSW index, English full-text search with a boost for messages that contain every term, and trigram matching that rescues typos and half-remembered names. Weighted reciprocal rank fusion merges them.
- **Context-aware embeddings.** Each message is embedded with its chat title and the turns just before it, so a bare "booked it!" still matches a search for the trip it was about.
- **Filters agents actually need.** `sender` (with `me` for your own messages), `person` (any chat that includes them), `chat`, `network`, `direction`, and time as plain English or ISO dates.
- **Thread reconstruction.** `get_conversation` reads around any message or across a time range, with cursors to keep scrolling older or newer.
- **Citations everywhere.** Every message line starts with its stable id. `get_messages` re-fetches cited ids so an agent can double-check itself.
- **Pagination for huge archives.** Relevance results page with consistent cursors; filter-only queries walk every match newest-first with keyset pagination.
- **People and chats across networks.** `find_people` merges one person's WhatsApp, Signal and Instagram identities by name. `list_conversations` shows who is in each chat and when it was active.
- **Pluggable embeddings.** Any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM, llama.cpp, TEI), in-process sentence-transformers, or a deterministic hashing embedder for tests. Models live side by side; switch with one env var.
- **Incremental and resumable.** The embedder works off `content_updated_at`, commits per batch, re-embeds edits, drops deleted messages, and wakes on Postgres `NOTIFY` the moment the sync writes something new.
- **MCP without the weight.** A dependency-free MCP server (stdio and Streamable HTTP), a REST API and a CLI share one tool registry. The only runtime dependency is psycopg.
- **Sync included.** Built on [beeper-pg-sync](https://github.com/joey12725/beeper-pg-sync): read-only Beeper Desktop sync with per-chat backfill, plus a Chrome extension for Instagram DMs. See [docs/sync.md](docs/sync.md).

## Quickstart

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and Postgres with pgvector (the bundled Compose file runs `pgvector/pgvector:pg17`).

### 1. Try it on the demo archive

```bash
git clone https://github.com/joey12725/inbox-recall.git
cd inbox-recall
docker compose up -d db
uv sync

export DATABASE_URL=postgresql://sync:sync@localhost:5432/messages
export EMBED_PROVIDER=hash        # zero-setup embedder; use ollama for real semantic search
uv run inbox-recall demo          # ~3,000 fabricated messages across 6 networks, embedded
uv run inbox-recall search lisbon flights --from maya --when "last spring"
uv run inbox-recall search 'from:me "pay you back"'
uv run inbox-recall search alfama --context 4      # a hit plus the thread around it
uv run inbox-recall eval          # the retrieval benchmark
```

`demo` refuses to run against a database that already holds real messages, so point it at a scratch database.

### 2. Index your own messages

```bash
cp .env.example .env              # add BEEPER_ACCESS_TOKEN; pick an embedding provider
set -a; source .env; set +a

ollama pull nomic-embed-text      # or set EMBED_PROVIDER=openai / lmstudio
uv run inbox-recall migrate
uv run beeper-pg-sync sync        # first sync from Beeper Desktop
uv run inbox-recall embed         # first embedding pass (incremental after this)
uv run inbox-recall stats
```

To get a Beeper token, open Beeper Desktop → Settings → Integrations, enable the Desktop API and add an approved connection. Instagram DMs can come in through the Chrome extension too ([docs/sync.md](docs/sync.md)).

Keep everything fresh with `uv run beeper-pg-sync run` (sync daemon) and `uv run inbox-recall serve --embed` (search API, MCP over HTTP and the embedding worker), or `docker compose --profile daemon up -d` to run both in containers.

### 3. Connect your agent

**Claude Code**

```bash
claude mcp add inbox-recall \
  -e DATABASE_URL=postgresql://sync:sync@localhost:5432/messages \
  -e EMBED_PROVIDER=ollama \
  -e EMBED_QUERY_PREFIX="search_query: " -e EMBED_DOCUMENT_PREFIX="search_document: " \
  -- uv --directory /path/to/inbox-recall run inbox-recall mcp
```

**Claude Desktop, Cursor and other JSON-configured clients**

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

**Agents on another machine** (OpenClaw, Hermes Agent, a home-server agent) connect over Streamable HTTP:

```bash
RECALL_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))") \
  uv run inbox-recall serve --embed --host 0.0.0.0
# MCP endpoint: http://<host>:8766/mcp  with  Authorization: Bearer $RECALL_TOKEN
```

More clients and prompting tips: [docs/mcp.md](docs/mcp.md).

## Tools

| Tool | What it does |
| --- | --- |
| `search_messages` | Hybrid search with filters and plain-English time. Empty query lists every match newest-first. Returns thread excerpts with `[#id]` citations and a cursor. |
| `get_conversation` | Read around a message, across a time range, or the latest messages in a chat, with older/newer cursors. |
| `get_messages` | Fetch specific ids, optionally with context, to verify a citation. |
| `list_conversations` | Chats by recency with message counts, date span and members; filter by title, network, person or time. |
| `find_people` | Resolve a name or handle across networks, with message counts per network. |
| `archive_stats` | Networks, date coverage and embedding coverage. |

All tools are read-only. Each returns a compact transcript by default, or the full structured result with `format: "json"`. The same tools are on the REST API (`GET /v1/search`, `/v1/conversations`, `/v1/messages/{id}/thread`, `/v1/people`, `/v1/stats`) and the CLI.

## Retrieval quality

`inbox-recall eval` runs 24 agent-style queries against the demo archive, each with known answers, in four groups: exact wording, paraphrase (same meaning, different words), filtered (people, chats, networks, time) and fuzzy (typos). CI runs it on every push with `nomic-embed-text` served by Ollama, through the same OpenAI-compatible provider you'd use at home.

| mode (nomic-embed-text) | all R@10 | exact wording | paraphrase | filtered | typos | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword only | 0.71 | 1.00 | 0.36 | 0.67 | 1.00 | 0.71 |
| semantic only | 0.85 | 1.00 | 0.64 | 0.86 | 1.00 | 0.71 |
| **hybrid** | **0.90** | **1.00** | **0.64** | **1.00** | **1.00** | **0.80** |

Hybrid matches or beats each ranker alone in every group, and the first relevant message is usually the top hit. Full-text search nails exact wording, the embedding model carries paraphrase, and filters run inside every ranker so "Maya, last spring" never starves the vector search. Methodology, the model-free baseline and a 250k-message latency run: [docs/retrieval.md](docs/retrieval.md).

## How it works

1. **Sync.** beeper-pg-sync writes every chat into `messages` with stable ids, merges the same message seen by two sources, and bumps `content_updated_at` on edits. Each write sends a `NOTIFY`.
2. **Embed.** `inbox-recall embed` picks up messages with no vector (or a stale one) for the active model, embeds each with its chat title and recent turns, and commits per batch. Each model gets its own partial HNSW index over `embedding::vector(dims)`, so models never mix and swapping one needs no migration.
3. **Retrieve.** One set of SQL filters feeds three rankers: HNSW cosine search (with pgvector's iterative scan so filters don't starve it), stored `to_tsvector('english')` full-text, and trigram similarity. Weighted reciprocal rank fusion merges them, with a bonus for messages that contain every query term.
4. **Expand.** Each hit pulls N messages either side from the `(conversation_id, sent_at, id)` index. Overlapping windows merge into one excerpt and reply targets are quoted inline.
5. **Serve.** The same tool registry backs MCP (stdio and HTTP), the REST API and the CLI.

Deeper notes: [docs/architecture.md](docs/architecture.md), [docs/retrieval.md](docs/retrieval.md), [docs/configuration.md](docs/configuration.md), [docs/extending.md](docs/extending.md).

## Development

```bash
uv sync
uv run ruff check src tests && uv run ruff format --check src tests
TEST_DATABASE_URL=postgresql://sync:sync@localhost:5432/messages_test uv run pytest

npm ci && npm run lint && npm test     # Chrome extension
```

Database tests need Postgres with pgvector and reset the `public` schema of `TEST_DATABASE_URL`, so use a throwaway database. The suite covers the sync layer, the embedding pipeline (including resume after failures and `NOTIFY` wake-ups), retrieval, filters, pagination, the MCP protocol over stdio and HTTP, and the benchmark, all on fabricated conversations with a deterministic embedder.

## Tech stack

Python 3.11+ · PostgreSQL 16/17 · pgvector (HNSW) · pg_trgm · Postgres full-text search · psycopg 3 · Model Context Protocol · Ollama / OpenAI-compatible embeddings · sentence-transformers (optional) · Beeper Desktop API · Chrome MV3 extension · pytest · ruff · Docker Compose · GitHub Actions

## Credits

Built by [Joey Hersh](https://github.com/joey12725). MIT licensed. A fork of [beeper-pg-sync](https://github.com/joey12725/beeper-pg-sync), which provides the sync layer. Not affiliated with Beeper, Anthropic, OpenAI or Meta; use it with your own accounts and follow each network's terms.
