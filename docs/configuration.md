# Configuration

Everything is read from the environment. `.env.example` lists every variable; copy it to `.env` and load it with `set -a; source .env; set +a` or Docker Compose's `env_file`. The sync and search layers share `DATABASE_URL`.

## Search (inbox-recall)

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | none | Postgres connection string, e.g. `postgresql://sync:sync@localhost:5432/messages`. The database needs the `vector` and `pg_trgm` extensions available (the Compose image has both). |
| `EMBED_PROVIDER` | `none` | `ollama`, `lmstudio`, `openai` (any OpenAI-compatible API), `sentence-transformers`, `hash`, or `none` for keyword-only search |
| `EMBED_MODEL` | per provider | `nomic-embed-text` (ollama), `text-embedding-nomic-embed-text-v1.5` (lmstudio), `text-embedding-3-small` (openai), `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| `EMBED_BASE_URL` | per provider | `http://127.0.0.1:11434/v1`, `http://127.0.0.1:1234/v1`, `https://api.openai.com/v1`. Point it at vLLM, llama.cpp, TEI or LiteLLM the same way. |
| `EMBED_API_KEY` | `OPENAI_API_KEY` | Bearer token for the embeddings API, if it needs one |
| `EMBED_DIMENSIONS` | model default | Ask for shorter vectors (OpenAI `text-embedding-3-*`). HNSW indexes take up to 2000 dimensions. For `hash`, the vector width (256). |
| `EMBED_BATCH_SIZE` | `64` | Messages per embedding request and per commit |
| `EMBED_CONTEXT_MESSAGES` | `2` | Earlier messages (same chat, within six hours) embedded with each message |
| `EMBED_QUERY_PREFIX` / `EMBED_DOCUMENT_PREFIX` | empty | Task prefixes for models that want them: `search_query: ` / `search_document: ` for nomic, `query: ` / `passage: ` for e5 |
| `RECALL_TIMEZONE` | system zone | IANA zone used for "last spring" and for timestamps shown to agents |
| `RECALL_TOKEN` | none | Bearer token for the HTTP API and MCP over HTTP. Without it the server only binds to loopback and rejects cross-origin browser requests. |
| `RECALL_HOST` / `RECALL_PORT` | `127.0.0.1` / `8766` | HTTP bind address |

The model name stored with each vector is `<provider>:<model>[@<dimensions>]`. Change any of those and inbox-recall treats it as a new model: it embeds into a fresh index and leaves the old one alone until you run `inbox-recall embed --drop-model NAME`. Search uses the active model; if it has no vectors yet, search runs keyword-only and says so.

### Picking an embedding model

| Setup | Settings |
| --- | --- |
| Fully local, good quality | Ollama + `nomic-embed-text` (768-d), prefixes `search_query: ` / `search_document: ` |
| Fully local, multilingual chats | Ollama + `bge-m3` (1024-d) |
| LM Studio | `EMBED_PROVIDER=lmstudio`, model name as LM Studio lists it |
| Hosted | `EMBED_PROVIDER=openai`, `text-embedding-3-small`, optionally `EMBED_DIMENSIONS=512` |
| In-process | `uv pip install sentence-transformers`, `EMBED_PROVIDER=sentence-transformers`, `EMBED_MODEL=BAAI/bge-small-en-v1.5` |
| Tests and demos | `EMBED_PROVIDER=hash`: deterministic, no model, lexical only |

### Commands

| Command | Does |
| --- | --- |
| `inbox-recall migrate` | create or upgrade the sync + search schema |
| `inbox-recall embed [--limit N] [--watch] [--status] [--drop-model NAME]` | embed new and edited messages; `--watch` keeps running and wakes on `NOTIFY` |
| `inbox-recall search QUERY [--from X] [--with X] [--in CHAT] [--network N] [--direction D] [--when TEXT] [--after D] [--before D] [--mode M] [--context N] [--limit N] [--cursor C] [--json]` | search; the query also accepts `from:` `with:` `in:` `on:` `when:"..."` `after:` `before:` `is:sent` `is:received` |
| `inbox-recall thread [MESSAGE_ID] [--conversation ID] [--before N] [--after N] [--when TEXT] [--cursor C]` | read a thread |
| `inbox-recall chats [TITLE] [--network N] [--with X] [--when TEXT]` | list conversations |
| `inbox-recall people [NAME] [--network N]` | find people across networks |
| `inbox-recall stats` | archive and embedding coverage |
| `inbox-recall mcp` | MCP server on stdio |
| `inbox-recall serve [--host H] [--port P] [--embed]` | REST API + MCP over HTTP, optionally with the embedding worker |
| `inbox-recall demo [--filler N]` | load the fabricated demo archive into a scratch database |
| `inbox-recall eval [--mode M] [--json]` | run the retrieval benchmark on the demo archive |

## Sync (beeper-pg-sync)

| Variable | Default | Purpose |
| --- | --- | --- |
| `BEEPER_URL` | `http://127.0.0.1:23373` | Beeper Desktop API base URL |
| `BEEPER_ACCESS_TOKEN` | none | Token from Beeper Desktop → Settings → Integrations |
| `BEEPER_NETWORKS` | all | Comma-separated allowlist, e.g. `instagram,whatsapp,signal` |
| `BEEPER_PAGES_PER_CHAT` | `4` | Older-history pages fetched per chat per run |
| `BEEPER_RESOLVE_NATIVE_IDS` | `false` | Ask bridges for native thread IDs via Matrix room state |
| `INGEST_TOKEN` | none | Shared secret for the extension; the server refuses to start without it |
| `INGEST_HOST` | `127.0.0.1` | Ingest server bind address (`0.0.0.0` inside Docker) |
| `INGEST_PORT` | `8765` | Ingest server port |
| `SYNC_INTERVAL` | `900` | Seconds between syncs in `beeper-pg-sync run` |
| `BEEPER_PG_SYNC_MIGRATIONS` | none | Extra migration directories, separated by `:` |

### Network names

Beeper bridge types map to these `network` values: `discord`, `googlechat`, `imessage`, `instagram`, `linkedin`, `matrix`, `messenger`, `reddit`, `signal`, `slack`, `sms` (Google Messages), `snapchat`, `telegram`, `whatsapp`, `x`. Unknown bridges fall back to a slug of Beeper's network name.

### Sync commands

| Command | Does |
| --- | --- |
| `beeper-pg-sync migrate` | create or upgrade the schema |
| `beeper-pg-sync sync [--full] [--network NAME] [--pages-per-chat N]` | one Beeper sync; `--full` ignores saved positions |
| `beeper-pg-sync import FILE...` | import extension exports (`.csv`, `.zip`, `.json`) |
| `beeper-pg-sync serve [--host H] [--port P]` | run only the extension ingest server |
| `beeper-pg-sync run [--interval S]` | migrate, serve, and sync on a loop (the daemon) |
| `beeper-pg-sync status` | per-network counts and recent runs as JSON |

## Running in Docker

`docker compose --profile daemon up -d` builds one image and runs two services beside the database:

- `sync`: `beeper-pg-sync run`, publishing the extension ingest server on `127.0.0.1:8765`. It reaches Beeper Desktop on the host through `host.docker.internal`. If your Beeper API only listens on loopback, run the sync daemon on the host instead (`uv run beeper-pg-sync run`).
- `recall`: `inbox-recall serve --embed`, publishing the API and MCP endpoint on `127.0.0.1:8766`. It listens on all interfaces inside the container, so it needs `RECALL_TOKEN` in `.env`.

Add `--profile ollama` for a bundled Ollama (`docker compose exec ollama ollama pull nomic-embed-text`, then `EMBED_BASE_URL=http://ollama:11434/v1`), or reach one on the host with `EMBED_BASE_URL=http://host.docker.internal:11434/v1`.

## Security notes

- Every inbox-recall tool is read-only. The HTTP server compares `RECALL_TOKEN` in constant time, refuses to bind beyond loopback without one, and rejects browser requests from non-local origins when running tokenless on loopback.

- The Beeper client only issues GET requests to the accounts, chats, messages and room-state endpoints. Anything else raises before a request is made.
- The ingest server requires `Authorization: Bearer $INGEST_TOKEN` on every endpoint except `/healthz`, compares tokens in constant time and answers CORS only for `chrome-extension://` origins.
- Keep the ingest server on loopback or behind your own TLS proxy. Message bodies are private data.
