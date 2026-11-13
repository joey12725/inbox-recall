# Architecture

inbox-recall has two layers in one database:

- **Sync** (`beeper_pg_sync`, from beeper-pg-sync): sources write canonical, deduplicated `messages` rows with stable ids.
- **Search** (`inbox_recall`): an embedding pipeline, hybrid retrieval and the agent interfaces, all reading those rows.

```
src/inbox_recall/
  migrations/1000_search.sql  pgvector + pg_trgm, embedding tables, full-text/trigram/order indexes
  schema.py           migrate (core + search), per-model HNSW index management
  embedders.py        OpenAI-compatible HTTP, sentence-transformers, deterministic hashing
  pipeline.py         incremental, resumable embedding with chat context; LISTEN/NOTIFY watch mode
  timeparse.py        "last spring" -> [start, end) in the user's time zone
  query.py            SearchRequest, inline operators (from: in: when: ...), cursors
  search.py           filters, the three rankers, rank fusion, excerpts, threads, people, chats, stats
  render.py           compact citation-first transcripts for agents
  tools.py            the six tool definitions shared by every interface
  mcp.py              JSON-RPC MCP server (stdio + HTTP handler)
  api.py              REST API + POST /mcp (stdlib HTTP server)
  cli.py              migrate | embed | search | thread | chats | people | stats | mcp | serve | demo | eval
  demo.py, eval.py    fabricated archive and retrieval benchmark
src/beeper_pg_sync/   the sync layer (see below)
extension/            Chrome MV3 extension for Instagram DMs
```

## Search layer

### Schema

`1000_search.sql` adds:

- `embedding_models (model, provider, dims, index_name)`: one row per vector space. The model name includes the provider and any dimension override (`ollama:nomic-embed-text`, `openai:text-embedding-3-small@512`, `hash-v1@256`).
- `message_embeddings (message_id, model, embedding vector, content_updated_at)`: one vector per message per model. `embedding` has no fixed width; `content_updated_at` is the message version that was embedded.
- A partial HNSW index per model, created on first use: `USING hnsw ((embedding::vector(768)) vector_cosine_ops) WHERE model = 'ollama:nomic-embed-text'`. Queries cast the same way and inline the model name, so the planner can prove the partial predicate and use the index.
- a stored `search_tsv` column (`to_tsvector('english', body)`) with a GIN index for ranked full-text search, `GIN (body gin_trgm_ops)` for fuzzy matching, `(conversation_id, sent_at, id)` for thread windows and keyset pagination, `(sender_id)` for person lookups, and a trigram index over participant names and handles.

### Embedding pipeline

The queue is a query, not a table: messages with text, not deleted, with no vector for the active model or with `messages.content_updated_at` newer than the embedded version. The pipeline walks that set in id order with keyset pagination, so a full pass over millions of rows is linear.

For each message it builds a short document: the network and chat title, up to `EMBED_CONTEXT_MESSAGES` earlier messages from the same chat within six hours, then the message itself. Each batch is embedded and upserted in one transaction. A crash loses at most one batch. Deleted or emptied messages lose their vectors at the start of every run.

`inbox-recall embed --watch` (and `serve --embed`) runs a pass, then waits on `LISTEN beeper_pg_sync`; the sync layer notifies after every ingest that adds or changes rows, so new messages are searchable within seconds. A poll interval catches anything missed.

### Retrieval

`Searcher.search()` builds one set of SQL predicates from the request (sender, person, chat, network, direction, time range) and runs up to three rankers with them:

1. **semantic**: `ORDER BY embedding::vector(d) <=> query::vector(d)` on the model's HNSW index, with `hnsw.iterative_scan = relaxed_order` (pgvector 0.8+) so selective filters keep scanning instead of returning too few rows. Results are re-sorted by exact distance.
2. **keyword**: full-text match on any query term, ordered by "contains every term" then `ts_rank_cd`. Quoted phrases, `OR` and `-term` switch to `websearch_to_tsquery` semantics.
3. **fuzzy**: `word_similarity` trigram matching on the query words that full-text search can't find anywhere, for queries of up to four words, used when fewer than three messages contain every term. It runs under a 300 ms statement timeout.

Scores merge with weighted reciprocal rank fusion (`k = 60`; weights 1.0 / 1.0 / 0.6) plus a bonus for messages containing every term. See [retrieval.md](retrieval.md) for the reasoning and the benchmark.

Each ranker returns a fixed pool (200 by default, or 3x the page size). The first page fixes the pool size in the cursor, so later pages slice the same fused list: no duplicates, no gaps. With an empty query, search switches to chronological browsing with keyset cursors over `(sent_at, id)`, which can walk any number of messages.

### Excerpts

For the page of hits, one `LATERAL` query pulls `context` messages before and after each hit from the `(conversation_id, sent_at, id)` index. Windows that share a message are merged with a union-find, so three hits in one exchange come back as one excerpt. Reply targets outside the window are attached as `in_reply_to`. Excerpts are ordered by their best hit.

### Interfaces

`tools.py` defines each tool once: JSON schema, handler and text renderer. `mcp.py` maps `tools/list` and `tools/call` onto it; `api.py` maps `GET /v1/...` routes onto it; `cli.py` calls the same handlers. Validation errors come back as tool errors (`isError: true`) rather than protocol errors, so the model can read them and retry.

# Sync layer

```
src/beeper_pg_sync/
  model.py           the ingestion contract: Account, Conversation, Participant, Message, Batch
  store.py           Store.ingest(batch): the only code that writes messages
  db.py              connections + migration runner
  migrations/        numbered SQL files (0001_init.sql)
  sources/beeper.py  Beeper Desktop API client, normalization, incremental sync
  sources/instagram.py  Chrome extension payloads and CSV/ZIP/JSON exports
  server.py          HTTP ingest endpoint for the extension
  cli.py             migrate | sync | import | serve | run | status
```

## Data model

A **conversation** belongs to one network (`instagram`, `whatsapp`, `discord`, ...). A **source** is how we learned about it (`beeper`, `instagram-extension`). Sources have their own IDs for the same thing, so each table that sources write to has a `*_sources` companion:

- `conversation_sources (source, source_conversation_id) -> conversation_id`, with `linked_by` recording whether the link came from the source itself, a shared native thread ID, a message overlap or a manual fix.
- `message_sources (source, source_conversation_id, source_message_id) -> message_id`, with the raw source metadata.

`messages` is the canonical table. Its `id` is assigned once and never changes, even when a second source later observes the same message. `messages.source` records which source created the row; that source alone may rewrite the content.

`time_precision` is `exact` for real timestamps (Beeper, Instagram native metadata), `approximate` for times derived from a displayed label, and `unknown` when there is nothing to go on.

`content_updated_at` changes only when `body`, `edited_at` or `deleted` change (a trigger enforces it). Reactions and bookkeeping updates leave it alone.

## Beeper sync

`BeeperSync.run()` walks `/v1/accounts` and every page of `/v1/chats`. Merged chat containers are skipped because their members appear separately. For each chat:

1. If the stored state is `complete` and `lastActivity` is unchanged, skip without fetching messages.
2. **New messages.** Page backwards from the newest message until a page reaches the stored `frontier` (newest timestamp from the last successful run). On the very first sync, stop after `pages_per_chat` pages and remember the oldest cursor.
3. **Backfill.** If a backfill cursor is stored, continue older history for up to `pages_per_chat` pages and save the new cursor. When Beeper reports no more history, the chat becomes `complete`.
4. Save `sync_state`: status (`complete`, `backfilling`, `partial`, `failed`), cursor, frontier, `lastActivity` and diagnostics.

Each page is ingested in its own transaction. A failure in one chat is recorded in `sync_state` and the run moves on; an authentication error stops the run.

Beeper sometimes returns an empty page with `hasMore=true` when a bridge has a history gap. On a first sync the cursor is parked for backfill. On later syncs the frontier is not advanced, so the next run retries rather than skipping messages.

Message normalization:

- `REACTION` events and hidden messages are not stored as messages (reactions are already aggregated on their target).
- Rich text is reduced to plain text with a parser that never loads URLs or runs scripts.
- Attachments keep type, MIME type, file name, size, duration and dimensions. Source URLs and media IDs are dropped.
- Matrix user IDs like `@whatsapp_15550100001:beeper.local` become network-native participant IDs (`15550100001`).

Beeper chat IDs are Matrix room IDs, not network-native thread IDs. With `BEEPER_RESOLVE_NATIVE_IDS=true`, the sync also asks the room's `m.bridge` state for the native thread ID (available on some self-hosted setups). If the endpoint is not there, it stops asking for the rest of the run.

## Instagram extension

The extension has three parts:

- `native.js` runs in the page's main world and copies a whitelist of message metadata (native ID, millisecond timestamp, sender ID, direction, reply target) from Instagram's rendered React props onto each message element as a data attribute. Nothing else from the page crosses over.
- `content.js` + `core.js` run in the isolated world, draw the panel in a Shadow DOM, scroll the thread and collect rows.
- `background.js` holds the server URL and token and makes the HTTP call, so Instagram's page never sees either.

The payload (`format: "beeper-pg-sync/instagram@1"`) carries the thread ID from `/direct/t/<thread>/`, the title and the rows. Rows without native timestamps get an `approx_time` computed in the browser's own time zone from the displayed label.

On the server, rows with a native ID use it as `source_message_id`. Rows without one get a fingerprint of their visible content plus an occurrence counter, so re-importing the same export is idempotent and two identical "ok" messages stay two rows.

## Cross-source matching

When a batch arrives for a conversation the source has not seen before:

1. If the batch carries a native thread ID that another conversation already has, link to it.
2. Otherwise, for fallback sources (`link_by_overlap=True`), look for a conversation on the same network, not yet linked to this source, that shares at least three matching messages, at least twice as many as any runner-up.
3. Otherwise create a new conversation.

A message the source has not seen before merges into an existing row when there is exactly one candidate in the same conversation, not yet observed by this source, with the same direction (or unknown), a matching formatting-insensitive body and a timestamp within 90 seconds (15 minutes for approximate times). Media-only messages need exact timestamps within 2 seconds. When several identical candidates exist, only a clear nearest match within 2 seconds is accepted. Everything else becomes a new row: a rare duplicate is easier to fix than a wrong merge.
