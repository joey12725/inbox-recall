# Extending

inbox-recall is a foundation as much as a tool: a stable message table with a clear ingestion boundary underneath, and a retrieval layer with explicit seams on top. Add a source and its messages become searchable; add a provider, a tool or a ranker without touching the rest.

## Stable hooks

| Hook | Use it for |
| --- | --- |
| `messages.id` (bigint, never reassigned) | foreign key for anything derived from a message: embeddings, labels, summaries |
| `messages.content_updated_at` (indexed) | incremental re-processing: "everything changed since my last run" |
| `pg_notify('beeper_pg_sync', json)` after each ingest that added or changed rows | wake a worker immediately instead of polling; payload has `conversation_id`, `source`, `added`, `changed` |
| `IngestResult.added` / `.changed` | message IDs touched by a `Store.ingest()` call, for in-process pipelines |
| `message_timeline` view | flat, joined rows for SQL, BI tools and LLM tool calls |
| `message_embeddings` (`message_id`, `model`) | reuse vectors for clustering, topic maps or recommendations |
| `BEEPER_PG_SYNC_MIGRATIONS` / `migrate(conn, extra_dirs=[...])` | ship your own tables in the same database and migration history |

## Adding an embedding provider

A provider is any object with `name`, `provider`, `query_prefix`, `document_prefix` and `embed(texts) -> list[list[float]]`. Most hosted and local servers already speak the OpenAI `/v1/embeddings` shape, so they need no code at all:

```bash
EMBED_PROVIDER=openai EMBED_BASE_URL=http://gpu-box.local:8080/v1 EMBED_MODEL=bge-m3 inbox-recall embed
```

For anything else, add a class to `inbox_recall/embedders.py` and a branch in `from_settings`. Give it a `name` that changes whenever its vector space changes (model, dimensions, normalization); the database keys vectors and HNSW indexes on that name.

## Adding a tool

Tools live in `inbox_recall/tools.py` as `Tool(name, title, description, schema, run, text)`. `run(searcher, arguments)` returns a JSON-serializable dict; `text(result)` renders it for the model. Add one there and it shows up in MCP `tools/list`, can be called over stdio and HTTP, and can be routed in `api.py`. Keep tools read-only.

## Tuning retrieval

The knobs are constants at the top of `inbox_recall/search.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `WEIGHTS` | semantic 1.0, keyword 1.0, fuzzy 0.6 | how much each ranker counts in the fusion |
| `RRF_K` | 60 | flatter (higher) or steeper (lower) rank curves |
| `ALL_TERMS_BONUS` | half a first-place keyword vote | extra credit for messages containing every query term |
| `DEFAULT_POOL` | 200 | candidates per ranker; also the relevance pagination depth |
| `FUZZY_WHEN_FEWER_THAN` | 3 | trigram matching runs when fewer exact all-term matches exist |

Change one, then run `inbox-recall eval` on the demo archive (and ideally on a handful of real questions of your own) to see what moved. Add benchmark cases to `inbox_recall/eval.py`; each is a query, optional filters and the demo message keys that answer it.

## Adding tables

Put SQL files in your own directory with a prefix range that won't collide: core sync migrations use `0001`-`0999` and inbox-recall uses `1000`-`1999`.

```python
from beeper_pg_sync.db import connect, migrate
from inbox_recall.schema import migrations_dir

with connect() as conn:
    migrate(conn, extra_dirs=[migrations_dir(), "my_project/migrations"])
```

Key anything derived from a message on `messages.id` and watch `messages.content_updated_at` to know when to recompute, the way `message_embeddings` does.

## Adding a source

A source is anything that can produce `beeper_pg_sync.model.Batch` objects:

```python
from beeper_pg_sync.model import Batch, Conversation, Message, Participant
from beeper_pg_sync.store import Store

batch = Batch(
    source="my-exporter",
    conversation=Conversation(source_conversation_id="thread-42", network="signal", title="Weekend plans"),
    messages=[
        Message(
            source_message_id="m-1",
            body="who's driving?",
            sent_at=some_aware_datetime,
            direction="incoming",
            sender=Participant(native_id="user-7", display_name="Ana Silva"),
        )
    ],
    link_by_overlap=True,  # merge into a chat another source already has, if messages overlap
)
Store(conn).ingest(batch)
```

Guidelines:

- Pick a stable `source` name and stable `source_message_id`s; that is what makes re-ingestion idempotent.
- Use timezone-aware datetimes and set `time_precision` honestly.
- Set `native_id` on the conversation when you know the network's own thread ID. Two sources that agree on it link instantly.
- Use `Store.get_state` / `Store.put_state` for resumable cursors, keyed by your source name and a scope of your choosing.
