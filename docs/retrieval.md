# Retrieval

Searching chat history is not like searching documents. Messages are short, people write in fragments, the important word is often in the message before, and the question usually names a person and a time ("what did Maya say about the trip last spring"). inbox-recall is tuned for that.

## Why hybrid

Each ranker fails in a different way on chat data:

- **Full-text search** finds exact words fast and precisely, and is useless when the message says "custard tart" and you ask about "pastry".
- **Embeddings** handle paraphrase, but a lone "booked it!" carries no topic, and rare names, codes and typos often embed poorly.
- **Trigram similarity** recovers "manteigria" → "Manteigaria" and "Lantern Hal" → "Lantern Hall", but ranks badly on full sentences.

Running all three under the same filters and fusing their ranks keeps each one's strengths. Reciprocal rank fusion (`score = Σ weight / (60 + rank)`) needs no score calibration between rankers, which matters because cosine distance, `ts_rank_cd` and trigram similarity live on unrelated scales.

Two details make the fusion behave on chat logs:

- **All-terms bonus.** A message that contains every query term gets an extra half of a first-place vote. Short filler like "let me check and get back to you" can sit close to a query in embedding space; a message that actually says "pay you back" should win.
- **Fuzzy as a backstop.** Trigram matching only runs when full-text search finds fewer than three messages with every term, and only for queries of four words or fewer. That's when typos and partial names are the likely problem, and it keeps fuzzy noise out of queries that already have good exact matches.

## Embedding messages with context

The embedding pipeline doesn't embed a message alone. It embeds:

```text
whatsapp chat: Maya Chen
Maya Chen: found a place in Alfama, the one with the blue tiled balcony and the view of the river
me: omg yes book it before someone else does
```

The chat title, then up to `EMBED_CONTEXT_MESSAGES` earlier messages from the same chat within six hours, then the message. "omg yes book it" now lives near "where are we staying in Lisbon" in vector space. Only the last line is the message being indexed; the context just places it. Six hours keeps a reply from absorbing yesterday's unrelated thread.

## Filters run inside every ranker

Sender, person, chat, network, direction and time are SQL predicates applied to all three rankers, not post-filters on a top-k list. For the vector ranker, pgvector 0.8's `hnsw.iterative_scan = relaxed_order` keeps the HNSW scan going until enough rows pass the filters, so "messages from Maya last spring" doesn't come back with two results because the 40 nearest neighbors were from someone else. `ef_search` scales with the candidate pool.

`sender: ["me"]` maps to outgoing messages. `person` resolves names and handles to participants across networks, then limits to every conversation they appear in (DMs and groups) plus chats titled with their name.

## Time

`when` takes plain English and resolves it in `RECALL_TIMEZONE` to a half-open range. A bare period is the latest one that has started ("march" asked in November is this March); "last X" is the latest one that has finished ("last spring" in November is this year's spring, "last november" is a year ago). Seasons are northern meteorological seasons. The full grammar is in the docstring of `inbox_recall/timeparse.py` and covered by its tests.

## Excerpts, not rows

Agents answer better from conversation than from isolated lines. Each hit brings `context` messages either side (default 2), windows that overlap merge into one excerpt, and replies to messages outside the window are quoted inline. Everything is ordered by time inside an excerpt and by best hit across excerpts. With `context: 0` you get just the hits.

## Pagination

Relevance pages: each ranker returns a fixed pool (200, or 3x the page size if larger). The pool size is baked into the cursor along with a fingerprint of the query and filters, so page 2 slices the same fused list as page 1: no repeats, no gaps, and a clear error if the cursor is reused with a different query. When the pool runs out the result says so.

Exhaustive listing: drop the query text. Search then walks every matching message newest-first with keyset cursors over `(sent_at, id)`, at constant cost per page however deep you go.

## Benchmark

`inbox-recall eval` runs 24 queries against the fabricated demo archive (about 3,000 messages over six networks and three years, mostly filler chatter). Each query lists the messages that answer it. Four groups:

| Group | Example | Tests |
| --- | --- | --- |
| keyword | "Glass Orchard" | exact wording |
| paraphrase | "where are we staying in Portugal" (the message says "a place in Alfama") | meaning without shared words |
| filtered | "lisbon flights", sender Maya, when "last spring" | people, chats, networks, time |
| fuzzy | "manteigria", "okafor dentst" | typos and partial names |

Results from CI (`.github/workflows/ci.yml`, job `eval`), Recall@10 per group:

**nomic-embed-text via Ollama**

| mode | all | keyword | paraphrase | filtered | fuzzy | all R@5 | all MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.71 | 1.00 | 0.36 | 0.67 | 1.00 | 0.71 | 0.71 |
| semantic | 0.85 | 1.00 | 0.64 | 0.86 | 1.00 | 0.83 | 0.71 |
| **hybrid** | **0.90** | **1.00** | **0.64** | **1.00** | **1.00** | **0.90** | **0.80** |

**Hashing embedder (no model)**

| mode | all | keyword | paraphrase | filtered | fuzzy | all R@5 | all MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.71 | 1.00 | 0.36 | 0.67 | 1.00 | 0.71 | 0.71 |
| semantic | 0.73 | 1.00 | 0.21 | 0.86 | 1.00 | 0.69 | 0.66 |
| hybrid | 0.81 | 1.00 | 0.36 | 1.00 | 1.00 | 0.77 | 0.74 |

What it shows:

- Hybrid is at least as good as the best single ranker in every group and better overall, with either embedder.
- A real embedding model nearly doubles paraphrase recall over keywords (0.36 → 0.64) and lifts MRR to 0.80, meaning the first relevant message is usually the top hit.
- Filters are where hybrid pulls ahead of pure vector search: the semantic ranker alone misses a filtered case that keyword evidence rescues.
- The hashing embedder is a lexical stand-in for tests. It still helps fusion (0.71 → 0.81) because its character trigrams act as a second fuzzy signal.

The demo archive is small and synthetic, so treat these as a regression harness and a sanity check, not a leaderboard. To measure your own setup, add cases to `inbox_recall/eval.py`, point `EMBED_*` at your model and run `inbox-recall demo && inbox-recall eval` against a scratch database.

## Large archives

Everything on the query path is index-backed: the per-model HNSW index, the GIN full-text and trigram indexes, and the `(conversation_id, sent_at, id)` b-tree for thread windows and browsing. Embedding is incremental, so a sync that adds 50 messages costs 50 embeddings. `scripts/scale_bench.py` builds a synthetic archive of any size and times the agent-facing calls; CI's manual `scale` job runs it at 250,000 messages.

SCALE_TABLE
