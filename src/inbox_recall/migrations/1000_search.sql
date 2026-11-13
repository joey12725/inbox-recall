-- inbox-recall search layer, on top of the beeper-pg-sync core schema.
--
-- Embeddings are keyed on the stable `messages.id` and on the model that
-- produced them, so you can switch models (or run two side by side) without a
-- migration. The vector column has no fixed width; each model gets its own
-- partial HNSW index over `embedding::vector(<dims>)`, created by
-- `inbox_recall.schema.ensure_model` the first time that model is used.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE embedding_models (
    model       text PRIMARY KEY,
    provider    text NOT NULL,
    dims        integer NOT NULL CHECK (dims > 0 AND dims <= 2000),
    index_name  text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE message_embeddings (
    message_id          bigint NOT NULL REFERENCES messages (id) ON DELETE CASCADE,
    model               text NOT NULL REFERENCES embedding_models (model) ON DELETE CASCADE,
    embedding           vector NOT NULL,
    -- The messages.content_updated_at value that was embedded. When the
    -- message changes, content_updated_at moves past this and it is re-embedded.
    content_updated_at  timestamptz NOT NULL,
    embedded_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (message_id, model)
);

CREATE INDEX message_embeddings_model ON message_embeddings (model, content_updated_at);

-- Keyword search: a stored English tsvector (ranking reads it instead of
-- re-parsing every matching body), and trigrams for names, typos and partial words.
ALTER TABLE messages
    ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED;
CREATE INDEX messages_search_tsv ON messages USING gin (search_tsv);
CREATE INDEX messages_body_trgm ON messages USING gin (body gin_trgm_ops);

-- Thread reconstruction walks a conversation by (sent_at, id) in both directions.
CREATE INDEX messages_conversation_order ON messages (conversation_id, sent_at, id);
CREATE INDEX messages_sender ON messages (sender_id);

-- Person lookups match display names and handles fuzzily.
CREATE INDEX participants_name_trgm ON participants
    USING gin ((coalesce(display_name, '') || ' ' || coalesce(handle, '')) gin_trgm_ops);
