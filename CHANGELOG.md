# Changelog

## 0.3.0 (2026-11-22)

The MCP tools, their arguments and the skill are unchanged, so agents need no update. New result fields are
additive.

### Embeddings

- **`openrouter:<model>` embedder.** The key comes from `OPENROUTER_API_KEY`, and every request asks for
  zero-data-retention providers that do not train on the data.
- **Remote embedders retry transient failures.** Network errors and HTTP 408/429/5xx are retried with backoff.
- **No probe when the dimension is known.** A store that records its dimension opens without contacting the
  endpoint, so servers, `stats` and backups start without network.
- **Larger remote chunks.** Remote models chunk at 8,000 characters, so memories stay whole. Local models keep
  1,200.
- **Query and passage prefixes.** e5 and Qwen3-Embedding models get their expected prefixes automatically.

### Recall

- **Lexical fallback.** When the embedding endpoint is unreachable, `recall` answers by word overlap. It
  flags this with `embedding: "lexical-fallback"` and a new `warning` field instead of failing.
- **`--decay-tags <LIST>`.** Only memories with these tags decay with age, so durable knowledge is never
  pushed down by age. This is opt-in, and the default behaviour is unchanged.

### Knowledge graph

- **Cleaner entity extraction.** Emphasis and section words (ВАЖНО, TODO, Status), months, weekdays and common
  Russian/Ukrainian sentence starters are no longer extracted as entities.
- **Aliases.** The MCP instructions now suggest `same_as` facts for alternative names.

### Commands

- **`synapstore reindex SRC DST --embedder SPEC`.** Rebuilds a store into a new file with another embedder. It
  keeps tags, timestamps, counters, facts (with remapped provenance), episodes and API tokens, joins memories
  that 0.2 had split into chunks, drops junk entities and leaves out HNSW tombstones. The source is only read.
- **`synapstore eval DB --queries FILE.tsv`.** A recall regression check with snippet-based gold answers
  (@1, @5, MRR, misses).
- **`serve --usage-log` and `synapstore usage`.** Report per-client tool calls, the weakest queries and the
  most-recalled and never-recalled memories.
- **Richer logs and `stats`.** Every MCP tool call logs its client, and `recall` also logs its hits and top
  similarity. `stats` shows `auth_tokens` and `decay.only_tags`.

### Storage

- Two additive tables, `usage_log` and `recall_stats`. The schema version is unchanged, and 0.2 files open in
  place.
