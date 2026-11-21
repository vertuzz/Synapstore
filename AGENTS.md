# Instructions for autonomous agents

## Working on this repository

- Read `CONTRIBUTING.md`. The rules there (dependency budget, 300-line files, deny(warnings)) are enforced by CI.
- Fast loop: `cargo test --lib` (seconds). Full loop: `cargo test --all-targets` (a few minutes; the 100k E2E
  and concurrency tests dominate).
- Tests that need scale read env vars: `SYNAPSTORE_FUZZ_ITERS`, `SYNAPSTORE_SYNTH_RECORDS`,
  `SYNAPSTORE_BENCH_RECORDS`. Keep defaults small; CI raises them.
- Never weaken an assertion to make a test pass. The decay probes, graph-vs-reference BFS, HNSW reachability
  and zero-copy allocation checks are product invariants.
- Map of the code:
  - `src/store/` — public API (`SynapStore`), write ops (`apply.rs`), the writer thread (`writer.rs`),
    embedder selection (`space.rs`) and API tokens (`tokens.rs`).
  - `src/hnsw/` — index search, insert (with the per-transaction write-back cache), repair, inspection.
  - `src/embed/` — the `Embedder` trait; `lexical`, `bert/` (pure-Rust BERT and its tokenizers), `fetch.rs`
    (model registry and download), `remote.rs` (OpenAI-compatible endpoints).
  - `src/http/` + `src/auth/` — Streamable HTTP transport and bearer-token authentication.
  - `src/graph/` — triple store writer and BFS reader.
  - `src/mcp/` — JSON-RPC framing, tool schemas (`tools.rs`), handlers.
  - `src/synth/` — deterministic synthetic world + correctness checks used by tests and benches.

## Using SynapStore as your memory

- `memorize` durable knowledge: decisions, user preferences, facts about the codebase. Include `facts` as
  `[subject, predicate, object]` triples when the relation matters; entity names are matched
  case-insensitively.
- `log_event` for the running trace (tool results, thoughts). It is cheap and append-only.
- `recall` before acting on a task you may have seen before; pass `max_age` to only consider recent context.
- `graph_query` to expand from an entity to related facts (depth 2 by default).
- When a memory you expect is missing, call `explain` with its id and your query instead of guessing.
- Pass `vector` from your embedding model if you have one. Otherwise the store's embedder is used: `lexical`
  by default, or a neural model chosen with `--embedder` when the store was created. Do not mix your own
  vectors and text queries in one store.
- Over HTTP, your token's scope decides which tools you see. A `forbidden` error means you need a token with a
  wider scope; do not retry it.
