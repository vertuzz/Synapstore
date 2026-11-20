# SynapStore

Single-file, ACID memory store for AI agents. It is written in pure Rust and served over the
[Model Context Protocol](https://modelcontextprotocol.io) on stdio.

One `.redb` file holds three memory layers:

| Layer | What it stores | How it is queried |
|---|---|---|
| **Semantic** | Text chunks + embeddings (384 / 768 / 1536 dims or any other) | HNSW index persisted in redb tables, SIMD cosine (AVX2+FMA / SSE2 / NEON) |
| **Knowledge graph** | Subject–Predicate–Object facts with provenance | SPO / POS / OPS indices, BFS up to depth 2 (configurable to 6) |
| **Episodic** | Append-only log of thoughts, actions, tool results | Time-ordered scans; drives time-decay re-ranking of recall |

Dependency budget: `redb`, `rkyv`, `serde`/`serde_json` (used only for MCP I/O) and `thiserror`. Everything
else is implemented in this crate and has no C bindings: the HNSW index, the SIMD kernels, XXH64, the
JSON-RPC layer, the logger, the metrics and the CLI parsing.

## Install

```sh
cargo install --git https://github.com/vertuzz/Synapstore
# or from a checkout
cargo install --path .
```

## Use with an MCP client

```sh
synapstore --mcp ~/.agent/memory.redb
```

Claude Code:

```sh
claude mcp add synapstore -- synapstore --mcp ~/.agent/memory.redb
```

Generic `mcpServers` config (Cursor, Claude Desktop, …):

```json
{
  "mcpServers": {
    "synapstore": { "command": "synapstore", "args": ["--mcp", "/home/me/.agent/memory.redb"] }
  }
}
```

Logs go to `<db>.log` (logfmt, rotated at 10 MiB, 5 files kept) so stdout carries only protocol traffic.
Supported protocol versions: `2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`.

### Tools

| Tool | Purpose |
|---|---|
| `memorize(text, vector?, entities?, facts?, tags?, …)` | Upsert a memory: chunk, embed, index, link entities, assert facts, log an episode |
| `recall(query? \| vector?, top_k, max_age?, tags?, …)` | HNSW search re-ranked by time-decay |
| `graph_query(subject, depth, direction?, predicate?)` | BFS over the knowledge graph + memories mentioning the entity |
| `graph_match(subject?, predicate?, object?)` | Exact pattern lookup (omitted = wildcard) |
| `add_facts`, `retract_fact` | Direct graph edits |
| `log_event`, `recent_events` | Episodic log |
| `forget(ids? \| text?)` | Delete memories |
| `explain(target_id \| target_text, query \| vector)` | Whitebox trace: exact rank, HNSW reachability, decay, filters → verdict |
| `repair_index` | Reconnect unreachable HNSW nodes (also runs automatically) |
| `stats(detailed?)` | Counts, index structure, latency histograms, counters |

When a tool fails it returns `isError: true` with `structuredContent.error = {kind, message}`; the server
keeps running. If `vector` is omitted, SynapStore uses its built-in **lexical** embedder: feature hashing of
words, stems, bigrams and character trigrams. This needs no model, but it measures word overlap, not
meaning. For semantic recall, pass vectors from a real embedding model. A store is pinned to one embedding
space (source and dimension) by its first write, and mixing spaces is rejected with `embedding_mismatch`.

### CLI

```
synapstore --mcp <DB>                 Serve MCP over stdio
synapstore stats <DB> [--detailed]    Statistics as JSON
synapstore dump <DB> [--vectors]      Export every table as JSON lines (whitebox export)
synapstore repair <DB>                Reconnect unreachable HNSW nodes
synapstore synth <DB> [--records N]   Load a synthetic agent-memory dataset
```

Run `synapstore --help` for the tuning flags: `--ef-search`, `--m`, `--ef-construction`,
`--half-life-days`, `--no-decay`, `--cache-mb` and the logging options.

## Design

- **Deterministic ids and upserts.** A chunk's id is `XXH64(normalize(text))` (case- and
  whitespace-insensitive). Entities and predicates get ids from their own namespaces. Memorizing the same
  content again updates the stored record and does not create a duplicate. Ids are 16-character hex
  strings on the wire.
- **Single writer.** Every mutation goes through an mpsc channel to one writer thread. That thread
  group-commits up to 64 queued operations in one redb transaction, falling back to one transaction per
  operation if any of them fails. Multiple *processes* can share a file: `try_begin_write` contention is
  retried with exponential backoff (5 retries, 20 ms base, 2 s cap) before `busy` is returned.
- **Zero-copy reads.** Records are rkyv archives that are validated and read in place, inside redb's page
  buffers. Vectors are stored as raw little-endian `f32`, and the SIMD kernels read them unaligned
  directly from those buffers. `tests/zero_copy.rs` asserts **0 heap bytes per read** with a counting
  allocator. redb uses its own page cache rather than `mmap`, so "zero-copy" means no copy beyond redb's
  cache.
- **HNSW in redb.** Each node is one record (`HnswNode { level, layers: [[Link{id, sim}]] }`) with
  M = 16, M0 = 32, ef_construction = 100 and ef_search = 64. A node's level is derived from its id, so
  rebuilds are reproducible.
  - Neighbors are chosen with the diversity heuristic, which keeps pruned candidates.
  - Forgotten nodes stay in the graph as tombstones, so the graph stays navigable.
  - A Lucene-style connectivity repair runs automatically, amortized (every max(1000, n/8) inserts).
  - Recall widens `ef` automatically when filters (`max_age`, `tags`) discard candidates.
- **Time-decay.** `freshness = 0.5^(age / half_life)` and
  `score = sim × (floor + (1 − floor)·freshness + boost·freshness)`, with defaults half-life 30 d,
  floor 0.5 and boost 0.1. Near-exact matches (cosine ≥ 0.92 → 0.99, smoothstep) are protected from the
  age penalty, so an old fact that matches exactly still beats a fresh fact that is only loosely related.
- **Graph.** `SPO (s,p)→[o]`, `POS (p,o)→[s]` and `OPS (o,p)→[s]` (for in-edges). Values are packed,
  sorted `u64` lists. `TRIPLES` holds provenance, and `ENTITY_CHUNKS` links entities to memories.

Source layout: `src/{math,hnsw,graph,store,mcp,synth}`. Every module is under 300 lines (CI enforces it).

## Benchmarks

The numbers below were measured on a 4-vCPU cloud VM with an AVX2 release build.
Reproduce them with `cargo bench --bench hnsw` and `cargo bench --bench synthetic`. Results are written to
`bench/results/*.json`.

**HNSW benchmark: 10,000 vectors × 384 dims**

| Metric | Value |
|---|---|
| recall@10 (vs. brute force) | 1.00 |
| query p50 / p95 | 443 µs / 605 µs |
| build | 558 inserts/s (fully durable, one commit per batch) |
| peak RSS | 80 MiB |
| SIMD speedup over scalar | 7.5× (384), 7.8× (768), 8.2× (1536) |

**Synthetic agent workload: 200k requested → 329k stored records (dim 128)**

The workload has 66.7k memories, 49.5k facts and 200k episodes.

| Metric | Value |
|---|---|
| recall@10 | 1.00 |
| recall p50 / p95 (with decay) | 1.41 ms / 1.87 ms |
| graph BFS depth-2 p50 / p95 | 1.86 ms / 2.41 ms (0 mismatches over 62k edges vs. reference BFS) |
| episodic scan p50 | 29 µs |
| decay probes | 20/20 |
| zero-copy reads | 0 B allocated per read |
| unreachable HNSW nodes | 0 |
| file / peak RSS | 272 MiB / 341 MiB |
| load | 310 memories/s, 50k events/s |

Writes are bounded by HNSW insertion. Each insert does a full beam search plus backlink pruning inside a
durable transaction. Use `memorize_batch` (library API) for bulk loads.

## Testing

```sh
cargo test --all-targets                         # unit, integration, 100k-record E2E, concurrency
SYNAPSTORE_FUZZ_ITERS=20000 cargo test --release --test fuzz_hnsw --test fuzz_graph --test fuzz_mcp
SYNAPSTORE_SYNTH_RECORDS=60000 cargo test --release --test synthetic
cargo build && python3 -m unittest discover -s tests/python -v   # black-box MCP over stdio
cargo bench --bench synthetic && python3 scripts/perf_gate.py bench/results/synthetic.json
```

On pull requests, CI runs the synthetic benchmark on the base commit and the head commit on the same
runner. `scripts/perf_gate.py` fails the PR if any of these regress:

- latency by more than 25 %
- peak RSS by more than 10 %
- load throughput by more than 25 %
- recall@10 by more than 0.005

It also fails if any absolute invariant breaks:

- recall@10 ≥ 0.95
- zero graph mismatches
- 100 % decay probes
- zero unreachable nodes
- zero-copy reads

## License

Apache-2.0. See [LICENSE](LICENSE).
