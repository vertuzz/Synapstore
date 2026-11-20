# Contributing

## Ground rules

1. **No new dependencies.** The budget is `redb`, `rkyv`, `serde`, `serde_json` (MCP I/O only) and `thiserror`.
   Algorithms are implemented in-tree.
2. **Pure Rust.** No C bindings, no `build.rs` that compiles native code.
3. **≤ 300 lines per file** under `src/`, `tests/`, `benches/` (`scripts/check_file_length.sh`). Split modules
   by responsibility rather than by line count.
4. **Warnings are errors.** `src/lib.rs` has `#![deny(warnings)]`, CI runs `clippy --all-targets -D warnings`
   on x86_64 and aarch64.
5. **Storage format changes** (tables in `src/tables.rs`, rkyv structs in `src/model.rs`) must bump
   `SCHEMA_VERSION` and come with a test that opens an old file or rejects it clearly.
6. **No globals.** Logger, metrics and config are passed explicitly; tests run in parallel.

## Before opening a PR

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
scripts/check_file_length.sh
cargo test --all-targets
cargo build && python3 -m unittest discover -s tests/python -v
```

If you touched `src/hnsw`, `src/math`, `src/store/recall.rs` or `src/graph`, also run the fuzzers with a
higher iteration count and the synthetic benchmark + perf gate:

```sh
SYNAPSTORE_FUZZ_ITERS=20000 cargo test --release --test fuzz_hnsw --test fuzz_graph
git stash && cargo bench --bench synthetic && cp bench/results/synthetic.json /tmp/base.json && git stash pop
cargo bench --bench synthetic && python3 scripts/perf_gate.py bench/results/synthetic.json /tmp/base.json
```

## Debugging retrieval

- `explain` (MCP tool) tells you exact rank vs. HNSW rank, whether the target is reachable from the entry
  point, its decay factor and which filter dropped it.
- `synapstore stats <db> --detailed` prints the level histogram, mean degree and reachability.
- `synapstore dump <db>` exports every table as JSON lines.
- Set `--log-level debug` to log every write transaction (ops, retries, latency).
