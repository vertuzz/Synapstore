## What & why

<!-- One paragraph: the problem and the change. -->

## Memory layers touched

- [ ] Semantic (HNSW / SIMD)
- [ ] Knowledge graph (SPO / POS / OPS)
- [ ] Episodic log / time-decay
- [ ] MCP surface (tool schemas, JSON-RPC)
- [ ] Storage format (tables, rkyv schemas) — **requires a `SCHEMA_VERSION` bump**

## Checklist

- [ ] `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` are clean
- [ ] `cargo test --all-targets` passes; new behaviour has a test
- [ ] Every module stays under 300 lines (`scripts/check_file_length.sh`)
- [ ] No new crate dependencies (budget: redb, rkyv, serde, serde_json, thiserror)
- [ ] Perf gate passes (`scripts/perf_gate.py`); numbers pasted below if hot paths changed

## Benchmarks (if relevant)

```
```
