# SynapStore

Single-file, ACID memory store for AI agents. It is written in pure Rust and served over the
[Model Context Protocol](https://modelcontextprotocol.io), either locally on stdio or remotely over HTTP with
token authentication.

One `.redb` file holds three memory layers:

| Layer | What it stores | How it is queried |
|---|---|---|
| **Semantic** | Text chunks + embeddings (384 / 768 / 1536 dims or any other) | HNSW index persisted in redb tables, SIMD cosine (AVX2+FMA / SSE2 / NEON) |
| **Knowledge graph** | Subject–Predicate–Object facts with provenance | SPO / POS / OPS indices, BFS up to depth 2 (configurable to 6) |
| **Episodic** | Append-only log of thoughts, actions, tool results | Time-ordered scans; drives time-decay re-ranking of recall |

Dependency budget: `redb`, `rkyv`, `serde`/`serde_json` (MCP I/O and model metadata) and `thiserror`.
Everything else is implemented in this crate and has no C bindings:

- the HNSW index and the SIMD kernels
- the BERT inference engine and its tokenizers
- SHA-256, the HTTP server, the JSON-RPC layer, the logger, the metrics and the CLI parsing

## Install

One command installs the binary and connects it to every agent it finds on the machine:

```sh
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/vertuzz/synapstore/main/install.sh | sh
# Windows (PowerShell)
irm https://raw.githubusercontent.com/vertuzz/synapstore/main/install.ps1 | iex
```

The script downloads a prebuilt binary from the latest GitHub release and verifies its SHA-256; without a release
for the platform it builds with `cargo`. It then runs `synapstore install`, which:

- creates the store `~/.synapstore/memory.redb`;
- registers the MCP server with each agent found: Claude Code, Codex CLI, OpenCode, Gemini CLI, Cursor,
  VS Code (Copilot), Windsurf and Claude Desktop;
- installs the memory skill (`SKILL.md`, the [Agent Skills](https://agentskills.io) format), which tells the agent
  when to recall and what to memorize. pi has no MCP, so it gets the skill, and the skill drives the
  `synapstore` CLI instead.

Pass options after `sh -s --`, for example `| sh -s -- --embedder multilingual --client claude-code,codex`.

| You want | Command |
|---|---|
| Everything for the agents found | `synapstore install` |
| Only some agents | `synapstore install --client claude-code,codex,pi` (or `--client all`) |
| Only the MCP server / only the skill | `synapstore install --only mcp` / `--only skill` |
| A neural embedder for the store | `synapstore install --embedder minilm` |
| A shared server instead of a local file | `SYNAPSTORE_TOKEN=syn_… synapstore install --url https://memory.example.com` |
| Preview the changes | `synapstore install --dry-run` |
| Undo (keeps the memory) | `synapstore uninstall` |

Config edits keep your formatting and comments, and the original file is saved next to it as
`<file>.synapstore.bak`.

You can also install the binary yourself (`cargo install --git https://github.com/vertuzz/synapstore`) and use
each agent's own mechanism:

| Agent | Command |
|---|---|
| Any agent with skills (Claude Code, Codex, OpenCode, Cursor, pi, …) | `npx skills add vertuzz/synapstore` |
| Claude Code plugin (MCP server + skill) | `/plugin marketplace add vertuzz/synapstore`, then `/plugin install synapstore@synapstore` |
| Gemini CLI extension | `gemini extensions install https://github.com/vertuzz/synapstore` |
| Claude Code, MCP only | `claude mcp add --scope user synapstore -- synapstore mcp` |
| Codex, MCP only | `codex mcp add synapstore -- synapstore mcp` |
| Anything else | `synapstore skill` prints the skill; paste it into `AGENTS.md` or your agent's rules |

Several agents can use the same store at once: each one starts its own `synapstore mcp` process, and redb's
multi-process mode serializes their writes.

### Upgrading

Nothing updates itself. What a new release changes and how it reaches each place:

| Part | Where it lives | How to update |
|---|---|---|
| Store logic, MCP instructions and tool descriptions | the `synapstore` binary that serves MCP | replace the binary; agents pick up the new instructions on their next session |
| Local CLI | `~/.local/bin` (Windows: `%LOCALAPPDATA%\synapstore\bin`) | rerun the install script (`SYNAPSTORE_NO_SETUP=1` only replaces the binary) |
| Skill copies written by `synapstore install` | `~/.claude/skills`, `~/.agents/skills`, … | `synapstore install --only skill` with the new binary; it reports `already up to date` when nothing changed |
| Claude Code plugin, Gemini extension, `npx skills` | read from this repository | update the plugin from `/plugin`, run `gemini extensions update synapstore`, or `npx skills add vertuzz/synapstore` again |

A new binary opens existing store files in place as long as the schema version (`src/tables.rs`) is unchanged.
Update the server before the clients: new tool arguments are optional, so older clients keep working.

## Quick start

```sh
synapstore remember "The user deploys with make release, never from CI." --fact "user|deploys with|make release"
synapstore recall "how do we deploy?"
synapstore graph user

# Expose one memory to several machines over HTTP:
synapstore serve --listen 127.0.0.1:8765 --embedder multilingual
```

## Embeddings

Text needs a vector before it can be indexed. Pick an embedder with `--embedder` when you create a store. The
choice is recorded in the file, so later opens use it automatically. A store never mixes vector spaces: opening
it with a different embedder is refused.

| `--embedder` | What it is | Download | Quality |
|---|---|---|---|
| `lexical` (default) | Feature hashing of words, stems and character trigrams | none | word overlap only |
| `minilm` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), English, 384 dims | 91 MB | good, fast |
| `bge-small` | [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), English, 384 dims | 134 MB | better English retrieval |
| `multilingual` | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), ~100 languages, 384 dims | 471 MB | cross-lingual |
| `hf:<owner>/<repo>[@rev]` | Any BERT-architecture model on the Hub with `model.safetensors` | varies | |
| `<directory>` | A local model directory (`config.json`, `model.safetensors`, `tokenizer.json` or `vocab.txt`) | none | |
| `openai:<model>[@<base-url>]` | Any OpenAI-compatible `/v1/embeddings` endpoint | none | that model's |
| `openrouter:<model>` | Any embedding model on OpenRouter, zero-data-retention routing | none | that model's |

**Built-in models** run on the CPU inside the `synapstore` process, in pure Rust:

- **Tokenizers:** WordPiece and SentencePiece/Unigram, matching Hugging Face `tokenizers` token for token.
- **Inference:** a BERT encoder with a packed AVX2/FMA GEMM kernel.
- **Validation:** `tests/embed_model.rs` compares the output against onnxruntime and HF tokenizers.

The weights download once on first use with the system `curl`, or ahead of time with `synapstore pull <model>`.

- **Integrity:** files are checked against the SHA-256 the Hub publishes (`X-Linked-Etag`).
- **Cache:** downloads go to `~/.cache/synapstore/models`; set `SYNAPSTORE_MODELS` to change it.
- **Hub environment variables:** `HF_ENDPOINT` selects a mirror, `HF_TOKEN` gives access to gated repos, and
  `HF_HUB_OFFLINE=1` forbids downloads.

`synapstore models` lists the models and shows which ones are cached.

Speed on 4 vCPUs (AVX2) with `minilm`:

- **A short query:** about 10 ms.
- **Bulk memorize:** about 215 chunks/s.

`bge-small` and `multilingual` have 12 layers instead of 6, so they run at about half that speed.

**Remote endpoints.** With `openai:nomic-embed-text@http://localhost:11434/v1`, SynapStore uses Ollama's
embeddings API. llama.cpp, vLLM, LM Studio, TEI and OpenAI itself work the same way:
`openai:text-embedding-3-small` uses `https://api.openai.com/v1`. The API key is read from
`SYNAPSTORE_EMBED_API_KEY`, or from `OPENAI_API_KEY`.

**OpenRouter.** `openrouter:<model>` uses `https://openrouter.ai/api/v1` with the key from `OPENROUTER_API_KEY`.
Every request asks OpenRouter to route only to zero-data-retention providers that do not train on the data
(`"provider": {"zdr": true, "data_collection": "deny"}`). On a personal bilingual (Russian/English) memory of
~200 notes and 85 test questions, `openrouter:perplexity/pplx-embed-v1-4b` (open weights, MIT) answered 93% of
questions with the right memory first and 98% within the top five, against 64% / 85% for `multilingual`; it
was also the best of the models tried at cross-language questions (96% top-five).

Remote embedders behave like this:

- **Chunking.** Their context windows hold whole memories, so a memory is cut only above 8,000 characters and
  `recall` returns it in one piece. Local models keep 1,200-character chunks.
- **Retries.** Network errors and HTTP 408/429/5xx are retried with backoff.
- **Offline start.** A store that already records its dimension opens without contacting the endpoint, so the
  server, `stats`, `dump` and backups work without network.
- **Endpoint down.** `recall` falls back to a word-overlap scan and says so (`"embedding": "lexical-fallback"`
  plus a `warning`); `memorize` fails with `embedding_unavailable`, and the agent can retry.

**Switching embedders.** A store's vectors cannot be converted in place. `synapstore reindex OLD NEW
--embedder <SPEC>` builds a new file from an old one, which is only read:

- **Memories** are re-embedded. Tags, entities, timestamps and counters are kept, and memories an older
  version split into several chunks are joined back together when the split is unambiguous.
- **Facts** keep their timestamps and their (remapped) source memory.
- **Episodes and API tokens** are copied as they are.
- **Junk entities** that old heuristics extracted from stop words, months or emphasis are dropped.

Compare the two files with `synapstore eval <DB> --queries questions.tsv`, then swap them.

**Your own vectors.** Agents can always pass `vector` to `memorize` and `recall`. Such a store is pinned to the
`external` space.

## Remote access over HTTP

```sh
synapstore serve memory.redb --listen 0.0.0.0:8765
```

`serve --listen` speaks MCP's **Streamable HTTP** transport (protocol revisions 2025-03-26 to 2025-11-25) at
`/mcp`. `/healthz` is an unauthenticated liveness probe.

**Authentication** uses bearer tokens (RFC 6750). On the first start without tokens, SynapStore creates one and
prints it once on stderr. Manage tokens from another terminal while the server runs:

```sh
synapstore token create memory.redb --name laptop                       # scope write (default)
synapstore token create memory.redb --name dashboard --scope read --expires-days 30
synapstore token list memory.redb
synapstore token revoke memory.redb dashboard                            # by name or id, effective immediately
```

| Scope | Tools |
|---|---|
| `read` | `recall`, `graph_query`, `graph_match`, `recent_events`, `explain`, `stats` |
| `write` | read + `memorize`, `add_facts`, `retract_fact`, `log_event`, `forget` |
| `admin` | write + `repair_index` |

Each token only sees the tools its scope allows in `tools/list`. Calling any other tool returns a `forbidden`
tool error.

Connect a client:

```sh
# All detected agents + skill + CLI (~/.synapstore/remote.json):
SYNAPSTORE_TOKEN=syn_… synapstore install --url https://memory.example.com

# Or register MCP manually in Claude Code:
claude mcp add --scope user --transport http synapstore https://memory.example.com/mcp \
  --header "Authorization: Bearer syn_…"
```

```json
{ "mcpServers": { "synapstore": {
    "type": "http", "url": "https://memory.example.com/mcp",
    "headers": { "Authorization": "Bearer syn_…" } } } }
```

### Self-hosting behind HTTPS

To host a single shared store on a Linux server with a domain pointing to it:

1. **Install the binary** (release tarball from GitHub Releases or `cargo install`) to `/usr/local/bin/synapstore`,
   create a dedicated system user and a private data directory:

   ```sh
   sudo useradd --system --home-dir /var/lib/synapstore --shell /usr/sbin/nologin --user-group synapstore
   sudo install -d -m 0700 -o synapstore -g synapstore /var/lib/synapstore /var/lib/synapstore/models
   ```

2. **Pre-download the embedder and create your first token** before starting the service (pass `SYNAPSTORE_MODELS`
   so `token create` and `serve` share the same cache and no throwaway `default` token is generated at startup):

   ```sh
   sudo -u synapstore SYNAPSTORE_MODELS=/var/lib/synapstore/models \
     /usr/local/bin/synapstore pull multilingual

   sudo -u synapstore SYNAPSTORE_MODELS=/var/lib/synapstore/models \
     /usr/local/bin/synapstore token create /var/lib/synapstore/memory.redb \
     --embedder multilingual --name laptop --scope write
   ```

   Save the printed `syn_…` secret once (only its SHA-256 hash is stored in `memory.redb`). Issue a separate token
   per device or agent so you can revoke any of them independently.

3. **Start the systemd service and TLS reverse proxy.** Copy [`deploy/synapstore.service`](deploy/synapstore.service)
   to `/etc/systemd/system/` (it binds `127.0.0.1:8765` with `--trust-proxy`) and add the block from
   [`deploy/Caddyfile`](deploy/Caddyfile) with your domain to `/etc/caddy/Caddyfile`:

   ```sh
   sudo cp deploy/synapstore.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now synapstore
   sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
   ```

4. **Verify** from your workstation:

   ```sh
   curl -fsS https://memory.example.com/healthz                          # {"status":"ok"}
   curl -i -X POST https://memory.example.com/mcp \
     -H "Content-Type: application/json" -d '{}'                         # 401 Unauthorized
   ```

5. **Use it** from any agent or the CLI:

   ```sh
   SYNAPSTORE_TOKEN=syn_… synapstore install --url https://memory.example.com
   synapstore remember "Deployed SynapStore" && synapstore recall deployed
   ```

   For scripts or CI without `synapstore install`, set `SYNAPSTORE_URL=https://memory.example.com/mcp` and
   `SYNAPSTORE_TOKEN=syn_…` (for example in a git-ignored `.env` file).

6. **Upgrade** later by backing up the store, swapping the binary and restarting:

   ```sh
   sudo systemctl stop synapstore
   sudo cp -a /var/lib/synapstore/memory.redb /var/lib/synapstore/memory.redb.bak
   sudo install -m 0755 synapstore-x86_64-linux/synapstore /usr/local/bin/synapstore   # from the release tarball
   sudo systemctl start synapstore && curl -fsS https://memory.example.com/healthz
   ```

### Security model

- **Tokens:** 256 bits from the OS CSPRNG, prefixed `syn_` so secret scanners recognize them.
- **Storage:** only the SHA-256 of each token is stored. A copied `.redb` file contains no usable credential.
- **Checks on every request:** each request is checked against the file. Revocation and expiry take effect
  immediately, with no restart.
- **Sessions:** an MCP session (`Mcp-Session-Id`) is bound to the token that created it. Another token cannot
  use it, and idle sessions expire after an hour.
- **Brute-force throttling:** an address that fails authentication 20 times within 5 minutes gets `429` until
  the window ends. Behind a proxy, pass `--trust-proxy` so the client address is taken from `X-Forwarded-For`.
- **Browsers:** requests carrying an `Origin` header are rejected unless that origin is in the
  `--allow-origin` list, which defends against DNS rebinding. CORS preflight is answered for allowed origins.
- **Limits:**
  - request bodies: 8 MiB, configurable with `--max-body-mb`
  - headers: at most 100, 32 KiB in total
  - connections: 30 s idle timeout
  - workers: a bounded pool, with `503` when it is saturated
- **No-auth mode:** `--no-auth` is refused unless the server binds to a loopback address.
- **TLS** is not built in, because it would break the dependency budget. On a public address, put a TLS reverse
  proxy in front of the server. [`deploy/Caddyfile`](deploy/Caddyfile) configures automatic Let's Encrypt
  certificates and proxies only `/mcp*` and `/healthz`, and [`deploy/synapstore.service`](deploy/synapstore.service)
  is a hardened systemd unit for it.

## MCP tools

| Tool | Purpose |
|---|---|
| `memorize(text, vector?, entities?, facts?, supersedes?, tags?, …)` | Upsert a memory: chunk, embed, index, link entities, assert facts, log an episode; `supersedes` forgets older memories in the same transaction |
| `recall(query? \| vector?, top_k, max_age?, tags?, …)` | HNSW search re-ranked by time-decay |
| `graph_query(subject, depth, direction?, predicate?)` | BFS over the knowledge graph + memories mentioning the entity |
| `graph_match(subject?, predicate?, object?)` | Exact pattern lookup (omitted = wildcard) |
| `add_facts`, `retract_fact` | Direct graph edits |
| `log_event`, `recent_events` | Episodic log |
| `forget(ids? \| text?)` | Delete memories and retract the facts they asserted |
| `explain(target_id \| target_text, query \| vector)` | Whitebox trace: exact rank, HNSW reachability, decay, filters → verdict |
| `repair_index` | Reconnect unreachable HNSW nodes (also runs automatically) |
| `stats(detailed?)` | Counts, index structure, embedder, latency histograms, counters |

When a tool fails, it returns `isError: true` with `structuredContent.error = {kind, message}`, and the server
keeps running. The stdio server logs to `<db>.log` (logfmt, rotated at 10 MiB, 5 files kept), so stdout carries
only protocol traffic.

Every tool call is logged as a `tool` line with the caller (`client=<token name>/<clientInfo.name>`), and for
`recall` the number of hits, the top similarity and the embedding used. With `serve --usage-log` the calls are
also kept in the store (last 20,000). `synapstore usage` then reports:

- calls per client and tool;
- the weakest recall queries (lowest top similarity), which are candidates for missing memories;
- the most recalled memories, and memories that were never recalled.

**Decay by tag.** By default every memory fades with age. `--decay-tags status,plan,todo,news,snapshot*` limits
decay to memories carrying one of those tags (`x*` matches by prefix). Durable knowledge, such as people,
contracts and decisions, then ranks by similarity alone however old it is.

## CLI

The memory commands are the same tool calls, with compact text output (`--json` for raw results). They use the
default store, `--db <FILE>` or `$SYNAPSTORE_DB`; with `--url` / `$SYNAPSTORE_URL` (token in
`$SYNAPSTORE_TOKEN`) they call a remote server instead.

```
synapstore recall <QUERY> [-k N] [--max-age 7d] [--tag T]
synapstore remember <TEXT|-> [--fact "s|p|o"]... [--tag T]... [--source S] [--supersedes ID]...
synapstore graph <ENTITY> [--depth N]          synapstore facts [--subject S] [--predicate P] [--object O]
synapstore fact add|rm <S> <P> <O>             synapstore forget <ID>...
synapstore log <TEXT|-> [--kind K]             synapstore events [--limit N] [--kind K]
synapstore explain <ID> [QUERY]                synapstore call <TOOL> [JSON]

synapstore install | uninstall | skill         Connect agents (see Install)
synapstore mcp [DB]                            Serve MCP over stdio
synapstore serve [DB] --listen <ADDR>          Serve MCP over HTTP with token auth
synapstore token create|list|revoke            Manage API tokens
synapstore models | pull <MODEL>               List / download embedding models
synapstore stats [DB] [--detailed]             Statistics as JSON
synapstore dump [DB] [--vectors]               Export every table as JSON lines (whitebox export)
synapstore repair [DB]                         Reconnect unreachable HNSW nodes
synapstore synth [DB] [--records N]            Load a synthetic agent-memory dataset
```

Run `synapstore --help` for all flags. They include `--embedder`, `--allow-origin`, `--trust-proxy`,
`--ef-search`, `--m`, `--half-life-days`, `--no-decay`, `--cache-mb` and the logging options.

## Design

- **Deterministic ids and upserts.** A chunk's id is `XXH64(normalize(text))`, which ignores case and
  whitespace. Memorizing the same content again updates the stored record and does not create a duplicate.
- **Single writer.** Every mutation goes through an mpsc channel to one writer thread.
  - That thread group-commits up to 64 queued operations in one redb transaction. If any of them fails, it falls
    back to one transaction per operation.
  - Within a transaction, HNSW vectors and nodes are served from a write-back cache, and each modified node is
    written once at commit.
  - Several *processes* can share a file: `try_begin_write` contention is retried with exponential backoff
    (5 retries, 20 ms base, 2 s cap).
- **Embedding off the writer.** Texts are chunked and embedded on the caller's thread, so slow models never block
  the single writer. `memorize_batch` spreads that work over all cores.
- **Zero-copy reads.** Records are rkyv archives that are validated and read in place, inside redb's page
  buffers. The SIMD kernels read vectors unaligned, directly from those buffers. `tests/zero_copy.rs` asserts
  **0 heap bytes per read**. redb uses its own page cache rather than `mmap`, so "zero-copy" means no copy
  beyond redb's cache.
- **HNSW in redb.** Parameters: M = 16, M0 = 32, ef_construction = 100, ef_search = 64.
  - Neighbors are chosen with the diversity heuristic.
  - Forgotten nodes stay in the graph as tombstones, so it stays navigable.
  - A Lucene-style connectivity repair runs automatically, amortized over inserts.
  - Recall widens `ef` automatically when filters discard candidates.
- **Time-decay.** `score = sim × (floor + (1 − floor)·freshness + boost·freshness)`, where
  `freshness = 0.5^(age / half_life)`. Defaults: half-life 30 d, floor 0.5, boost 0.1. Near-exact matches are
  protected from the age penalty.
- **Graph.** `SPO (s,p)→[o]`, `POS (p,o)→[s]` and `OPS (o,p)→[s]`, with packed sorted id lists. `TRIPLES` holds
  provenance, and `ENTITY_CHUNKS` links entities to memories.
  - Facts are never derived from text: they come from `memorize(facts)` or `add_facts`. Entity *links* are
    automatic (explicit `entities`, or runs of capitalized words).
  - A fact's provenance is the latest memory that asserted it. `forget` retracts the facts whose provenance is a
    deleted memory (a scan of `TRIPLES`; forgetting is rare), and facts added without a source are kept.

Source layout: `src/{math,hnsw,graph,store,embed,http,auth,mcp,synth}`. Every module is under 300 lines (CI
enforces it).

## Benchmarks

Measured on a 4-vCPU cloud VM with an AVX2 release build (`cargo bench --bench hnsw`,
`cargo bench --bench synthetic`). Results are written to `bench/results/*.json`.

**HNSW: 10,000 vectors × 384 dims**

| Metric | Value |
|---|---|
| recall@10 (vs. brute force) | 1.00 |
| query p50 / p95 | 444 µs / 568 µs |
| build | 1,587 inserts/s, fully durable (was 558/s before the write-back cache) |
| peak RSS | 105 MiB |
| SIMD speedup over scalar | 7.5× (384), 7.8× (768), 8.2× (1536) |

**Synthetic agent workload: 329k stored records (dim 128)**

The workload has 66.7k memories, 49.5k facts and 200k episodes.

| Metric | Value |
|---|---|
| recall@10 | 1.00 |
| recall p50 / p95 (with decay) | 1.25 ms / 1.48 ms |
| graph BFS depth-2 p50 / p95 | 1.77 ms / 2.50 ms (0 mismatches over 62k edges vs. reference BFS) |
| episodic scan p50 | 29 µs |
| decay probes | 20/20 |
| zero-copy reads | 0 B allocated per read |
| unreachable HNSW nodes | 0 |
| file / peak RSS | 272 MiB / 399 MiB |
| load | 840 memories/s (was 310/s), 47k events/s |

## Testing

```sh
cargo test --all-targets                         # unit, integration, HTTP/auth, 100k-record E2E, concurrency
SYNAPSTORE_FUZZ_ITERS=20000 cargo test --release --test fuzz_hnsw --test fuzz_graph --test fuzz_mcp
SYNAPSTORE_SYNTH_RECORDS=60000 cargo test --release --test synthetic
cargo build && python3 -m unittest discover -s tests/python -v   # black-box MCP over stdio and HTTP
synapstore pull minilm && SYNAPSTORE_TEST_MODEL_DOWNLOAD=1 cargo test --release --test embed_model
```

On pull requests, CI runs the synthetic benchmark on the base commit and the head commit on the same runner.
`scripts/perf_gate.py` fails the PR if any of these regress:

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

Apache-2.0. See [LICENSE](LICENSE). The downloadable models keep their own licenses: Apache-2.0 for MiniLM and
MIT for BGE and E5.
