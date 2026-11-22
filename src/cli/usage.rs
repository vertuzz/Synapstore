pub const USAGE: &str = "\
SynapStore — single-file ACID memory for AI agents (HNSW + knowledge graph + episodic log).

SET UP YOUR AGENTS:
    synapstore install [--client <LIST>] [--only mcp|skill] [--embedder <SPEC>] [--dry-run]
                                          Register the MCP server and the memory skill with every
                                          agent found (claude-code, codex, opencode, gemini, cursor,
                                          vscode, windsurf, claude-desktop, pi; or `all`)
    synapstore install --url <URL>        Same, pointing agents at a remote server (token from
                                          SYNAPSTORE_TOKEN is written into their configs)
    synapstore uninstall [--client <LIST>]
    synapstore skill                      Print the agent skill (SKILL.md) to stdout

MEMORY (store: --db, else $SYNAPSTORE_DB, else ~/.synapstore/memory.redb):
    synapstore recall <QUERY> [-k N] [--max-age 7d] [--tag T]
    synapstore remember <TEXT|-> [--fact \"s|p|o\"]... [--tag T]... [--source S]
                        [--supersedes <ID>]...   Replace older memories (and their facts)
    synapstore graph <ENTITY> [--depth N]
    synapstore facts [--subject S] [--predicate P] [--object O]
    synapstore fact add|rm <S> <P> <O>
    synapstore log <TEXT|-> [--kind K]    Append to the episodic log
    synapstore events [--limit N] [--kind K]
    synapstore forget <ID>...
    synapstore explain <ID> [QUERY]       Why a memory is (not) recalled
    synapstore call <TOOL> [JSON]         Any MCP tool, raw JSON result
      --json                              Machine-readable output
      --url <URL>                         Use a remote server ($SYNAPSTORE_URL; token: $SYNAPSTORE_TOKEN)

SERVERS AND ADMIN ([DB] defaults like above):
    synapstore mcp [DB]                   Serve MCP over stdio (what agents launch)
    synapstore serve [DB] --listen <ADDR> Serve MCP over HTTP with bearer-token auth
    synapstore token create [DB] --name <NAME> [--scope read|write|admin] [--expires-days N]
    synapstore token list [DB]            List API tokens (secrets are never stored)
    synapstore token revoke [DB] <ID|NAME>
    synapstore models                     List built-in embedding models
    synapstore pull <MODEL>               Download a model ahead of time
    synapstore stats [DB] [--detailed]    Print store statistics as JSON
    synapstore dump [DB] [--vectors]      Export every table as JSON lines to stdout
    synapstore repair [DB]                Reconnect unreachable HNSW nodes
    synapstore reindex <SRC> <DST> --embedder <SPEC> [--no-merge]
                                          Rebuild a store into a new file with another embedder
                                          (keeps facts, events, tokens; SRC is only read)
    synapstore eval [DB] --queries <FILE.tsv> [-k N]
                                          Recall regression check: `query<TAB>snippet|snippet`
    synapstore usage [DB] [--limit N]     Who called what, weakest queries, never-recalled memories
                                          (needs `serve --usage-log`)
    synapstore synth [DB] [--records N] [--seed S] [--dim D]
                                          Load a synthetic agent-memory dataset
    synapstore --version | --help

EMBEDDINGS:
    --embedder <SPEC>       Text embedder for a new store (remembered in the file):
                              lexical            built-in word hashing, no model (default)
                              minilm             English neural model, downloaded once (~90 MB)
                              bge-small          stronger English model (~130 MB)
                              multilingual       ~100 languages (~470 MB)
                              hf:<owner>/<repo>  any BERT model with safetensors on the Hub
                              <DIR>              a local model directory
                              openai:<model>[@<base-url>]  OpenAI-compatible endpoint
                                                 (Ollama: openai:nomic-embed-text@http://localhost:11434/v1)
                              openrouter:<model> OpenRouter, zero-data-retention providers only
                                                 (key: OPENROUTER_API_KEY), e.g.
                                                 openrouter:perplexity/pplx-embed-v1-4b
    --dim <N>               Lexical embedder dimension for a new store (default 384)

HTTP SERVER (serve --listen):
    --listen <ADDR>         Address to bind, e.g. 127.0.0.1:8765 or 0.0.0.0:8765
    --allow-origin <ORIGIN> Allow browser requests from ORIGIN (repeatable)
    --threads <N>           Worker threads (default 16)
    --max-body-mb <N>       Request size limit (default 8)
    --trust-proxy           Use X-Forwarded-For for the client address (behind a proxy)
    --no-auth               Disable token auth (refused unless bound to 127.0.0.1/::1)
    --usage-log             Record tool calls per client in the store (`synapstore usage`)

OPTIONS:
    --db <FILE>             Store file for any command
    --log <FILE>            Log file (default for servers: <DB>.log)
    --no-log                Disable file logging
    --log-level <LEVEL>     error | warn | info | debug (default info)
    --log-max-mb <N>        Rotate after N MiB (default 10)
    --log-keep <N>          Rotated files to keep (default 5)
    --ef-search <N>         HNSW search beam width (default 64)
    --m <N>                 HNSW links per node, fixed at creation (default 16)
    --ef-construction <N>   HNSW build beam width, fixed at creation (default 100)
    --half-life-days <X>    Time-decay half-life (default 30)
    --no-decay              Disable time-decay
    --decay-tags <LIST>     Only memories with one of these tags decay (comma list, `x*` =
                            prefix); others rank by similarity alone (default: all decay)
    --cache-mb <N>          redb page cache (default 256)
";
