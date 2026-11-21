---
name: synapstore
description: Long-term memory that persists across sessions and agents (SynapStore). Use it to recall what is already known before starting a task, and to save durable knowledge you learn - user preferences, decisions, facts about the project, people and systems. Trigger on "remember", "recall", "what do you know about", "last time", "as before", or whenever context from earlier sessions would help.
license: Apache-2.0
compatibility: Needs the synapstore binary (MCP server or CLI) - https://github.com/vertuzz/synapstore
metadata:
  homepage: https://github.com/vertuzz/synapstore
---

# SynapStore memory

SynapStore is a single-file memory store shared by every agent on this machine (or on a server). It keeps
three kinds of memory: semantic memories (searchable by meaning, fresher ones rank higher), a knowledge graph
of `subject - predicate - object` facts, and an append-only event log.

## Which interface to use

1. **MCP tools** - if tools named `memorize`, `recall`, `graph_query`, `log_event` (possibly prefixed, e.g.
   `mcp__synapstore__recall`) are available, use them.
2. **CLI** - otherwise run the `synapstore` command in the shell. It works the same way and prints compact
   text (add `--json` for machine-readable output). If `synapstore` is not on PATH, tell the user to install
   it: `curl -fsSL https://raw.githubusercontent.com/vertuzz/synapstore/main/install.sh | sh`.

## When to use it

- **Start of a task**: `recall` with a short description of the task or the entity involved. Use what comes
  back; do not ask the user for things memory already answers.
- **You learn something durable**: `memorize` it - a preference, a decision and its reason, how a system is
  set up, a fix that worked. One self-contained statement per memory, written so it makes sense without the
  current conversation ("The user deploys with `make release`, never from CI", not "they said yes").
- **Relations matter**: add facts as `[subject, predicate, object]` triples, then `graph_query` an entity to
  see everything connected to it.
- **Running trace** (optional): `log_event` for notable actions or tool results; `recent_events` reads them
  back newest first.
- Do **not** store secrets, tokens, passwords or large file contents. Do not memorize what is already in the
  repository - store the non-obvious knowledge around it.

## CLI reference

```sh
synapstore recall "how does the user like to deploy?"           # top 5 memories, fresh ones first
synapstore recall "build failures" -k 10 --max-age 7d             # only the last 7 days
synapstore remember "The user prefers Postgres over MySQL for new services." \
    --fact "user|prefers|Postgres" --tag preference
synapstore graph Postgres                                         # facts and memories around an entity
synapstore facts --predicate prefers                              # exact fact lookup (any position)
synapstore log "Ran the migration on staging; 3 tables changed" --kind action
synapstore events --limit 10
synapstore forget <ID>                                            # ids are printed by recall/remember
synapstore explain <ID> "query"                                   # why a memory is (not) recalled
echo "long text" | synapstore remember -                          # read the memory from stdin
```

Every command accepts `--json`. The store is `~/.synapstore/memory.redb` unless `--db PATH` or
`SYNAPSTORE_DB` says otherwise; `SYNAPSTORE_URL` (+ `SYNAPSTORE_TOKEN`) points the CLI at a remote server.

## MCP tool reference

| Tool | Use |
|---|---|
| `recall` | `{query, top_k?, max_age?, tags?}` semantic search with time decay |
| `memorize` | `{text, facts?, tags?, source?}` store or update a memory (same text = same id) |
| `graph_query` | `{subject, depth?}` facts within N hops plus linked memories |
| `graph_match` | `{subject?, predicate?, object?}` exact fact lookup |
| `add_facts` / `retract_fact` | edit the knowledge graph directly |
| `log_event` / `recent_events` | episodic log |
| `forget` | `{ids}` or `{text}` delete memories |
| `explain` | `{target_id, query}` debug a missing recall instead of guessing |

A `forbidden` error means the token is read-only for that tool; do not retry it.
