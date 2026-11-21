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
  see everything connected to it. See "Keeping the graph up to date" below.
- **Something changed or was wrong**: memorize the corrected statement and pass the old memory's id in
  `supersedes` (CLI: `--supersedes <ID>`). The old memory and the facts it asserted are removed in the same
  write, so recall and the graph never show both versions.
- **Running trace** (optional): `log_event` for notable actions or tool results; `recent_events` reads them
  back newest first.
- Do **not** store secrets, tokens, passwords or large file contents. Do not memorize what is already in the
  repository - store the non-obvious knowledge around it.

## Keeping the graph up to date

The store never extracts facts from text: the graph only contains the triples agents pass. It updates
itself only in one way - `forget` (and `supersedes`) also retracts the facts that the deleted memory asserted.

- **Add facts with every memory that states a durable relation**: people and their roles, family, where
  someone lives or works, what they own, projects and their stack, important dates. Example: "Dana is the CTO
  of Acme since 2025" → `[Dana, role, CTO]`, `[Dana, works_at, Acme]`.
- **Reuse names**: check `graph_match` / `synapstore facts --subject X` before inventing a new spelling of an
  entity. Names match case-insensitively but "Bob" and "Robert" are different entities. Use one canonical
  entity for the user (their name if known).
- **Short snake_case predicates**: `works_at`, `role`, `parent_of`, `married_to`, `lives_in`, `owns`, `uses`,
  `builds`, `born_on`. Prefer an existing predicate over a synonym.
- **No fast-changing state** as facts (prices, today's status, counters) - keep those in memory text.
- **Corrections**: use `supersedes` on `memorize`. For a fact that was added on its own with `add_facts`,
  `retract_fact` the old triple first, then add the new one.

## CLI reference

```sh
synapstore recall "how does the user like to deploy?"           # top 5 memories, fresh ones first
synapstore recall "build failures" -k 10 --max-age 7d             # only the last 7 days
synapstore remember "The user prefers Postgres over MySQL for new services." \
    --fact "user|prefers|Postgres" --tag preference
synapstore remember "The user now prefers SQLite for small tools." \
    --fact "user|prefers|SQLite" --supersedes <ID>                # replace an outdated memory and its facts
synapstore graph Postgres                                         # facts and memories around an entity
synapstore facts --predicate prefers                              # exact fact lookup (any position)
synapstore log "Ran the migration on staging; 3 tables changed" --kind action
synapstore events --limit 10
synapstore forget <ID>                                            # also retracts the facts it asserted
synapstore explain <ID> "query"                                   # why a memory is (not) recalled
echo "long text" | synapstore remember -                          # read the memory from stdin
```

Every command accepts `--json`. The store is `~/.synapstore/memory.redb` unless `--db PATH` or
`SYNAPSTORE_DB` says otherwise; `SYNAPSTORE_URL` (+ `SYNAPSTORE_TOKEN`) points the CLI at a remote server.

## MCP tool reference

| Tool | Use |
|---|---|
| `recall` | `{query, top_k?, max_age?, tags?}` semantic search with time decay |
| `memorize` | `{text, facts?, supersedes?, tags?, source?}` store or update a memory (same text = same id) |
| `graph_query` | `{subject, depth?}` facts within N hops plus linked memories |
| `graph_match` | `{subject?, predicate?, object?}` exact fact lookup |
| `add_facts` / `retract_fact` | edit the knowledge graph directly |
| `log_event` / `recent_events` | episodic log |
| `forget` | `{ids}` or `{text}` delete memories and the facts they asserted |
| `explain` | `{target_id, query}` debug a missing recall instead of guessing |

A `forbidden` error means the token is read-only for that tool; do not retry it.
