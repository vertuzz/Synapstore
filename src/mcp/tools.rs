//! MCP tool catalogue (`tools/list`) with JSON Schemas for every argument.

use serde_json::{Value, json};

fn vector_schema(what: &str) -> Value {
    json!({"type": "array", "items": {"type": "number"},
        "description": format!("{what}. Optional: when omitted the store embeds the text with its own embedder. \
            A store is pinned to one embedding space (dimension + source) by its first write.")})
}

fn fact_schema() -> Value {
    json!({"type": "array", "description": "Facts as {subject, predicate, object} objects or [s, p, o] triples.",
    "items": {"oneOf": [
        {"type": "object", "required": ["subject", "predicate", "object"],
         "properties": {"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}}},
        {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}
    ]}})
}

fn duration_schema(desc: &str) -> Value {
    json!({"type": ["number", "string"], "description": format!("{desc} Seconds, or a string like \"30m\", \"12h\", \"7d\".")})
}

fn tool(name: &str, desc: &str, props: Value, required: &[&str], read_only: bool) -> Value {
    json!({
        "name": name,
        "description": desc,
        "inputSchema": {"type": "object", "properties": props, "required": required},
        "annotations": {"readOnlyHint": read_only, "idempotentHint": true, "openWorldHint": false}
    })
}

/// The full tool list.
pub fn definitions() -> Value {
    json!([
        tool(
            "memorize",
            "Store a memory. The text is chunked (when no vector is given), embedded, indexed in HNSW, \
             its entities are linked into the knowledge graph, facts are asserted, and an episode is appended. \
             Idempotent: the same text (case/whitespace-insensitive) always maps to the same id and is upserted. \
             The store never derives facts from text: pass `facts` for durable relations (people and roles, \
             family, places, ownership, projects and their stack, key dates). To correct a memory, pass its id \
             in `supersedes` instead of only adding a new one.",
            json!({
                "text": {"type": "string", "description": "Memory content."},
                "vector": vector_schema("Embedding of `text`"),
                "entities": {"type": "array", "items": {"type": "string"},
                    "description": "Entity names mentioned. Omit to auto-extract capitalized names."},
                "facts": fact_schema(),
                "supersedes": {"type": "array", "items": {"type": "string"},
                    "description": "Ids of memories this one corrects or replaces. They are forgotten, with the \
                        facts they asserted, in the same transaction."},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string", "description": "Provenance (file, URL, tool…)."},
                "kind": {"type": "string", "description": "Episode kind for the episodic log (default \"memory\")."},
                "timestamp_ms": {"type": "integer", "description": "Unix ms of the memory (default now)."}
            }),
            &["text"],
            false
        ),
        tool(
            "recall",
            "Semantic search over memories (HNSW) re-ranked by time-decay: fresh relevant memories are boosted, \
             old ones penalized unless they are a near-perfect match.",
            json!({
                "query": {"type": "string", "description": "Query text (built-in embedder stores)."},
                "vector": vector_schema("Query embedding"),
                "top_k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 5},
                "max_age": duration_schema("Ignore memories not updated within this window."),
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Keep memories with any of these tags."},
                "half_life_days": {"type": "number", "description": "Override time-decay half-life."},
                "decay": {"type": "boolean", "default": true, "description": "Set false to rank by raw similarity."},
                "ef": {"type": "integer", "description": "HNSW beam width (accuracy/speed trade-off)."},
                "now_ms": {"type": "integer", "description": "Evaluate decay as of this unix ms (simulation)."}
            }),
            &[],
            true
        ),
        tool(
            "graph_query",
            "Breadth-first traversal of the knowledge graph from an entity: all facts within `depth` hops, \
             reached entities, and memories mentioning the entity.",
            json!({
                "subject": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 6, "default": 2},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "default": "both"},
                "predicate": {"type": "string", "description": "Only follow this predicate."},
                "limit": {"type": "integer", "default": 200, "description": "Max facts returned."},
                "memories": {"type": "integer", "default": 5, "description": "Max linked memories returned."}
            }),
            &["subject"],
            true
        ),
        tool(
            "graph_match",
            "Exact fact lookup by pattern; omitted positions are wildcards (e.g. predicate+object = reverse lookup).",
            json!({
                "subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"},
                "limit": {"type": "integer", "default": 100}
            }),
            &[],
            true
        ),
        tool(
            "add_facts",
            "Assert facts (subject-predicate-object) directly into the knowledge graph (upsert).",
            json!({"facts": fact_schema(),
                "source_id": {"type": "string", "description": "Chunk id the facts come from."},
                "timestamp_ms": {"type": "integer"}}),
            &["facts"],
            false
        ),
        tool(
            "retract_fact",
            "Remove one fact from the knowledge graph.",
            json!({"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}}),
            &["subject", "predicate", "object"],
            false
        ),
        tool(
            "log_event",
            "Append an event (thought, action, tool result…) to the episodic log.",
            json!({"kind": {"type": "string"}, "text": {"type": "string"},
                "timestamp_ms": {"type": "integer"}, "chunk_id": {"type": "string"}}),
            &["text"],
            false
        ),
        tool(
            "recent_events",
            "Read the episodic log, newest first.",
            json!({"limit": {"type": "integer", "default": 20}, "kind": {"type": "string"},
                "since_ms": {"type": "integer"}, "until_ms": {"type": "integer"}}),
            &[],
            true
        ),
        tool(
            "forget",
            "Delete memories by id or by exact text. Facts that a deleted memory asserted are retracted too; \
             facts added with `add_facts` and no `source_id` are kept.",
            json!({"ids": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"}}),
            &[],
            false
        ),
        tool(
            "explain",
            "Whitebox trace: why a memory is (not) returned for a query — exact rank, HNSW reachability, \
             time-decay and filters.",
            json!({"target_id": {"type": "string"}, "target_text": {"type": "string"},
                "query": {"type": "string"}, "vector": vector_schema("Query embedding"),
                "top_k": {"type": "integer", "default": 5}, "ef": {"type": "integer"},
                "max_age": duration_schema("Same as recall."), "now_ms": {"type": "integer"}}),
            &[],
            true
        ),
        tool(
            "repair_index",
            "Maintenance: reconnect HNSW nodes that pruning left unreachable (normally automatic and amortized).",
            json!({}),
            &[],
            false
        ),
        tool(
            "stats",
            "Store size, index structure, configuration and performance metrics.",
            json!({"detailed": {"type": "boolean", "default": false,
                "description": "Scan the HNSW graph for level/degree/connectivity statistics."}}),
            &[],
            true
        ),
    ])
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_tool_has_a_schema() {
        let defs = super::definitions();
        let tools = defs.as_array().unwrap();
        assert_eq!(tools.len(), 12);
        for t in tools {
            assert!(t["name"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }
}
