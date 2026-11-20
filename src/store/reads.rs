//! Graph and episodic read paths.

use super::SynapStore;
use super::types::{EntityView, EventQuery, EventView, FactView, GraphOutcome, GraphQuery, MemoryRef};
use crate::archive;
use crate::error::Result;
use crate::graph::{BfsParams, GraphReader};
use crate::hash::{IdMap, entity_id, predicate_id};
use crate::metrics::Op;
use crate::model::{Chunk, MemoryItem};
use crate::tables::{CHUNKS, EPISODES};
use crate::timefmt::rfc3339;
use redb::{ReadableDatabase, ReadableTable};

pub const MAX_DEPTH: usize = 6;

fn fact_view(g: &GraphReader, s: u64, p: u64, o: u64, depth: u8, names: &mut IdMap<String>) -> Result<FactView> {
    let meta = g.triple(s, p, o)?;
    Ok(FactView {
        subject: g.name(s, names)?,
        predicate: g.name(p, names)?,
        object: g.name(o, names)?,
        depth,
        updated_at: meta.as_ref().map_or(0, |t| t.updated_at),
        source_chunk: meta.map(|t| t.source_chunk).filter(|c| *c != 0),
    })
}

impl SynapStore {
    /// BFS over the knowledge graph from `subject` (PRD `graph_query`).
    pub fn graph_query(&self, q: &GraphQuery) -> Result<GraphOutcome> {
        self.timed(Op::GraphQuery, || {
            let txn = self.db().begin_read()?;
            let g = GraphReader::open(&txn)?;
            let root = entity_id(&q.subject);
            let Some(term) = g.term(root)? else {
                return Ok(GraphOutcome { subject: q.subject.clone(), ..Default::default() });
            };
            let params = BfsParams {
                depth: q.depth.clamp(1, MAX_DEPTH),
                direction: q.direction,
                predicate: q.predicate.as_deref().map(predicate_id),
                max_edges: q.limit.clamp(1, 100_000),
            };
            let res = g.bfs(root, &params)?;
            let mut names = IdMap::default();
            let facts = res
                .edges
                .iter()
                .map(|e| fact_view(&g, e.s, e.p, e.o, e.depth, &mut names))
                .collect::<Result<Vec<_>>>()?;
            let entities = res
                .nodes
                .iter()
                .skip(1)
                .map(|(id, d)| Ok(EntityView { name: g.name(*id, &mut names)?, depth: *d }))
                .collect::<Result<Vec<_>>>()?;
            let chunks = txn.open_table(CHUNKS)?;
            let mut memories = Vec::new();
            for id in g.chunks_of(root)? {
                if let Some(cg) = chunks.get(id)? {
                    let c = archive::view::<Chunk>(cg.value())?;
                    memories.push(MemoryRef { id, text: c.text.as_str().into(), updated_at: c.updated_at.to_native() });
                }
            }
            memories.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
            memories.truncate(q.memories);
            Ok(GraphOutcome { found: true, subject: term.name, facts, entities, memories, truncated: res.truncated })
        })
    }

    /// Pattern match on facts; `None` positions are wildcards.
    pub fn graph_match(
        &self,
        s: Option<&str>,
        p: Option<&str>,
        o: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FactView>> {
        self.timed(Op::GraphMatch, || {
            let txn = self.db().begin_read()?;
            let g = GraphReader::open(&txn)?;
            let found =
                g.match_pattern(s.map(entity_id), p.map(predicate_id), o.map(entity_id), limit.clamp(1, 100_000))?;
            let mut names = IdMap::default();
            found.iter().map(|(s, p, o)| fact_view(&g, *s, *p, *o, 0, &mut names)).collect()
        })
    }

    /// Most recent episodic events first.
    pub fn recent_events(&self, q: &EventQuery) -> Result<Vec<EventView>> {
        self.timed(Op::RecentEvents, || {
            let txn = self.db().begin_read()?;
            let t = txn.open_table(EPISODES)?;
            let lo = (q.since_ms.unwrap_or(0), 0);
            let hi = (q.until_ms.unwrap_or(u64::MAX), u64::MAX);
            let limit = q.limit.clamp(1, 10_000);
            let mut out = Vec::new();
            if lo > hi {
                return Ok(out);
            }
            for item in t.range(lo..=hi)?.rev() {
                let (_, v) = item?;
                let e = archive::view::<MemoryItem>(v.value())?;
                if q.kind.as_deref().is_some_and(|k| !k.eq_ignore_ascii_case(e.kind.as_str())) {
                    continue;
                }
                let ts = e.ts.to_native();
                let chunk = e.chunk_id.to_native();
                out.push(EventView {
                    id: e.id.to_native(),
                    ts,
                    time: rfc3339(ts),
                    kind: e.kind.as_str().into(),
                    text: e.text.as_str().into(),
                    chunk_id: (chunk != 0).then_some(chunk),
                });
                if out.len() >= limit {
                    break;
                }
            }
            Ok(out)
        })
    }

    /// Fetch one chunk record (owned).
    pub fn get_chunk(&self, id: u64) -> Result<Option<Chunk>> {
        let txn = self.db().begin_read()?;
        let t = txn.open_table(CHUNKS)?;
        let r = t.get(id)?.map(|g| archive::decode::<Chunk>(g.value())).transpose()?;
        Ok(r)
    }
}
