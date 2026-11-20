//! Read side of the knowledge graph: BFS traversal and pattern matching.

use crate::archive;
use crate::error::Result;
use crate::hash::{IdMap, IdSet};
use crate::idlist;
use crate::model::{Term, Triple};
use crate::tables::{ENTITY_CHUNKS, OPS, POS, Pair, SPO, TERMS, TRIPLES, Triplet, prefix};
use redb::{ReadOnlyTable, ReadTransaction, ReadableTable};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    Out,
    In,
    #[default]
    Both,
}

impl Direction {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "out" | "outgoing" => Some(Direction::Out),
            "in" | "incoming" => Some(Direction::In),
            "both" | "any" => Some(Direction::Both),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BfsParams {
    pub depth: usize,
    pub direction: Direction,
    pub predicate: Option<u64>,
    pub max_edges: usize,
}

impl Default for BfsParams {
    fn default() -> Self {
        BfsParams { depth: 2, direction: Direction::Both, predicate: None, max_edges: 500 }
    }
}

/// A fact reached at `depth` hops from the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub s: u64,
    pub p: u64,
    pub o: u64,
    pub depth: u8,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BfsResult {
    pub edges: Vec<Edge>,
    /// Reached vertices with their hop distance (root has 0).
    pub nodes: Vec<(u64, u8)>,
    pub truncated: bool,
}

/// Read-only handles to every graph table.
pub struct GraphReader {
    pub terms: ReadOnlyTable<u64, &'static [u8]>,
    pub spo: ReadOnlyTable<Pair, &'static [u8]>,
    pub pos: ReadOnlyTable<Pair, &'static [u8]>,
    pub ops: ReadOnlyTable<Pair, &'static [u8]>,
    pub triples: ReadOnlyTable<Triplet, &'static [u8]>,
    pub entity_chunks: ReadOnlyTable<u64, &'static [u8]>,
}

impl GraphReader {
    pub fn open(txn: &ReadTransaction) -> Result<Self> {
        Ok(GraphReader {
            terms: txn.open_table(TERMS)?,
            spo: txn.open_table(SPO)?,
            pos: txn.open_table(POS)?,
            ops: txn.open_table(OPS)?,
            triples: txn.open_table(TRIPLES)?,
            entity_chunks: txn.open_table(ENTITY_CHUNKS)?,
        })
    }

    pub fn term(&self, id: u64) -> Result<Option<Term>> {
        match self.terms.get(id)? {
            Some(g) => Ok(Some(archive::decode(g.value())?)),
            None => Ok(None),
        }
    }

    /// Display name of a term, read zero-copy (falls back to the hex id).
    pub fn name(&self, id: u64, cache: &mut IdMap<String>) -> Result<String> {
        if let Some(n) = cache.get(&id) {
            return Ok(n.clone());
        }
        let n = match self.terms.get(id)? {
            Some(g) => archive::view::<Term>(g.value())?.name.as_str().to_string(),
            None => crate::hash::id_to_hex(id),
        };
        cache.insert(id, n.clone());
        Ok(n)
    }

    pub fn triple(&self, s: u64, p: u64, o: u64) -> Result<Option<Triple>> {
        match self.triples.get((s, p, o))? {
            Some(g) => Ok(Some(archive::decode(g.value())?)),
            None => Ok(None),
        }
    }

    /// Chunk ids mentioning `entity`.
    pub fn chunks_of(&self, entity: u64) -> Result<Vec<u64>> {
        Ok(match self.entity_chunks.get(entity)? {
            Some(g) => idlist::iter(g.value()).collect(),
            None => Vec::new(),
        })
    }

    /// Visit `(predicate, other_end)` for edges leaving (`out`) or entering `node`.
    fn edges_of(&self, node: u64, out: bool, f: &mut dyn FnMut(u64, u64)) -> Result<()> {
        let table = if out { &self.spo } else { &self.ops };
        for item in table.range(prefix(node))? {
            let (k, v) = item?;
            let p = k.value().1;
            for other in idlist::iter(v.value()) {
                f(p, other);
            }
        }
        Ok(())
    }

    /// Breadth-first traversal from `root` up to `params.depth` hops.
    pub fn bfs(&self, root: u64, params: &BfsParams) -> Result<BfsResult> {
        let mut res = BfsResult::default();
        let mut depth_of: IdMap<u8> = IdMap::default();
        depth_of.insert(root, 0);
        res.nodes.push((root, 0));
        let mut seen: HashSet<(u64, u64, u64)> = HashSet::new();
        let mut frontier = vec![root];
        let dirs: &[bool] = match params.direction {
            Direction::Out => &[true],
            Direction::In => &[false],
            Direction::Both => &[true, false],
        };
        'levels: for d in 1..=params.depth.min(u8::MAX as usize) as u8 {
            let mut next = Vec::new();
            for &node in &frontier {
                for &out in dirs {
                    let mut found = Vec::new();
                    self.edges_of(node, out, &mut |p, other| found.push((p, other)))?;
                    for (p, other) in found {
                        if params.predicate.is_some_and(|want| want != p) {
                            continue;
                        }
                        let (s, o) = if out { (node, other) } else { (other, node) };
                        if !seen.insert((s, p, o)) {
                            continue;
                        }
                        if res.edges.len() >= params.max_edges {
                            res.truncated = true;
                            break 'levels;
                        }
                        res.edges.push(Edge { s, p, o, depth: d });
                        if let std::collections::hash_map::Entry::Vacant(e) = depth_of.entry(other) {
                            e.insert(d);
                            res.nodes.push((other, d));
                            next.push(other);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(res)
    }

    /// All facts matching a pattern where `None` is a wildcard.
    pub fn match_pattern(
        &self,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
        limit: usize,
    ) -> Result<Vec<(u64, u64, u64)>> {
        let mut out = Vec::new();
        let mut push = |t: (u64, u64, u64)| {
            if out.len() < limit && o.is_none_or(|x| x == t.2) && p.is_none_or(|x| x == t.1) {
                out.push(t);
            }
            out.len() < limit
        };
        match (s, p, o) {
            (Some(s), Some(p), _) => {
                if let Some(g) = self.spo.get((s, p))? {
                    idlist::iter(g.value()).all(|x| push((s, p, x)));
                }
            }
            (None, Some(p), Some(o)) => {
                if let Some(g) = self.pos.get((p, o))? {
                    idlist::iter(g.value()).all(|x| push((x, p, o)));
                }
            }
            (Some(s), None, _) => {
                for item in self.spo.range(prefix(s))? {
                    let (k, v) = item?;
                    let p = k.value().1;
                    if !idlist::iter(v.value()).all(|x| push((s, p, x))) {
                        break;
                    }
                }
            }
            (None, _, Some(o)) => {
                for item in self.ops.range(prefix(o))? {
                    let (k, v) = item?;
                    let p = k.value().1;
                    if !idlist::iter(v.value()).all(|x| push((x, p, o))) {
                        break;
                    }
                }
            }
            (None, Some(p), None) => {
                for item in self.pos.range(prefix(p))? {
                    let (k, v) = item?;
                    let o = k.value().1;
                    if !idlist::iter(v.value()).all(|x| push((x, p, o))) {
                        break;
                    }
                }
            }
            (None, None, None) => {
                for item in self.triples.range(..)? {
                    if !push(item?.0.value()) {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Ids of every entity reachable within `depth` hops (both directions).
pub fn neighborhood(g: &GraphReader, root: u64, depth: usize) -> Result<IdSet> {
    let params = BfsParams { depth, max_edges: usize::MAX, ..Default::default() };
    Ok(g.bfs(root, &params)?.nodes.into_iter().map(|(id, _)| id).collect())
}
