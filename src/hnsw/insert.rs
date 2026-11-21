//! Node insertion (Malkov & Yashunin, Alg. 1) with the neighbor-selection
//! heuristic (Alg. 4, keepPrunedConnections) for both new links and the
//! shrinking of overflowing neighbor lists.

use super::cache::{Cached, WriteCache};
use super::search::{descend, search_layer};
use super::{EntryPoint, Graph, HnswParams, Scored, View, level_for};
use crate::archive::f32s_to_le;
use crate::error::{Error, Result};
use crate::hash::id_to_hex;
use crate::model::{HnswNode, Link};
use crate::tables::{keys, meta_get};
use redb::{ReadableTable, Table};

type BytesTable<'t> = Table<'t, u64, &'static [u8]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InsertOutcome {
    pub level: u8,
    /// The node already existed and its vector changed: links were rebuilt.
    pub relinked: bool,
    /// The node already existed with an identical vector: nothing to do.
    pub unchanged: bool,
    /// Distance evaluations performed.
    pub visited: u64,
}

/// Heuristic neighbor selection: keep a candidate only if it is closer to
/// the base than to every already-selected neighbor; then top up with the
/// best pruned candidates. `cands` must be sorted by decreasing similarity.
pub fn select_neighbors<G: Graph>(g: &G, cands: &[Scored], m: usize) -> Result<Vec<Link>> {
    let mut chosen: Vec<(Scored, std::rc::Rc<[f32]>)> = Vec::with_capacity(m);
    let mut pruned = Vec::new();
    for c in cands {
        if chosen.len() >= m {
            break;
        }
        let v = g.owned_vector(c.id)?;
        if chosen.iter().all(|(_, s)| crate::math::dot(&v, s) < c.sim) {
            chosen.push((*c, v));
        } else {
            pruned.push(*c);
        }
    }
    let mut out: Vec<Link> = chosen.iter().map(|(s, _)| Link { id: s.id, sim: s.sim }).collect();
    for p in pruned {
        if out.len() >= m {
            break;
        }
        out.push(Link { id: p.id, sim: p.sim });
    }
    Ok(out)
}

fn add_backlink<V, N>(g: &Cached<'_, '_, V, N>, target: u64, level: usize, link: Link, max_links: usize) -> Result<()>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    let Some(mut node) = g.node(target)? else {
        return Err(Error::Corrupt {
            table: "hnsw_nodes",
            key: id_to_hex(target),
            reason: "neighbor vanished during insert".into(),
        });
    };
    let Some(layer) = node.layers.get_mut(level) else {
        return Ok(());
    };
    match layer.iter_mut().find(|l| l.id == link.id) {
        Some(existing) => existing.sim = link.sim,
        None => layer.push(link),
    }
    if layer.len() > max_links {
        let mut cands: Vec<Scored> = layer.iter().map(|l| Scored { sim: l.sim, id: l.id }).collect();
        cands.sort_by(|a, b| b.cmp(a));
        *layer = select_neighbors(g, &cands, max_links)?;
    }
    g.cache.put_node(target, node);
    Ok(())
}

/// Insert (or upsert) unit vector `unit` under `id`, writing nodes directly.
pub fn insert(
    vectors: &mut BytesTable<'_>,
    nodes: &mut BytesTable<'_>,
    meta: &mut Table<'_, &'static str, u64>,
    p: &HnswParams,
    id: u64,
    unit: &[f32],
) -> Result<InsertOutcome> {
    let cache = WriteCache::default();
    let out = insert_cached(vectors, nodes, meta, p, id, unit, &cache)?;
    cache.flush(nodes)?;
    Ok(out)
}

/// Insert through a transaction-wide [`WriteCache`]; modified nodes stay in
/// the cache until the caller flushes it (before commit).
pub fn insert_cached(
    vectors: &mut BytesTable<'_>,
    nodes: &mut BytesTable<'_>,
    meta: &mut Table<'_, &'static str, u64>,
    p: &HnswParams,
    id: u64,
    unit: &[f32],
    cache: &WriteCache,
) -> Result<InsertOutcome> {
    let bytes = f32s_to_le(unit);
    let level = level_for(id, p.m, p.level_cap);
    let existed = Cached { cache, view: View::new(&*vectors, &*nodes) }.node(id)?.is_some();
    if existed && vectors.get(id)?.is_some_and(|g| g.value() == bytes.as_slice()) {
        return Ok(InsertOutcome { level, unchanged: true, ..Default::default() });
    }
    vectors.insert(id, bytes.as_slice())?;
    cache.put_vector(id, unit);
    let mut out = InsertOutcome { level, relinked: existed, ..Default::default() };
    let Some(ep) = EntryPoint::load(&*meta)? else {
        cache.put_node(id, HnswNode::empty(level));
        meta.insert(keys::HNSW_ENTRY, id)?;
        meta.insert(keys::HNSW_MAX_LEVEL, u64::from(level))?;
        meta.insert(keys::HNSW_COUNT, 1)?;
        return Ok(out);
    };
    let lvl = usize::from(level);
    let top = lvl.min(usize::from(ep.max_level));
    let mut layers: Vec<Vec<Link>> = vec![Vec::new(); lvl + 1];
    let g = Cached { cache, view: View::new(&*vectors, &*nodes) };
    let mut eps = descend(&g, unit, ep, top, Some(id), &mut out.visited)?;
    for l in (0..=top).rev() {
        let w = search_layer(&g, unit, &eps, p.ef_construction, l, Some(id), &mut out.visited)?;
        layers[l] = select_neighbors(&g, &w, p.max_links(l))?;
        if !w.is_empty() {
            eps = w;
        }
    }
    cache.put_node(id, HnswNode { level, layers: layers.clone() });
    for (l, links) in layers.iter().enumerate() {
        for link in links {
            add_backlink(&g, link.id, l, Link { id, sim: link.sim }, p.max_links(l))?;
        }
    }
    if level > ep.max_level {
        meta.insert(keys::HNSW_ENTRY, id)?;
        meta.insert(keys::HNSW_MAX_LEVEL, u64::from(level))?;
    }
    if !existed {
        let count = meta_get(&*meta, keys::HNSW_COUNT)?.unwrap_or(0);
        meta.insert(keys::HNSW_COUNT, count + 1)?;
    }
    Ok(out)
}
