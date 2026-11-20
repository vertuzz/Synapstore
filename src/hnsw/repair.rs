//! Connectivity repair. Heuristic neighbor pruning can strip the last
//! incoming link of a node — notably among *identical* vectors, where the
//! diversity test ties — leaving it unreachable from the entry point. Like
//! Lucene's `connectComponents`, this pass finds every node not reachable on
//! layer 0 and links it from its nearest reachable node.

use super::search::knn;
use super::{EntryPoint, HnswParams, View};
use crate::archive::{self, le_to_f32s};
use crate::error::{Error, Result};
use crate::hash::{IdSet, id_to_hex};
use crate::model::{HnswNode, Link};
use crate::tables::{keys, meta_get};
use redb::{ReadableTable, Table};
use serde::Serialize;

type BytesTable<'t> = Table<'t, u64, &'static [u8]>;

/// Minimum number of inserts between two automatic repairs.
pub const AUTO_REPAIR_MIN: u64 = 1000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RepairOutcome {
    pub nodes: u64,
    pub unreachable_before: u64,
    pub links_added: u64,
}

/// Extend `reach` with everything reachable on layer 0 from `start`.
pub fn extend_reach<N: ReadableTable<u64, &'static [u8]>>(nodes: &N, start: u64, reach: &mut IdSet) -> Result<()> {
    let mut stack = vec![start];
    reach.insert(start);
    while let Some(id) = stack.pop() {
        if let Some(g) = nodes.get(id)? {
            let node = archive::view::<HnswNode>(g.value())?;
            for l in node.layers.first().into_iter().flat_map(|l| l.iter()) {
                let n = l.id.to_native();
                if reach.insert(n) {
                    stack.push(n);
                }
            }
        }
    }
    Ok(())
}

fn push_link(nodes: &mut BytesTable<'_>, from: u64, link: Link) -> Result<()> {
    let mut node: HnswNode = match nodes.get(from)? {
        Some(g) => archive::decode(g.value())?,
        None => {
            return Err(Error::Corrupt {
                table: "hnsw_nodes",
                key: id_to_hex(from),
                reason: "missing during repair".into(),
            });
        }
    };
    if node.layers[0].iter().all(|l| l.id != link.id) {
        node.layers[0].push(link);
        nodes.insert(from, archive::encode(&node)?.as_slice())?;
    }
    Ok(())
}

/// Reconnect every node unreachable from the entry point on layer 0.
pub fn repair(
    vectors: &BytesTable<'_>,
    nodes: &mut BytesTable<'_>,
    ep: Option<EntryPoint>,
    p: &HnswParams,
) -> Result<RepairOutcome> {
    let mut out = RepairOutcome::default();
    let Some(ep) = ep else { return Ok(out) };
    let mut reach = IdSet::default();
    extend_reach(&*nodes, ep.id, &mut reach)?;
    let mut orphans = Vec::new();
    for item in nodes.range(..)? {
        let id = item?.0.value();
        out.nodes += 1;
        if !reach.contains(&id) {
            orphans.push(id);
        }
    }
    out.unreachable_before = orphans.len() as u64;
    for u in orphans {
        if reach.contains(&u) {
            continue; // reconnected through an earlier orphan
        }
        let q = match vectors.get(u)? {
            Some(g) => le_to_f32s(g.value()),
            None => continue,
        };
        let mut visited = 0;
        let cands = knn(&View::new(vectors, &*nodes), ep, &q, 8, p.ef_construction, &mut visited)?;
        let Some(c) = cands.into_iter().find(|c| c.id != u && reach.contains(&c.id)) else {
            continue;
        };
        // Two-way link: the orphan becomes reachable and points back into
        // the main component. Lists may exceed `m0` by one until the next
        // regular shrink.
        push_link(nodes, c.id, Link { id: u, sim: c.sim })?;
        push_link(nodes, u, Link { id: c.id, sim: c.sim })?;
        out.links_added += 1;
        extend_reach(&*nodes, u, &mut reach)?;
    }
    Ok(out)
}

/// Count inserts and run [`repair`] once enough have accumulated
/// (`max(AUTO_REPAIR_MIN, nodes / 8)`), keeping the amortized cost O(1).
pub fn maybe_repair(
    vectors: &BytesTable<'_>,
    nodes: &mut BytesTable<'_>,
    meta: &mut Table<'_, &'static str, u64>,
    p: &HnswParams,
    inserted: u64,
) -> Result<Option<RepairOutcome>> {
    if inserted == 0 {
        return Ok(None);
    }
    let since = meta_get(&*meta, keys::HNSW_SINCE_REPAIR)?.unwrap_or(0) + inserted;
    let count = meta_get(&*meta, keys::HNSW_COUNT)?.unwrap_or(0);
    if since < AUTO_REPAIR_MIN.max(count / 8) {
        meta.insert(keys::HNSW_SINCE_REPAIR, since)?;
        return Ok(None);
    }
    meta.insert(keys::HNSW_SINCE_REPAIR, 0)?;
    let ep = EntryPoint::load(&*meta)?;
    repair(vectors, nodes, ep, p).map(Some)
}
