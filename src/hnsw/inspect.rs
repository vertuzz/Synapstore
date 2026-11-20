//! Whitebox tooling for the index (PRD §9): exact brute-force search for
//! ground truth, structural statistics and connectivity checks.

use super::{EntryPoint, Scored};
use crate::archive;
use crate::error::Result;
use crate::hash::IdSet;
use crate::math;
use crate::model::HnswNode;
use redb::ReadableTable;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Exact top-`k` by scanning every stored vector.
pub fn brute_force<V>(vectors: &V, q: &[f32], k: usize) -> Result<Vec<Scored>>
where
    V: ReadableTable<u64, &'static [u8]>,
{
    let mut heap: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(k + 1);
    for item in vectors.range(..)? {
        let (key, val) = item?;
        let s = Scored { sim: math::dot_le(q, val.value()), id: key.value() };
        if heap.len() < k {
            heap.push(Reverse(s));
        } else if heap.peek().is_some_and(|w| s > w.0) {
            heap.pop();
            heap.push(Reverse(s));
        }
    }
    let mut out: Vec<Scored> = heap.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| b.cmp(a));
    Ok(out)
}

/// Structural summary of the graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexStats {
    pub nodes: u64,
    pub max_level: u8,
    /// Number of nodes whose top level is `i`.
    pub level_histogram: Vec<u64>,
    /// Mean out-degree per layer.
    pub mean_degree: Vec<f64>,
    /// Nodes with no layer-0 links (only legal for a single-node index).
    pub isolated: u64,
}

pub fn index_stats<N>(nodes: &N) -> Result<IndexStats>
where
    N: ReadableTable<u64, &'static [u8]>,
{
    let mut st = IndexStats::default();
    let mut degree_sum: Vec<u64> = Vec::new();
    let mut per_layer: Vec<u64> = Vec::new();
    for item in nodes.range(..)? {
        let (_, val) = item?;
        let node = archive::view::<HnswNode>(val.value())?;
        st.nodes += 1;
        let lvl = usize::from(node.level);
        st.max_level = st.max_level.max(node.level);
        if st.level_histogram.len() <= lvl {
            st.level_histogram.resize(lvl + 1, 0);
            degree_sum.resize(lvl + 1, 0);
            per_layer.resize(lvl + 1, 0);
        }
        st.level_histogram[lvl] += 1;
        for (l, layer) in node.layers.iter().enumerate() {
            degree_sum[l] += layer.len() as u64;
            per_layer[l] += 1;
        }
        if node.layers.first().is_none_or(|l| l.is_empty()) {
            st.isolated += 1;
        }
    }
    st.mean_degree =
        degree_sum.iter().zip(&per_layer).map(|(d, n)| if *n == 0 { 0.0 } else { *d as f64 / *n as f64 }).collect();
    Ok(st)
}

/// Number of nodes reachable on layer 0 from the entry point (BFS).
pub fn reachable_from_entry<N>(nodes: &N, ep: Option<EntryPoint>) -> Result<u64>
where
    N: ReadableTable<u64, &'static [u8]>,
{
    let Some(ep) = ep else { return Ok(0) };
    let mut seen = IdSet::default();
    let mut stack = vec![ep.id];
    seen.insert(ep.id);
    while let Some(id) = stack.pop() {
        if let Some(g) = nodes.get(id)? {
            let node = archive::view::<HnswNode>(g.value())?;
            if let Some(layer) = node.layers.first() {
                for l in layer.iter() {
                    let n = l.id.to_native();
                    if seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
    }
    Ok(seen.len() as u64)
}
