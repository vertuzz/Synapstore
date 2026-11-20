//! Layer search (beam search with a dynamic candidate list) and k-NN.

use super::{EntryPoint, Scored, View};
use crate::error::Result;
use crate::hash::IdSet;
use redb::ReadableTable;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Beam search on one layer starting from `entry`. Returns up to `ef`
/// results sorted by decreasing similarity. `exclude` may be traversed but
/// is never returned (used when re-linking an existing node). `visited`
/// accumulates the number of distance evaluations.
pub fn search_layer<V, N>(
    view: &View<'_, V, N>,
    q: &[f32],
    entry: &[Scored],
    ef: usize,
    level: usize,
    exclude: Option<u64>,
    visited: &mut u64,
) -> Result<Vec<Scored>>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    let ef = ef.max(1);
    let mut seen = IdSet::with_capacity_and_hasher(ef * 8, Default::default());
    let mut cand: BinaryHeap<Scored> = BinaryHeap::with_capacity(ef * 2);
    let mut res: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(ef + 1);
    for e in entry {
        if seen.insert(e.id) {
            cand.push(*e);
            if Some(e.id) != exclude {
                res.push(Reverse(*e));
            }
        }
    }
    while res.len() > ef {
        res.pop();
    }
    let mut buf = Vec::with_capacity(64);
    while let Some(c) = cand.pop() {
        if res.len() >= ef && res.peek().is_some_and(|w| c.sim < w.0.sim) {
            break;
        }
        view.neighbors(c.id, level, &mut buf)?;
        for &n in &buf {
            if !seen.insert(n) {
                continue;
            }
            let sim = view.sim(q, n)?;
            *visited += 1;
            if res.len() < ef || res.peek().is_some_and(|w| sim > w.0.sim) {
                let s = Scored { sim, id: n };
                cand.push(s);
                if Some(n) != exclude {
                    res.push(Reverse(s));
                    if res.len() > ef {
                        res.pop();
                    }
                }
            }
        }
    }
    let mut out: Vec<Scored> = res.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| b.cmp(a));
    Ok(out)
}

/// Greedy descent from the top layer down to (but excluding) `stop_level`.
pub fn descend<V, N>(
    view: &View<'_, V, N>,
    q: &[f32],
    ep: EntryPoint,
    stop_level: usize,
    exclude: Option<u64>,
    visited: &mut u64,
) -> Result<Vec<Scored>>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    let mut eps = vec![Scored { sim: view.sim(q, ep.id)?, id: ep.id }];
    *visited += 1;
    let mut level = usize::from(ep.max_level);
    while level > stop_level {
        let w = search_layer(view, q, &eps, 1, level, exclude, visited)?;
        if !w.is_empty() {
            eps = w;
        }
        level -= 1;
    }
    Ok(eps)
}

/// Approximate nearest neighbors of unit vector `q`: returns up to
/// `max(ef, k)` candidates sorted by decreasing similarity.
pub fn knn<V, N>(
    view: &View<'_, V, N>,
    ep: EntryPoint,
    q: &[f32],
    k: usize,
    ef: usize,
    visited: &mut u64,
) -> Result<Vec<Scored>>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    let eps = descend(view, q, ep, 0, None, visited)?;
    search_layer(view, q, &eps, ef.max(k), 0, None, visited)
}
