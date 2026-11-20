//! Bespoke HNSW (Hierarchical Navigable Small World) index stored *inside*
//! redb: every graph node is a record of the `hnsw_nodes` table holding its
//! neighbor ids per layer, and `get_node()` is a lookup in the current redb
//! transaction. Vectors are unit-normalized, so cosine similarity = dot.

pub mod insert;
pub mod inspect;
pub mod repair;
pub mod search;
pub mod view;

pub use insert::{InsertOutcome, insert};
pub use search::{knn, search_layer};
pub use view::{EntryPoint, View};

use crate::hash::mix64;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// Max neighbors per node on layers ≥ 1.
    pub m: usize,
    /// Max neighbors per node on layer 0 (usually `2·m`).
    pub m0: usize,
    /// Beam width while building.
    pub ef_construction: usize,
    /// Default beam width while searching.
    pub ef_search: usize,
    /// Hard cap on node levels.
    pub level_cap: u8,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams { m: 16, m0: 32, ef_construction: 100, ef_search: 64, level_cap: 16 }
    }
}

impl HnswParams {
    pub fn max_links(&self, level: usize) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }
}

const LEVEL_SALT: u64 = 0x4c45_5645_4c53_414c;

/// Deterministic level for node `id`: `floor(-ln(u) / ln(m))` with `u`
/// derived from the id, so re-inserting content always yields the same
/// graph shape (idempotency, PRD §6.3).
pub fn level_for(id: u64, m: usize, cap: u8) -> u8 {
    let u = ((mix64(id ^ LEVEL_SALT) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let ml = 1.0 / (m.max(2) as f64).ln();
    let level = (-u.ln() * ml).floor();
    if level >= f64::from(cap) { cap } else { level as u8 }
}

/// A candidate with its similarity to the query (higher is closer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub sim: f32,
    pub id: u64,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.total_cmp(&other.sim).then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_distribution_is_geometric() {
        let n = 100_000u64;
        let mut hist = [0u64; 17];
        for id in 0..n {
            hist[usize::from(level_for(mix64(id), 16, 16))] += 1;
        }
        // P(level >= 1) = 1/16.
        let frac = 1.0 - hist[0] as f64 / n as f64;
        assert!((frac - 1.0 / 16.0).abs() < 0.005, "{frac}");
        assert_eq!(level_for(5, 16, 16), level_for(5, 16, 16));
        assert!(level_for(0, 2, 3) <= 3);
    }

    #[test]
    fn scored_orders_by_similarity() {
        let mut v = [Scored { sim: 0.1, id: 1 }, Scored { sim: 0.9, id: 2 }, Scored { sim: 0.5, id: 3 }];
        v.sort_by(|a, b| b.cmp(a));
        assert_eq!(v.iter().map(|s| s.id).collect::<Vec<_>>(), vec![2, 3, 1]);
    }
}
