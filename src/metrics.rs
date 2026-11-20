//! Lock-free telemetry (PRD §9): per-operation latency histograms and
//! counters. A `Metrics` value is owned by the store and passed explicitly.

use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

const BUCKETS: usize = 40;

/// Log2-bucketed latency histogram (bucket `i` holds `[2^i, 2^(i+1))` µs).
pub struct LatencyHist {
    count: AtomicU64,
    sum_us: AtomicU64,
    max_us: AtomicU64,
    buckets: [AtomicU64; BUCKETS],
}

/// Index of the bucket for `us` microseconds.
pub fn bucket_of(us: u64) -> usize {
    (63 - (us | 1).leading_zeros() as usize).min(BUCKETS - 1)
}

/// Estimate the `q`-quantile from bucket counts (upper bucket bound).
pub fn quantile(buckets: &[u64], q: f64) -> u64 {
    let total: u64 = buckets.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = ((total as f64) * q).ceil().max(1.0) as u64;
    let mut seen = 0;
    for (i, c) in buckets.iter().enumerate() {
        seen += c;
        if seen >= target {
            return (1u64 << (i + 1)) - 1;
        }
    }
    u64::MAX
}

impl Default for LatencyHist {
    fn default() -> Self {
        LatencyHist {
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl LatencyHist {
    pub fn record(&self, d: Duration) {
        let us = d.as_micros().min(u128::from(u64::MAX)) as u64;
        self.count.fetch_add(1, Relaxed);
        self.sum_us.fetch_add(us, Relaxed);
        self.max_us.fetch_max(us, Relaxed);
        self.buckets[bucket_of(us)].fetch_add(1, Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Relaxed)
    }

    pub fn snapshot(&self) -> Value {
        let count = self.count();
        let b: Vec<u64> = self.buckets.iter().map(|x| x.load(Relaxed)).collect();
        let mean = self.sum_us.load(Relaxed).checked_div(count).unwrap_or(0);
        json!({
            "count": count,
            "mean_us": mean,
            "max_us": self.max_us.load(Relaxed),
            "p50_us_le": quantile(&b, 0.50),
            "p95_us_le": quantile(&b, 0.95),
            "p99_us_le": quantile(&b, 0.99),
        })
    }
}

macro_rules! ops {
    ($($name:ident => $label:literal),* $(,)?) => {
        /// Instrumented operations.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Op { $($name),* }
        impl Op {
            pub const ALL: &'static [Op] = &[$(Op::$name),*];
            pub fn label(self) -> &'static str {
                match self { $(Op::$name => $label),* }
            }
        }
    };
}

ops! {
    Memorize => "memorize",
    Recall => "recall",
    GraphQuery => "graph_query",
    GraphMatch => "graph_match",
    AddFacts => "add_facts",
    LogEvent => "log_event",
    RecentEvents => "recent_events",
    Forget => "forget",
    Explain => "explain",
    WriteTxn => "write_txn",
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        /// Store-wide counters (process lifetime).
        #[derive(Default)]
        pub struct Counters { $(pub $name: AtomicU64),* }
        impl Counters {
            pub fn snapshot(&self) -> Value {
                json!({ $(stringify!($name): self.$name.load(Relaxed)),* })
            }
        }
    };
}

counters! {
    errors,
    write_retries,
    busy_failures,
    group_commits,
    grouped_ops,
    hnsw_searches,
    hnsw_nodes_visited,
    hnsw_inserts,
    hnsw_relinks,
    hnsw_repairs,
    hnsw_repair_links,
    zero_copy_reads,
    hash_collisions,
    chunks_created,
    chunks_updated,
    triples_created,
    events_logged,
}

#[derive(Default)]
pub struct Metrics {
    hists: [LatencyHist; 10],
    pub counters: Counters,
}

impl Metrics {
    pub fn hist(&self, op: Op) -> &LatencyHist {
        &self.hists[op as usize]
    }

    pub fn record(&self, op: Op, d: Duration) {
        self.hist(op).record(d);
    }

    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Relaxed);
    }

    pub fn snapshot(&self) -> Value {
        let mut lat = serde_json::Map::new();
        for op in Op::ALL {
            if self.hist(*op).count() > 0 {
                lat.insert(op.label().into(), self.hist(*op).snapshot());
            }
        }
        json!({
            "simd_backend": crate::math::backend(),
            "latency": lat,
            "counters": self.counters.snapshot(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_quantiles() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(1), 0);
        assert_eq!(bucket_of(2), 1);
        assert_eq!(bucket_of(1023), 9);
        assert_eq!(bucket_of(1024), 10);
        let mut b = vec![0u64; BUCKETS];
        b[3] = 90;
        b[10] = 10;
        assert_eq!(quantile(&b, 0.5), 15);
        assert_eq!(quantile(&b, 0.95), 2047);
        assert_eq!(quantile(&[0; 4], 0.5), 0);
    }

    #[test]
    fn op_table_fits_histograms() {
        assert_eq!(Op::ALL.len(), 10);
        let m = Metrics::default();
        m.record(Op::Recall, Duration::from_micros(100));
        let s = m.snapshot();
        assert_eq!(s["latency"]["recall"]["count"], 1);
    }
}
