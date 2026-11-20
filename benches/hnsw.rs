//! Iteration 2 benchmark: SIMD kernels at 384/768/1536 dims and an HNSW
//! index of 10,000 generated vectors (build throughput, latency, recall@10).
//! Knobs: SYNAPSTORE_BENCH_VECTORS, SYNAPSTORE_BENCH_DIM, SYNAPSTORE_BENCH_QUERIES.

mod support;

use serde_json::json;
use std::hint::black_box;
use std::time::Instant;
use synapstore::math;
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::synth::check::percentiles;
use synapstore::synth::rng::Rng;
use synapstore::{Config, SynapStore};

fn kernel_ns(dim: usize, simd: bool) -> f64 {
    let mut rng = Rng::new(dim as u64);
    let a: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
    let iters = 2_000_000 / dim * 100;
    let start = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        acc += if simd {
            math::cosine(black_box(&a), black_box(&b))
        } else {
            // SAFETY: both slices are valid for `dim` reads.
            let (ab, aa, bb) = unsafe { math::scalar::dot3_raw(a.as_ptr(), b.as_ptr(), dim) };
            ab / (aa.sqrt() * bb.sqrt())
        };
    }
    black_box(acc);
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn main() {
    let n = support::env_usize("SYNAPSTORE_BENCH_VECTORS", 10_000);
    let dim = support::env_usize("SYNAPSTORE_BENCH_DIM", 384);
    let queries = support::env_usize("SYNAPSTORE_BENCH_QUERIES", 300);
    let mut kernels = serde_json::Map::new();
    for d in [384, 768, 1536] {
        let (simd, scalar) = (kernel_ns(d, true), kernel_ns(d, false));
        kernels.insert(d.to_string(), json!({"simd_ns": simd, "scalar_ns": scalar, "speedup": scalar / simd}));
    }

    let scratch = support::Scratch::new("hnsw");
    let store = SynapStore::open(scratch.0.join("h.redb"), Config::default()).unwrap();
    let mut rng = Rng::new(42);
    let centers: Vec<Vec<f32>> = (0..100).map(|_| rng.unit_vector(dim)).collect();
    let noise = synapstore::synth::world::noise_for_cos(0.75, dim);
    let vecs: Vec<Vec<f32>> = (0..n).map(|i| rng.perturb(&centers[i % 100], noise)).collect();
    let started = Instant::now();
    for (b, chunk) in vecs.chunks(1000).enumerate() {
        let inputs: Vec<MemorizeInput> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut m = MemorizeInput::text(format!("vector {}", b * 1000 + i)).with_vector(v.clone());
                m.entities = Some(vec![]);
                m
            })
            .collect();
        store.memorize_batch(&inputs).unwrap();
    }
    let build_secs = started.elapsed().as_secs_f64();

    let mut lat = Vec::with_capacity(queries);
    let (mut hit, mut total) = (0usize, 0usize);
    for _ in 0..queries {
        let pick = rng.below(n);
        let q = rng.perturb(&vecs[pick], synapstore::synth::world::noise_for_cos(0.9, dim));
        let exact: Vec<u64> = store.exact_search(&q, 10).unwrap().into_iter().map(|(id, _)| id).collect();
        let t = Instant::now();
        let out = store.recall(&RecallQuery { vector: Some(q), top_k: 10, ..Default::default() }).unwrap();
        lat.push(t.elapsed().as_micros() as u64);
        hit += out.hits.iter().filter(|h| exact.contains(&h.id)).count();
        total += exact.len();
    }
    let (p50, p95) = percentiles(&mut lat);
    let st = store.stats(true).unwrap();
    support::write_result(
        "hnsw",
        &json!({
            "simd_backend": math::backend(),
            "kernels": kernels,
            "vectors": n,
            "dim": dim,
            "build_secs": build_secs,
            "inserts_per_sec": n as f64 / build_secs,
            "recall_at_10": hit as f64 / total.max(1) as f64,
            "query_p50_us": p50,
            "query_p95_us": p95,
            "index": st["index"],
            "file_mb": st["file_bytes"].as_f64().unwrap_or(0.0) / 1048576.0,
            "peak_rss_mb": support::peak_rss_mb(),
        }),
    );
}
