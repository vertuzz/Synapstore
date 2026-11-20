//! Memory profiling of the read path (Iteration 5): a counting global
//! allocator proves that rkyv views do not copy record payloads out of
//! redb's page buffers, and that `recall` only materializes the final hits.

mod common;

use redb::{Database, ReadableDatabase, ReadableTable};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use synapstore::archive::{decode, view};
use synapstore::model::Chunk;
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::synth::rng::Rng;
use synapstore::tables::CHUNKS;
use synapstore::{Config, SynapStore};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: forwards to the system allocator; only adds thread-local counting.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = TRACK.try_with(|t| {
            if t.get() {
                let _ = BYTES.try_with(|b| b.set(b.get() + l.size() as u64));
            }
        });
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes allocated by this thread while running `f`.
fn measure<T>(f: impl FnOnce() -> T) -> (T, u64) {
    BYTES.with(|b| b.set(0));
    TRACK.with(|t| t.set(true));
    let r = f();
    TRACK.with(|t| t.set(false));
    (r, BYTES.with(Cell::get))
}

const N: usize = 1500;
const TEXT: usize = 4096;

#[test]
fn reads_are_zero_copy() {
    let dir = common::TempDir::new("zc");
    let path = dir.file("z.redb");
    let mut rng = Rng::new(1);
    let vecs: Vec<Vec<f32>> = (0..N).map(|_| rng.unit_vector(32)).collect();
    let ids: Vec<u64> = {
        let s = SynapStore::open(&path, Config::default()).unwrap();
        let inputs: Vec<MemorizeInput> = (0..N)
            .map(|i| {
                let mut m = MemorizeInput::text(format!("{i:05} {}", "x".repeat(TEXT))).with_vector(vecs[i].clone());
                m.entities = Some(vec![]);
                m
            })
            .collect();
        s.memorize_batch(&inputs).unwrap().into_iter().map(|o| o.chunk_ids[0]).collect()
    };

    let db = Database::open(&path).unwrap();
    let txn = db.begin_read().unwrap();
    let t = txn.open_table(CHUNKS).unwrap();
    let scan = |zero_copy: bool| {
        let mut total = 0usize;
        for id in &ids {
            let g = t.get(*id).unwrap().unwrap();
            total += if zero_copy {
                view::<Chunk>(g.value()).unwrap().text.len()
            } else {
                decode::<Chunk>(g.value()).unwrap().text.len()
            };
        }
        total
    };
    scan(true); // warm the page cache
    let (sum_view, bytes_view) = measure(|| scan(true));
    let (sum_decode, bytes_decode) = measure(|| scan(false));
    assert_eq!(sum_view, sum_decode);
    assert!(sum_view >= N * TEXT);
    println!("zero-copy scan: {bytes_view} B allocated; owned decode: {bytes_decode} B");
    assert!(bytes_view < (N * 256) as u64, "view path allocated {bytes_view} B for {N} records");
    assert!(bytes_decode >= (N * TEXT) as u64, "decode path must copy the payload");
}

#[test]
fn recall_materializes_only_final_hits() {
    let dir = common::TempDir::new("zc-recall");
    let s = SynapStore::open(dir.file("r.redb"), Config::default()).unwrap();
    let mut rng = Rng::new(2);
    let inputs: Vec<MemorizeInput> = (0..N)
        .map(|i| {
            let mut m = MemorizeInput::text(format!("{i:05} {}", "y".repeat(TEXT))).with_vector(rng.unit_vector(32));
            m.entities = Some(vec![]);
            m
        })
        .collect();
    s.memorize_batch(&inputs).unwrap();
    let q = RecallQuery { vector: Some(rng.unit_vector(32)), top_k: 3, ef: Some(200), ..Default::default() };
    s.recall(&q).unwrap();
    let (out, bytes) = measure(|| s.recall(&q).unwrap());
    assert_eq!(out.candidates, 200);
    println!("recall over {} candidates allocated {bytes} B", out.candidates);
    // 3 hits × 4 KiB are copied out; copying all 200 candidates would be ≥ 800 KiB.
    assert!(bytes < (200 * TEXT / 4) as u64, "recall allocated {bytes} B");
    let zc = s.metrics().snapshot()["counters"]["zero_copy_reads"].as_u64().unwrap();
    assert!(zc >= 400);
}
