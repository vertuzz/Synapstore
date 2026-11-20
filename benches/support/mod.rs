//! Shared benchmark helpers: env knobs, peak RSS, allocation counting,
//! JSON result files consumed by `scripts/perf_gate.py`.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Peak resident set size in MiB (Linux `VmHWM`; 0 elsewhere).
pub fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Global allocator that counts bytes while `TRACKING` is on.
pub struct Counting;
pub static TRACKING: AtomicBool = AtomicBool::new(false);
pub static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if TRACKING.load(Relaxed) {
            BYTES.fetch_add(l.size() as u64, Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

/// Bytes allocated (process-wide) while running `f`.
pub fn allocated<T>(f: impl FnOnce() -> T) -> (T, u64) {
    BYTES.store(0, Relaxed);
    TRACKING.store(true, Relaxed);
    let r = f();
    TRACKING.store(false, Relaxed);
    (r, BYTES.load(Relaxed))
}

/// Write `value` to `bench/results/<name>.json` (or `$SYNAPSTORE_BENCH_OUT`).
pub fn write_result(name: &str, value: &serde_json::Value) {
    let dir = std::env::var("SYNAPSTORE_BENCH_OUT").unwrap_or_else(|_| "bench/results".into());
    let _ = std::fs::create_dir_all(&dir);
    let path = format!("{dir}/{name}.json");
    std::fs::write(&path, serde_json::to_string_pretty(value).unwrap()).expect("write bench result");
    println!("{}\n-> {path}", serde_json::to_string_pretty(value).unwrap());
}

/// A scratch directory under the system temp dir, removed on drop.
pub struct Scratch(pub std::path::PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("synapstore-bench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Scratch(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
