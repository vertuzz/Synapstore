//! Multi-agent operation (PRD §6): many threads share one store through the
//! single writer thread, and several *processes* share one file.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::synth::rng::Rng;
use synapstore::{Config, SynapStore};

#[test]
fn threads_write_and_read_concurrently() {
    let dir = common::TempDir::new("threads");
    let store = SynapStore::open(dir.file("t.redb"), Config::default()).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let (s, stop) = (store.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut rng = Rng::new(99);
            let mut reads = 0;
            while !stop.load(Ordering::Relaxed) {
                let q = RecallQuery { vector: Some(rng.unit_vector(24)), top_k: 3, ..Default::default() };
                s.recall(&q).unwrap();
                reads += 1;
            }
            reads
        })
    };
    let writers: Vec<_> = (0..8)
        .map(|t| {
            let s = store.clone();
            std::thread::spawn(move || {
                let mut rng = Rng::new(t);
                for i in 0..60 {
                    // Threads overlap on half of their texts: upserts must converge.
                    let text = if i % 2 == 0 { format!("shared {i}") } else { format!("thread {t} item {i}") };
                    let v = Rng::new(i as u64 + if i % 2 == 0 { 0 } else { 1000 * (t + 1) }).unit_vector(24);
                    let _ = rng.next_u64();
                    s.memorize(MemorizeInput::text(text).with_vector(v)).unwrap();
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    assert!(reader.join().unwrap() > 0);
    let st = store.stats(true).unwrap();
    assert_eq!(st["counts"]["chunks"], 30 + 8 * 30);
    assert_eq!(st["index"]["nodes"], 30 + 8 * 30);
    assert_eq!(st["index"]["reachable_from_entry"], 30 + 8 * 30);
    // Concurrent submissions are merged into group commits.
    let c = &st["metrics"]["counters"];
    assert!(c["group_commits"].as_u64().unwrap() > 0, "{c}");
}

fn spawn_mcp(db: &std::path::Path) -> std::process::Child {
    Command::new(env!("CARGO_BIN_EXE_synapstore"))
        .args(["--mcp", db.to_str().unwrap(), "--no-log"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn synapstore")
}

#[test]
fn several_processes_share_one_file() {
    let dir = common::TempDir::new("procs");
    let db = dir.file("shared.redb");
    let procs = 3;
    let per = 40;
    let mut children: Vec<_> = (0..procs).map(|_| spawn_mcp(&db)).collect();
    // Interleave writes from every process so their transactions contend.
    let mut readers: Vec<_> = children.iter_mut().map(|c| BufReader::new(c.stdout.take().unwrap())).collect();
    for i in 0..per {
        for (p, child) in children.iter_mut().enumerate() {
            let msg = serde_json::json!({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": {"name": "memorize", "arguments": {"text": format!("process {p} note {i}"), "entities": []}}});
            writeln!(child.stdin.as_mut().unwrap(), "{msg}").unwrap();
        }
        for r in readers.iter_mut() {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(v["result"]["isError"], false, "{line}");
        }
    }
    for mut c in children {
        drop(c.stdin.take());
        assert!(c.wait().unwrap().success());
    }
    let s = SynapStore::open(&db, Config::default()).unwrap();
    let st = s.stats(true).unwrap();
    assert_eq!(st["counts"]["chunks"], procs * per);
    assert_eq!(st["index"]["reachable_from_entry"], procs * per);
}
