//! Model download through `curl` from a fake Hugging Face Hub: files are
//! fetched, verified against `X-Linked-Etag`, cached atomically, and a
//! corrupted transfer is rejected. (One test: it sets process env vars.)

#[path = "support/tiny_bert.rs"]
#[allow(dead_code)]
mod tiny_bert;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use synapstore::crypto::{hex, sha256};
use synapstore::embed::{Embedder, fetch};

/// Serves `/<repo>/resolve/<rev>/<file>`; `corrupt` flips the announced hash.
fn hub(files: HashMap<String, Vec<u8>>, corrupt: Arc<Mutex<bool>>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap() == 0 || h.trim().is_empty() {
                    break;
                }
            }
            let file = path.strip_prefix("/org/tiny/resolve/main/").unwrap_or("").to_string();
            let mut s = s;
            match files.get(&file) {
                Some(body) => {
                    let mut etag = hex(&sha256(body));
                    if *corrupt.lock().unwrap() {
                        etag = "0".repeat(64);
                    }
                    let lfs = if file.ends_with(".safetensors") {
                        format!("X-Linked-Etag: \"{etag}\"\r\n")
                    } else {
                        String::new()
                    };
                    write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{lfs}Connection: close\r\n\r\n", body.len())
                        .unwrap();
                    s.write_all(body).unwrap();
                }
                None => write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap(),
            }
        }
    });
    format!("http://{addr}")
}

#[test]
fn downloads_verifies_and_caches() {
    if std::process::Command::new("curl").arg("--version").output().is_err() {
        eprintln!("skipping: curl not installed");
        return;
    }
    let cache = std::env::temp_dir().join(format!("synapstore-models-{}", std::process::id()));
    std::fs::remove_dir_all(&cache).ok();
    let files: HashMap<String, Vec<u8>> = tiny_bert::files(16).into_iter().collect();
    let corrupt = Arc::new(Mutex::new(true));
    let endpoint = hub(files, corrupt.clone());
    // SAFETY: this test binary has a single test, so no other thread reads the environment.
    unsafe {
        std::env::set_var("HF_ENDPOINT", &endpoint);
        std::env::set_var("SYNAPSTORE_MODELS", &cache);
        std::env::remove_var("HF_HUB_OFFLINE");
    }
    let r = fetch::resolve("hf:org/tiny").unwrap();
    assert!(r.cache_dir().starts_with(&cache));

    let err = fetch::load_model("hf:org/tiny", &|_| {}).err().expect("corrupt download must fail");
    assert!(err.to_string().contains("checksum mismatch"), "{err}");
    assert!(!r.cache_dir().exists(), "a failed download must not leave a cache entry");

    *corrupt.lock().unwrap() = false;
    let lines = Mutex::new(Vec::new());
    let e = fetch::load_model("hf:org/tiny", &|m| lines.lock().unwrap().push(m.to_string())).unwrap();
    assert_eq!(e.dim(), 16);
    assert!(lines.lock().unwrap()[0].starts_with("downloading org/tiny@main"));
    let manifest = std::fs::read_to_string(r.cache_dir().join(".synapstore-complete")).unwrap();
    assert!(manifest.contains("model.safetensors") && manifest.contains("config.json"));
    assert!(!r.cache_dir().join("tokenizer.json").exists(), "404s for optional files are skipped");

    // Cached: works offline, with no further requests.
    unsafe { std::env::set_var("HF_HUB_OFFLINE", "1") };
    let again = fetch::load_model("hf:org/tiny", &|_| {}).unwrap();
    assert_eq!(again.source(), e.source());
    assert!(fetch::load_model("hf:org/other", &|_| {}).is_err());
    std::fs::remove_dir_all(&cache).ok();
}
