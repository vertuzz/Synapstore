//! PRD §10 "self-regulating code": no source module may exceed 300 lines.

use std::path::Path;

const LIMIT: usize = 300;

fn walk(dir: &Path, out: &mut Vec<(String, usize)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let lines = std::fs::read_to_string(&p).map(|s| s.lines().count()).unwrap_or(0);
            out.push((p.display().to_string(), lines));
        }
    }
}

#[test]
fn every_module_is_at_most_300_lines() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for d in ["src", "tests", "benches"] {
        walk(&root.join(d), &mut files);
    }
    assert!(files.len() > 30, "walked too few files: {files:?}");
    let over: Vec<_> = files.iter().filter(|(_, n)| *n > LIMIT).collect();
    assert!(over.is_empty(), "modules over {LIMIT} lines: {over:?}");
}
