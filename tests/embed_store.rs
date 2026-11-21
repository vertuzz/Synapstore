//! A store driven by a neural (BERT) embedder: chunking, embedding,
//! recall, and remembering the embedder across reopen.

#[path = "support/tiny_bert.rs"]
#[allow(dead_code)]
mod tiny_bert;

use synapstore::embed::EmbedderSpec;
use synapstore::store::{MemorizeInput, RecallQuery};
use synapstore::{Config, SynapStore};

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("synapstore-embed-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn neural_store_round_trip_and_pinning() {
    let dir = tmp("pin");
    let model = dir.join("tiny");
    tiny_bert::write(&model, 16);
    let db = dir.join("m.redb");
    let spec = EmbedderSpec::Model(model.display().to_string());
    {
        let s = SynapStore::open(&db, Config { embedder: spec.clone(), ..Config::default() }).unwrap();
        assert_eq!(s.embedder().dim(), 16);
        assert!(s.embedder().source().starts_with("bert:tiny:"));
        let o = s.memorize(MemorizeInput::text("alice drinks dark roast coffee")).unwrap();
        assert_eq!(o.dim, 16);
        s.memorize(MemorizeInput::text("rust is a fast and safe python")).unwrap();
        let hits = s.recall(&RecallQuery::text("alice drinks dark roast coffee", 1)).unwrap().hits;
        assert!(hits[0].text.contains("coffee"));
    }
    {
        // No --embedder: the store remembers which one built it.
        let s = SynapStore::open(&db, Config::default()).unwrap();
        assert!(s.embedder().source().starts_with("bert:tiny:"));
        assert_eq!(s.stats(false).unwrap()["vector"]["embedder_spec"], model.display().to_string());
        assert_eq!(s.recall(&RecallQuery::text("alice drinks dark roast coffee", 1)).unwrap().hits.len(), 1);
    }
    // An explicit, different embedder is refused instead of mixing spaces.
    let err = SynapStore::open(&db, Config { embedder: EmbedderSpec::Lexical, ..Config::default() }).err().unwrap();
    assert!(err.to_string().contains("was built with embedder"), "{err}");
    // A model whose weights changed under the same name is refused too.
    tiny_bert::write(&model, 8);
    assert!(SynapStore::open(&db, Config { embedder: spec, ..Config::default() }).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lexical_store_still_defaults() {
    let dir = tmp("lex");
    let s = SynapStore::open(dir.join("m.redb"), Config::default()).unwrap();
    assert_eq!(s.embedder().source(), synapstore::embed::SOURCE_BUILTIN);
    s.memorize(MemorizeInput::text("Bob likes green tea")).unwrap();
    assert_eq!(s.stats(false).unwrap()["vector"]["embedder_spec"], serde_json::Value::Null);
    std::fs::remove_dir_all(&dir).ok();
}
