//! Neural embedder vs. a reference implementation (Hugging Face tokenizers +
//! onnxruntime, see `tests/fixtures/minilm_reference.json`).
//!
//! Needs the model files: set `SYNAPSTORE_TEST_MODEL_DIR` to a directory
//! holding all-MiniLM-L6-v2 (`config.json`, `model.safetensors`,
//! `tokenizer.json` or `vocab.txt`), or `SYNAPSTORE_TEST_MODEL_DOWNLOAD=1`
//! to fetch it from the Hub. Skipped otherwise.

use serde_json::Value;
use synapstore::embed::Embedder;
use synapstore::embed::bert::{BertEmbedder, BertOptions};

fn model() -> Option<BertEmbedder> {
    if let Ok(dir) = std::env::var("SYNAPSTORE_TEST_MODEL_DIR") {
        let opts = BertOptions { max_tokens: Some(512), ..Default::default() };
        return Some(BertEmbedder::load(std::path::Path::new(&dir), "minilm", opts).expect("load model"));
    }
    if std::env::var("SYNAPSTORE_TEST_MODEL_DOWNLOAD").is_ok() {
        let e = synapstore::embed::fetch::load_model("minilm", &|m| eprintln!("{m}")).expect("download model");
        let dir = synapstore::embed::fetch::resolve("minilm").unwrap().cache_dir();
        let opts = BertOptions { max_tokens: Some(512), ..Default::default() };
        drop(e);
        return Some(BertEmbedder::load(&dir, "minilm", opts).expect("load model"));
    }
    eprintln!("skipping: set SYNAPSTORE_TEST_MODEL_DIR or SYNAPSTORE_TEST_MODEL_DOWNLOAD=1");
    None
}

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("fixtures/minilm_reference.json")).unwrap().as_array().unwrap().clone()
}

#[test]
fn matches_reference_tokens_and_embeddings() {
    let Some(m) = model() else { return };
    for r in reference() {
        let text = r["text"].as_str().unwrap();
        let want_ids: Vec<u32> = r["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(m.tokenize(text), want_ids, "tokens for {text:?}");
        let want: Vec<f32> = r["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let got = m.embed_raw(text);
        let cos: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let max_diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(cos > 0.9999 && max_diff < 2e-4, "{text:?}: cos {cos}, max diff {max_diff}");
    }
}

#[test]
fn semantic_similarity_beats_word_overlap() {
    let Some(m) = model() else { return };
    let q = m.embed_query("How do I keep my laptop from overheating?").unwrap();
    let docs = m
        .embed_documents(&[
            "Clean the fans and use a cooling pad so the notebook stays cool.",
            "My laptop bag has a pocket for keeping documents.",
        ])
        .unwrap();
    let sim = |d: &Vec<f32>| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>();
    assert!(sim(&docs[0]) > sim(&docs[1]) + 0.1, "{} vs {}", sim(&docs[0]), sim(&docs[1]));
    assert_eq!(m.dim(), 384);
    assert!(m.source().starts_with("bert:minilm:"));
}

/// multilingual-e5-small: SentencePiece/Unigram tokenizer. Set
/// `SYNAPSTORE_TEST_E5_DIR`. The reference ran the int8 ONNX export, so
/// embeddings are compared with a looser tolerance; token ids are exact.
#[test]
fn multilingual_e5_matches_reference() {
    let Ok(dir) = std::env::var("SYNAPSTORE_TEST_E5_DIR") else {
        eprintln!("skipping: set SYNAPSTORE_TEST_E5_DIR");
        return;
    };
    let opts = BertOptions { max_tokens: Some(512), ..Default::default() };
    let m = BertEmbedder::load(std::path::Path::new(&dir), "e5", opts).expect("load model");
    let refs: Vec<Value> =
        serde_json::from_str::<Value>(include_str!("fixtures/e5_reference.json")).unwrap().as_array().unwrap().clone();
    let mut worst = 1.0f32;
    for r in refs {
        let text = r["text"].as_str().unwrap();
        let want_ids: Vec<u32> = r["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(m.tokenize(text), want_ids, "tokens for {text:?}");
        let want: Vec<f32> = r["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let cos: f32 = m.embed_raw(text).iter().zip(&want).map(|(a, b)| a * b).sum();
        worst = worst.min(cos);
    }
    eprintln!("worst cosine vs int8 reference: {worst}");
    assert!(worst > 0.97, "worst cosine vs int8 reference {worst}");
}
