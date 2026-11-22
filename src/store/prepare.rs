//! Pure preprocessing executed on the *caller's* thread so the single
//! writer thread only does storage work: validation, chunking, embedding,
//! normalization and entity extraction.

use super::types::{EventInput, FactInput, MemorizeInput};
use crate::chunker::chunk_text;
use crate::embed::{Embedder, SOURCE_EXTERNAL};
use crate::error::{Error, Result};
use crate::extract::extract_entities;
use crate::graph::validate_name;
use crate::hash::{NS_EVENT, chunk_id, normalize, xxh64};
use crate::math;

pub const MAX_TEXT_BYTES: usize = 1 << 20;
pub const MAX_TAGS: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedChunk {
    pub id: u64,
    pub text: String,
    pub unit: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedEvent {
    pub ts: u64,
    pub id: u64,
    pub kind: String,
    pub text: String,
    pub chunk_id: u64,
}

/// A fully validated `memorize` request.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMemorize {
    pub chunks: Vec<PreparedChunk>,
    pub entities: Vec<String>,
    pub facts: Vec<FactInput>,
    pub tags: Vec<String>,
    pub source: String,
    pub ts: u64,
    pub embedding: String,
    pub event: PreparedEvent,
}

fn check_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::InvalidInput("text must not be empty".into()));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(Error::InvalidInput(format!("text exceeds {MAX_TEXT_BYTES} bytes")));
    }
    Ok(())
}

/// Clean, de-duplicate (case-insensitively) and bound a tag list.
pub fn clean_tags(tags: &[String]) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let t = t.trim();
        if !t.is_empty() && !out.iter().any(|x| x.eq_ignore_ascii_case(t)) {
            out.push(t.to_string());
        }
    }
    if out.len() > MAX_TAGS {
        return Err(Error::InvalidInput(format!("more than {MAX_TAGS} tags")));
    }
    Ok(out)
}

/// Build the episodic record for an event.
pub fn prepare_event(e: &EventInput, now: u64) -> Result<PreparedEvent> {
    check_text(&e.text)?;
    let kind = if e.kind.trim().is_empty() { "event".to_string() } else { e.kind.trim().to_string() };
    let id = match xxh64(format!("{}\u{0}{}", normalize(&kind), normalize(&e.text)).as_bytes(), NS_EVENT) {
        0 => 1,
        h => h,
    };
    Ok(PreparedEvent {
        ts: e.timestamp_ms.unwrap_or(now),
        id,
        kind,
        text: e.text.clone(),
        chunk_id: e.chunk_id.unwrap_or(0),
    })
}

/// Characters per chunk: local models are capped at `max_chunk_chars`;
/// remote models keep their (much larger) preference, so a memory usually
/// stays one chunk and recall returns it whole.
pub fn chunk_size(embedder: &dyn Embedder, max_chunk_chars: usize) -> usize {
    match embedder.chunk_chars() {
        Some(c) if embedder.source().starts_with("remote:") => c.max(max_chunk_chars),
        Some(c) => c.min(max_chunk_chars),
        None => max_chunk_chars,
    }
}

/// Validate and expand a `memorize` request; text without a caller vector
/// is chunked and embedded with `embedder` (on the caller's thread).
pub fn prepare_memorize(
    input: &MemorizeInput,
    now: u64,
    embedder: &dyn Embedder,
    max_chunk_chars: usize,
) -> Result<PreparedMemorize> {
    check_text(&input.text)?;
    let (chunks, embedding) = match &input.vector {
        Some(v) => {
            let unit = math::normalize(v)?;
            let text = input.text.trim().to_string();
            (vec![PreparedChunk { id: chunk_id(&text), text, unit }], SOURCE_EXTERNAL.to_string())
        }
        None => {
            let size = chunk_size(embedder, max_chunk_chars);
            let texts = chunk_text(&input.text, size);
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let vecs = embedder.embed_documents(&refs)?;
            let chunks = texts
                .into_iter()
                .zip(vecs)
                .map(|(t, v)| Ok(PreparedChunk { id: chunk_id(&t), unit: math::normalize(&v)?, text: t }))
                .collect::<Result<Vec<_>>>()?;
            (chunks, embedder.source().to_string())
        }
    };
    let mut entities: Vec<String> = match &input.entities {
        Some(list) => list.iter().map(|e| e.trim().to_string()).filter(|e| !e.is_empty()).collect(),
        None => extract_entities(&input.text),
    };
    for f in &input.facts {
        validate_name("subject", &f.subject)?;
        validate_name("predicate", &f.predicate)?;
        validate_name("object", &f.object)?;
        entities.push(f.subject.trim().to_string());
        entities.push(f.object.trim().to_string());
    }
    let mut uniq: Vec<String> = Vec::new();
    for e in entities {
        validate_name("entity", &e)?;
        if !uniq.iter().any(|u| normalize(u) == normalize(&e)) {
            uniq.push(e);
        }
    }
    let ts = input.timestamp_ms.unwrap_or(now);
    let preview: String = input.text.trim().chars().take(280).collect();
    let event = prepare_event(
        &EventInput {
            kind: input.kind.clone().unwrap_or_else(|| "memory".into()),
            text: preview,
            timestamp_ms: Some(ts),
            chunk_id: chunks.first().map(|c| c.id),
        },
        now,
    )?;
    Ok(PreparedMemorize {
        chunks,
        entities: uniq,
        facts: input.facts.clone(),
        tags: clean_tags(&input.tags)?,
        source: input.source.clone().unwrap_or_default(),
        ts,
        embedding,
        event,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::Lexical;

    fn prepare_memorize(input: &MemorizeInput, now: u64, dim: usize, chunk: usize) -> Result<PreparedMemorize> {
        super::prepare_memorize(input, now, &Lexical { dim }, chunk)
    }

    #[test]
    fn external_vector_is_single_normalized_chunk() {
        let input = MemorizeInput::text("Alice met Bob").with_vector(vec![3.0, 4.0]);
        let p = prepare_memorize(&input, 10, 8, 50).unwrap();
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.chunks[0].unit, vec![0.6, 0.8]);
        assert_eq!(p.embedding, SOURCE_EXTERNAL);
        assert_eq!(p.entities, vec!["Alice", "Bob"]);
        assert_eq!(p.ts, 10);
    }

    #[test]
    fn long_text_is_chunked_with_builtin_embeddings() {
        let text = "One sentence about Rust. ".repeat(20);
        let p = prepare_memorize(&MemorizeInput::text(text), 1, 16, 60).unwrap();
        assert!(p.chunks.len() > 1);
        assert!(p.chunks.iter().all(|c| c.unit.len() == 16));
        assert_eq!(p.event.chunk_id, p.chunks[0].id);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(prepare_memorize(&MemorizeInput::text("  "), 1, 8, 50).is_err());
        let nan = MemorizeInput::text("x").with_vector(vec![f32::NAN]);
        assert!(prepare_memorize(&nan, 1, 8, 50).is_err());
        let mut bad_fact = MemorizeInput::text("x");
        bad_fact.facts.push(FactInput { subject: "a".into(), predicate: " ".into(), object: "b".into() });
        assert!(prepare_memorize(&bad_fact, 1, 8, 50).is_err());
    }

    #[test]
    fn tags_and_entities_are_deduplicated() {
        let mut input = MemorizeInput::text("x");
        input.tags = vec!["A".into(), "a".into(), " ".into(), "b".into()];
        input.entities = Some(vec!["Acme".into(), "ACME".into()]);
        let p = prepare_memorize(&input, 1, 8, 50).unwrap();
        assert_eq!(p.tags, vec!["A", "b"]);
        assert_eq!(p.entities, vec!["Acme"]);
    }
}
