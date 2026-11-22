//! Degraded `recall` for when the embedding endpoint is unreachable: a
//! brute-force word-overlap scan over every memory, so agents still get an
//! answer (flagged in `embedding` and `warning`) instead of an error.

use super::SynapStore;
use super::recall::{RankOpts, hit, rank};
use super::types::{RecallOutcome, RecallQuery};
use crate::archive;
use crate::embed::tokens;
use crate::error::Result;
use crate::hnsw::Scored;
use crate::model::Chunk;
use crate::tables::{CHUNKS, TERMS};
use crate::timefmt::now_ms;
use redb::{ReadTransaction, ReadableTable};
use std::collections::HashSet;

/// Marker in `RecallOutcome::embedding` for fallback answers.
pub const LEXICAL_FALLBACK: &str = "lexical-fallback";

/// Word keys: the first six characters of each token (a cheap stem that
/// also works for inflected languages), single letters dropped.
pub fn word_keys(text: &str) -> HashSet<String> {
    tokens(text).into_iter().filter(|t| t.chars().count() > 1).map(|t| t.chars().take(6).collect()).collect()
}

/// Share of the query's word keys found in `doc` (0..=1).
pub fn overlap(query: &HashSet<String>, doc: &HashSet<String>) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let matches = |k: &String| {
        doc.contains(k)
            || k.chars().count() >= 4
                && doc
                    .iter()
                    .any(|d| d.chars().count() >= 4 && (d.starts_with(k.as_str()) || k.starts_with(d.as_str())))
    };
    query.iter().filter(|k| matches(k)).count() as f32 / query.len() as f32
}

impl SynapStore {
    pub(super) fn lexical_fallback(
        &self,
        q: &RecallQuery,
        txn: &ReadTransaction,
        why: String,
    ) -> Result<RecallOutcome> {
        self.log().warn("recall_fallback", &[("error", why.clone())]);
        let query = word_keys(q.text.as_deref().unwrap_or_default());
        let (chunks, terms) = (txn.open_table(CHUNKS)?, txn.open_table(TERMS)?);
        let mut cands = Vec::new();
        for item in chunks.iter()? {
            let (k, v) = item?;
            let c = archive::view::<Chunk>(v.value())?;
            let sim = overlap(&query, &word_keys(c.text.as_str()));
            if sim > 0.0 {
                cands.push(Scored { sim, id: k.value() });
            }
        }
        let cfg = self.config();
        let now = q.now_ms.unwrap_or_else(now_ms);
        let decay_p = q.decay.unwrap_or(cfg.decay);
        let opts =
            RankOpts { now, decay: &decay_p, decay_tags: &cfg.decay_tags, max_age_ms: q.max_age_ms, tags: &q.tags };
        let ranked = rank(&chunks, &cands, &opts, self.metrics())?;
        let mut hits = Vec::new();
        for r in ranked.iter().take(q.top_k.clamp(1, 1000)) {
            if let Some(h) = hit(&chunks, &terms, r)? {
                hits.push(h);
            }
        }
        Ok(RecallOutcome {
            hits,
            candidates: cands.len(),
            embedding: LEXICAL_FALLBACK.into(),
            warning: Some(format!("semantic search unavailable ({why}); results ranked by word overlap")),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_uses_short_stems() {
        let q = word_keys("страховка машины");
        let d = word_keys("Страховку машин оформили в InShared");
        assert_eq!(overlap(&q, &d), 1.0);
        assert_eq!(overlap(&word_keys("dentist"), &d), 0.0);
        assert_eq!(overlap(&word_keys(""), &d), 0.0);
    }
}
