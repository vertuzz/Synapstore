//! `synapstore eval`: a recall regression check on your own data.
//!
//! The queries file is TSV, one `query<TAB>snippet|snippet…` per line; a
//! query is answered when a hit's text contains one of its snippets
//! (case-insensitive). Snippets survive re-chunking and re-embedding, so
//! the same file compares stores before and after a migration.

use crate::SynapStore;
use crate::error::{Error, Result};
use crate::store::RecallQuery;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq)]
pub struct EvalCase {
    pub query: String,
    pub snippets: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct EvalMiss {
    pub query: String,
    /// 1-based rank of the first relevant hit within `top_k`, if any.
    pub rank: Option<usize>,
    pub top_hit: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct EvalReport {
    pub queries: usize,
    pub top_k: usize,
    pub at_1: f64,
    pub at_5: f64,
    pub mrr: f64,
    pub mean_latency_ms: f64,
    pub embedding: String,
    /// Queries whose answer was not ranked first.
    pub misses: Vec<EvalMiss>,
}

/// Parse the TSV queries file (blank lines and `#` comments are skipped).
pub fn parse_cases(text: &str) -> Result<Vec<EvalCase>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let (q, s) = line
            .split_once('\t')
            .ok_or_else(|| Error::InvalidInput(format!("line {}: expected `query<TAB>snippet|…`", n + 1)))?;
        let snippets: Vec<String> = s.split('|').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).collect();
        if q.trim().is_empty() || snippets.is_empty() {
            return Err(Error::InvalidInput(format!("line {}: empty query or snippets", n + 1)));
        }
        out.push(EvalCase { query: q.trim().to_string(), snippets });
    }
    Ok(out)
}

fn short(s: &str) -> String {
    let t: String = s.chars().take(120).collect();
    if t.len() < s.len() { format!("{t}…") } else { t }
}

/// Run every case against `store` with `recall(query, top_k)`.
pub fn run(store: &SynapStore, cases: &[EvalCase], top_k: usize) -> Result<EvalReport> {
    let top_k = top_k.clamp(1, 100);
    let mut rep = EvalReport { queries: cases.len(), top_k, ..Default::default() };
    let (mut ms, mut h1, mut h5, mut rr) = (0.0, 0usize, 0usize, 0.0);
    for c in cases {
        let t = std::time::Instant::now();
        let out = store.recall(&RecallQuery::text(c.query.clone(), top_k))?;
        ms += t.elapsed().as_secs_f64() * 1000.0;
        rep.embedding = out.embedding.clone();
        let rank = out.hits.iter().position(|h| {
            let text = h.text.to_lowercase();
            c.snippets.iter().any(|s| text.contains(s.as_str()))
        });
        h1 += usize::from(rank == Some(0));
        h5 += usize::from(rank.is_some_and(|r| r < 5));
        rr += rank.map_or(0.0, |r| 1.0 / (r + 1) as f64);
        if rank != Some(0) {
            let top_hit = out.hits.first().map(|h| short(&h.text));
            rep.misses.push(EvalMiss { query: c.query.clone(), rank: rank.map(|r| r + 1), top_hit });
        }
    }
    let n = cases.len().max(1) as f64;
    rep.at_1 = 100.0 * h1 as f64 / n;
    rep.at_5 = 100.0 * h5 as f64 / n;
    rep.mrr = rr / n;
    rep.mean_latency_ms = ms / n;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tsv_cases() {
        let c = parse_cases("# comment\nwho is Bob\tBob works| acme \n\n").unwrap();
        assert_eq!(c, vec![EvalCase { query: "who is Bob".into(), snippets: vec!["bob works".into(), "acme".into()] }]);
        assert!(parse_cases("no tab here").is_err());
        assert!(parse_cases("q\t | ").is_err());
    }
}
