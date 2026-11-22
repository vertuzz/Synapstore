//! `synapstore reindex SRC DST`: rebuild a store into a new file with
//! another embedder. The source is opened read-only and never modified.
//!
//! * memories are re-chunked and re-embedded; their tags, entities,
//!   timestamps and counters are kept. Memories an older version split into
//!   several chunks are joined back (only when the split is unambiguous);
//! * facts keep their timestamps, counters and (remapped) source memory;
//! * the episodic log and API tokens are copied as they are;
//! * entities that older heuristics extracted from stop words, months or
//!   emphasis ("Он", "ВАЖНО", "November") are dropped unless a fact uses them;
//! * HNSW tombstones of forgotten memories are not carried over.

use crate::archive::decode;
use crate::chunker::{chunk_text, sentences};
use crate::error::{Error, Result};
use crate::extract::is_junk_entity;
use crate::hash::{IdMap, IdSet, chunk_id};
use crate::model::{Chunk, MemoryItem, Term, Triple};
use crate::store::import::{ImportFact, ImportMemory, ImportOp};
use crate::store::prepare::{PreparedChunk, PreparedEvent, chunk_size};
use crate::tables::{AUTH_TOKENS, CHUNKS, EPISODES, TERMS, TRIPLES};
use crate::{SynapStore, math};
use redb::{ReadOnlyDatabase, ReadableDatabase, ReadableTable};
use serde::Serialize;
use std::path::Path;

/// Chunk size of the stores being migrated (the 0.2 default).
pub const OLD_CHUNK_CHARS: usize = 1200;

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ReindexReport {
    pub chunks_in: usize,
    pub memories: usize,
    pub merged_groups: usize,
    pub chunks_out: usize,
    pub facts: usize,
    pub events: usize,
    pub tokens: usize,
    pub entities_dropped: usize,
    pub embedding: String,
}

struct Source {
    chunks: Vec<Chunk>,
    terms: IdMap<Term>,
    triples: Vec<Triple>,
    events: Vec<MemoryItem>,
    tokens: Vec<(Vec<u8>, Vec<u8>)>,
}

fn read_source(path: &Path) -> Result<Source> {
    let db = ReadOnlyDatabase::open(path)?;
    let txn = db.begin_read()?;
    let mut s = Source { chunks: vec![], terms: IdMap::default(), triples: vec![], events: vec![], tokens: vec![] };
    for item in txn.open_table(CHUNKS)?.iter()? {
        s.chunks.push(decode(item?.1.value())?);
    }
    for item in txn.open_table(TERMS)?.iter()? {
        let (k, v) = item?;
        s.terms.insert(k.value(), decode(v.value())?);
    }
    for item in txn.open_table(TRIPLES)?.iter()? {
        s.triples.push(decode(item?.1.value())?);
    }
    for item in txn.open_table(EPISODES)?.iter()? {
        s.events.push(decode(item?.1.value())?);
    }
    for item in txn.open_table(AUTH_TOKENS)?.iter()? {
        let (k, v) = item?;
        s.tokens.push((k.value().to_vec(), v.value().to_vec()));
    }
    Ok(s)
}

/// Would greedy packing at `max` chars have closed each chunk before the next?
fn chain_ok(texts: &[&str], max: usize) -> bool {
    texts.windows(2).all(|w| {
        let first = sentences(w[1]).first().map_or(0, |s| s.chars().count());
        w[0].chars().count() + 1 + first > max
    })
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (i, x) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, *x);
            out.push(p);
        }
    }
    out
}

/// The unique order (starting at `head`) in which `texts` are consecutive
/// chunks of one memory, or `None` when there is no or more than one.
pub fn split_order(texts: &[&str], head: usize, max: usize) -> Option<Vec<usize>> {
    let rest: Vec<usize> = (0..texts.len()).filter(|i| *i != head).collect();
    if rest.len() > 4 {
        return None;
    }
    let ok: Vec<Vec<usize>> = permutations(&rest)
        .into_iter()
        .map(|p| std::iter::once(head).chain(p).collect::<Vec<_>>())
        .filter(|o| chain_ok(&o.iter().map(|i| texts[*i]).collect::<Vec<_>>(), max))
        .collect();
    (ok.len() == 1).then(|| ok.into_iter().next().unwrap_or_default())
}

/// Group chunk indices into memories. Chunks written by one `memorize`
/// share `created_at` and `source`, and only the first is referenced by its
/// episode (`heads`); anything ambiguous stays a memory of its own.
pub fn group_memories(chunks: &[Chunk], heads: &IdSet, max: usize) -> (Vec<Vec<usize>>, usize) {
    let mut by_key: Vec<((u64, &str), Vec<usize>)> = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        let key = (c.created_at, c.source.as_str());
        match by_key.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => by_key.push((key, vec![i])),
        }
    }
    let (mut out, mut merged) = (Vec::new(), 0);
    for (_, members) in by_key {
        let hs: Vec<usize> = (0..members.len()).filter(|j| heads.contains(&chunks[members[*j]].id)).collect();
        let texts: Vec<&str> = members.iter().map(|i| chunks[*i].text.as_str()).collect();
        match (members.len(), hs.as_slice()) {
            (n, [h]) if n > 1 => match split_order(&texts, *h, max) {
                Some(order) => {
                    merged += 1;
                    out.push(order.into_iter().map(|j| members[j]).collect());
                }
                None => out.extend(members.into_iter().map(|i| vec![i])),
            },
            _ => out.extend(members.into_iter().map(|i| vec![i])),
        }
    }
    (out, merged)
}

/// Rebuild `src` into the empty store `dst` (opened with the new embedder).
pub fn reindex(src: &Path, dst: &SynapStore, merge_split: bool, progress: &dyn Fn(&str)) -> Result<ReindexReport> {
    if dst.space()?.0.is_some() {
        return Err(Error::InvalidInput(format!("{} is not empty; reindex needs a new file", dst.path().display())));
    }
    let s = read_source(src)?;
    let heads: IdSet = s.events.iter().map(|e| e.chunk_id).filter(|c| *c != 0).collect();
    let (groups, merged) = if merge_split {
        group_memories(&s.chunks, &heads, OLD_CHUNK_CHARS)
    } else {
        ((0..s.chunks.len()).map(|i| vec![i]).collect(), 0)
    };
    let in_facts: IdSet = s.triples.iter().flat_map(|t| [t.subject, t.object]).collect();
    let size = chunk_size(dst.embedder(), dst.config().max_chunk_chars);
    let mut rep = ReindexReport { chunks_in: s.chunks.len(), merged_groups: merged, ..Default::default() };
    let mut remap: IdMap<u64> = IdMap::default();
    let mut ops = Vec::new();
    for (n, g) in groups.iter().enumerate() {
        let cs: Vec<&Chunk> = g.iter().map(|i| &s.chunks[*i]).collect();
        let text = cs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        let (mut tags, mut entities, mut ents) = (Vec::<String>::new(), Vec::<String>::new(), IdSet::default());
        for c in &cs {
            for t in &c.tags {
                if !tags.iter().any(|x| x.eq_ignore_ascii_case(t)) {
                    tags.push(t.clone());
                }
            }
            for e in &c.entities {
                let Some(term) = s.terms.get(e) else { continue };
                if !ents.insert(*e) {
                    continue;
                }
                if is_junk_entity(&term.name) && !in_facts.contains(e) {
                    rep.entities_dropped += 1;
                } else {
                    entities.push(term.name.clone());
                }
            }
        }
        let texts = chunk_text(&text, size);
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vecs = dst.embedder().embed_documents(&refs)?;
        let chunks = texts
            .iter()
            .zip(vecs)
            .map(|(t, v)| Ok(PreparedChunk { id: chunk_id(t), text: t.clone(), unit: math::normalize(&v)? }))
            .collect::<Result<Vec<_>>>()?;
        let Some(first) = chunks.first().map(|c| c.id) else { continue };
        for c in &cs {
            remap.insert(c.id, first);
        }
        rep.chunks_out += chunks.len();
        ops.push(ImportOp::Memory(Box::new(ImportMemory {
            chunks,
            embedding: dst.embedder().source().to_string(),
            tags,
            entities,
            source: cs[0].source.clone(),
            created_at: cs.iter().map(|c| c.created_at).min().unwrap_or(0),
            updated_at: cs.iter().map(|c| c.updated_at).max().unwrap_or(0),
            hits: cs.iter().map(|c| c.hits).max().unwrap_or(1),
        })));
        if ops.len() == 25 || n + 1 == groups.len() {
            progress(&format!("embedded {} of {} memories", n + 1, groups.len()));
            dst.import(std::mem::take(&mut ops))?;
        }
    }
    rep.memories = groups.len();
    let name = |id: &u64| s.terms.get(id).map(|t| t.name.clone());
    for t in &s.triples {
        let (Some(subject), Some(predicate), Some(object)) = (name(&t.subject), name(&t.predicate), name(&t.object))
        else {
            continue;
        };
        let source_chunk = remap.get(&t.source_chunk).copied().unwrap_or(t.source_chunk);
        let (created_at, updated_at, hits) = (t.created_at, t.updated_at, t.hits);
        ops.push(ImportOp::Fact(ImportFact { subject, predicate, object, created_at, updated_at, source_chunk, hits }));
    }
    rep.facts = ops.len();
    for e in &s.events {
        let chunk_id = remap.get(&e.chunk_id).copied().unwrap_or(e.chunk_id);
        let (ts, id, kind, text) = (e.ts, e.id, e.kind.clone(), e.text.clone());
        ops.push(ImportOp::Event(PreparedEvent { ts, id, kind, text, chunk_id }));
    }
    rep.events = s.events.len();
    rep.tokens = s.tokens.len();
    ops.extend(s.tokens.into_iter().map(|(key, value)| ImportOp::Token { key, value }));
    dst.import(ops)?;
    dst.repair_index()?;
    rep.embedding = dst.embedder().source().to_string();
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_order_finds_the_unique_chain() {
        let a = "A".repeat(1150) + ".";
        let b = "B".repeat(100) + ". tail";
        let c = "C".repeat(60) + ".";
        // a then b: 1151 + 1 + 101 > 1200; b then c: 108 + 1 + 61 is not.
        assert_eq!(split_order(&[&a, &b], 0, 1200), Some(vec![0, 1]));
        assert_eq!(split_order(&[&b, &c], 0, 1200), None);
        assert_eq!(split_order(&[&c, &a], 1, 1200), Some(vec![1, 0]));
    }
}
