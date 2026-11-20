//! Persistent data contracts. Every record is archived with rkyv and read back
//! zero-copy straight out of redb page buffers (see [`crate::archive`]).

use rkyv::{Archive, Deserialize, Serialize};

/// Layer 1 — a semantic chunk. Its id is `hash(normalized(text))`; the
/// embedding lives in the `vectors` table under the same id.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Chunk {
    pub id: u64,
    pub text: String,
    pub tags: Vec<String>,
    /// Entity ids mentioned by this chunk (graph back-links).
    pub entities: Vec<u64>,
    pub source: String,
    /// Unix milliseconds.
    pub created_at: u64,
    /// Unix milliseconds of the last (re)memorization; drives time-decay.
    pub updated_at: u64,
    /// How many times this exact content has been memorized.
    pub hits: u32,
}

/// Kind discriminator for [`Term`].
pub const TERM_ENTITY: u8 = 0;
pub const TERM_PREDICATE: u8 = 1;

/// A named graph vertex (entity) or edge label (predicate).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Term {
    pub id: u64,
    pub name: String,
    pub kind: u8,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Layer 2 — one Subject-Predicate-Object fact with provenance.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: u64,
    pub predicate: u64,
    pub object: u64,
    pub created_at: u64,
    pub updated_at: u64,
    /// Chunk that asserted the fact, or 0.
    pub source_chunk: u64,
    pub hits: u32,
}

/// Layer 3 — one append-only episodic event (thought, action, tool result).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MemoryItem {
    /// Unix milliseconds.
    pub ts: u64,
    /// `hash(kind, text)`: identical events at the same instant are idempotent.
    pub id: u64,
    pub kind: String,
    pub text: String,
    /// Related semantic chunk, or 0.
    pub chunk_id: u64,
}

/// One HNSW edge. `sim` caches the cosine similarity between both endpoints
/// so neighbor lists can be pruned without re-reading vectors.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Link {
    pub id: u64,
    pub sim: f32,
}

/// An HNSW graph node: one redb record holding neighbor keys per layer.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HnswNode {
    pub level: u8,
    pub layers: Vec<Vec<Link>>,
}

impl HnswNode {
    pub fn empty(level: u8) -> Self {
        HnswNode { level, layers: vec![Vec::new(); usize::from(level) + 1] }
    }
}
