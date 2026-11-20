//! Read access to the HNSW tables inside any redb transaction (read or
//! write). Vectors are scored zero-copy, straight from the page buffers.

use crate::archive;
use crate::error::{Error, Result};
use crate::hash::id_to_hex;
use crate::math;
use crate::model::HnswNode;
use crate::tables::{keys, meta_get};
use redb::{AccessGuard, ReadableTable};

/// Entry point of the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryPoint {
    pub id: u64,
    pub max_level: u8,
}

impl EntryPoint {
    /// Load the entry point from the `meta` table (None for an empty index).
    pub fn load<M: ReadableTable<&'static str, u64>>(meta: &M) -> Result<Option<Self>> {
        let id = meta_get(meta, keys::HNSW_ENTRY)?;
        let lvl = meta_get(meta, keys::HNSW_MAX_LEVEL)?.unwrap_or(0);
        Ok(id.map(|id| EntryPoint { id, max_level: lvl.min(u64::from(u8::MAX)) as u8 }))
    }
}

/// Borrowed view over the `vectors` and `hnsw_nodes` tables.
pub struct View<'a, V, N> {
    pub vectors: &'a V,
    pub nodes: &'a N,
}

fn corrupt(table: &'static str, id: u64, reason: &str) -> Error {
    Error::Corrupt { table, key: id_to_hex(id), reason: reason.into() }
}

impl<'a, V, N> View<'a, V, N>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    pub fn new(vectors: &'a V, nodes: &'a N) -> Self {
        View { vectors, nodes }
    }

    /// Stored vector bytes of `id` (zero-copy guard).
    pub fn vector(&self, id: u64) -> Result<AccessGuard<'a, &'static [u8]>> {
        self.vectors.get(id)?.ok_or_else(|| corrupt("vectors", id, "missing vector for indexed node"))
    }

    /// Similarity between the query and stored vector `id`.
    pub fn sim(&self, q: &[f32], id: u64) -> Result<f32> {
        let g = self.vector(id)?;
        let bytes = g.value();
        if bytes.len() != q.len() * 4 {
            return Err(corrupt("vectors", id, "vector length differs from query"));
        }
        Ok(math::dot_le(q, bytes))
    }

    /// Similarity between two stored vectors.
    pub fn sim_ids(&self, a: u64, b: u64) -> Result<f32> {
        let (ga, gb) = (self.vector(a)?, self.vector(b)?);
        Ok(math::dot_le_le(ga.value(), gb.value()))
    }

    /// Neighbor ids of `id` at `level` (written into `out`, zero-copy read).
    pub fn neighbors(&self, id: u64, level: usize, out: &mut Vec<u64>) -> Result<()> {
        out.clear();
        let Some(g) = self.nodes.get(id)? else {
            return Err(corrupt("hnsw_nodes", id, "dangling neighbor id"));
        };
        let node = archive::view::<HnswNode>(g.value())?;
        if let Some(layer) = node.layers.get(level) {
            out.extend(layer.iter().map(|l| l.id.to_native()));
        }
        Ok(())
    }

    /// Owned copy of a node (write paths and inspection).
    pub fn node(&self, id: u64) -> Result<Option<HnswNode>> {
        match self.nodes.get(id)? {
            Some(g) => Ok(Some(archive::decode::<HnswNode>(g.value())?)),
            None => Ok(None),
        }
    }
}
