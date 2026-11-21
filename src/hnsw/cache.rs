//! Write-back cache for HNSW inserts inside one write transaction.
//!
//! An insert reads a few thousand vectors/nodes and rewrites dozens of
//! neighbor lists; consecutive inserts touch largely the same
//! neighborhoods. Serving those from memory, and writing each modified
//! node back once at the end, removes most B-tree traffic from the write
//! path. Reads outside the writer keep using zero-copy [`View`]s.

use super::view::{Graph, View};
use crate::archive;
use crate::error::Result;
use crate::hash::IdHasher;
use crate::model::HnswNode;
use redb::{ReadableTable, Table};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::rc::Rc;

type Map<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;

/// Clean entries are dropped beyond this many cached vectors.
const MAX_VECTORS: usize = 200_000;
/// Dirty nodes are flushed by the caller beyond this many.
pub const FLUSH_AT: usize = 50_000;

#[derive(Default)]
pub struct WriteCache {
    vectors: RefCell<Map<Rc<[f32]>>>,
    nodes: RefCell<Map<HnswNode>>,
    dirty: RefCell<HashSet<u64, BuildHasherDefault<IdHasher>>>,
}

impl WriteCache {
    pub fn put_vector(&self, id: u64, v: &[f32]) {
        let mut vs = self.vectors.borrow_mut();
        if vs.len() >= MAX_VECTORS {
            vs.clear();
        }
        vs.insert(id, v.into());
    }

    /// Store a modified node (written back by [`WriteCache::flush`]).
    pub fn put_node(&self, id: u64, node: HnswNode) {
        self.nodes.borrow_mut().insert(id, node);
        self.dirty.borrow_mut().insert(id);
    }

    pub fn dirty_len(&self) -> usize {
        self.dirty.borrow().len()
    }

    /// Write every modified node to `nodes`.
    pub fn flush(&self, nodes: &mut Table<'_, u64, &'static [u8]>) -> Result<usize> {
        let dirty: Vec<u64> = self.dirty.borrow_mut().drain().collect();
        let map = self.nodes.borrow();
        for id in &dirty {
            if let Some(n) = map.get(id) {
                nodes.insert(*id, archive::encode(n)?.as_slice())?;
            }
        }
        Ok(dirty.len())
    }

    /// Forget everything (after other code modified the tables directly).
    /// Must only be called with no unflushed nodes.
    pub fn clear(&self) {
        debug_assert!(self.dirty.borrow().is_empty(), "clearing a cache with unflushed nodes");
        self.vectors.borrow_mut().clear();
        self.nodes.borrow_mut().clear();
    }
}

/// [`Graph`] over the cache, falling back to (and filling from) the tables.
pub struct Cached<'c, 'v, V, N> {
    pub cache: &'c WriteCache,
    pub view: View<'v, V, N>,
}

impl<V, N> Cached<'_, '_, V, N>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    fn vector(&self, id: u64) -> Result<Rc<[f32]>> {
        if let Some(v) = self.cache.vectors.borrow().get(&id) {
            return Ok(v.clone());
        }
        let v: Rc<[f32]> = archive::le_to_f32s(self.view.vector(id)?.value()).into();
        let mut vs = self.cache.vectors.borrow_mut();
        if vs.len() >= MAX_VECTORS {
            vs.clear();
        }
        vs.insert(id, v.clone());
        Ok(v)
    }

    /// Owned copy of node `id` (cache first).
    pub fn node(&self, id: u64) -> Result<Option<HnswNode>> {
        if let Some(n) = self.cache.nodes.borrow().get(&id) {
            return Ok(Some(n.clone()));
        }
        let n = self.view.node(id)?;
        if let Some(n) = &n {
            self.cache.nodes.borrow_mut().insert(id, n.clone());
        }
        Ok(n)
    }
}

impl<V, N> Graph for Cached<'_, '_, V, N>
where
    V: ReadableTable<u64, &'static [u8]>,
    N: ReadableTable<u64, &'static [u8]>,
{
    fn sim(&self, q: &[f32], id: u64) -> Result<f32> {
        let v = self.vector(id)?;
        if v.len() != q.len() {
            return self.view.sim(q, id);
        }
        Ok(crate::math::dot(q, &v))
    }

    fn sim_ids(&self, a: u64, b: u64) -> Result<f32> {
        Ok(crate::math::dot(&self.vector(a)?, &self.vector(b)?))
    }

    fn owned_vector(&self, id: u64) -> Result<Rc<[f32]>> {
        self.vector(id)
    }

    fn neighbors(&self, id: u64, level: usize, out: &mut Vec<u64>) -> Result<()> {
        if let Some(n) = self.cache.nodes.borrow().get(&id) {
            out.clear();
            if let Some(layer) = n.layers.get(level) {
                out.extend(layer.iter().map(|l| l.id));
            }
            return Ok(());
        }
        // Not cached yet: read zero-copy; only nodes being modified are cached.
        self.view.neighbors(id, level, out)
    }
}
