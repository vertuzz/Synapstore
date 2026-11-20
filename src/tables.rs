//! redb table layout. One file, several B-trees:
//!
//! | table          | key                   | value                         |
//! |----------------|-----------------------|-------------------------------|
//! | `chunks`       | chunk id              | rkyv [`Chunk`](crate::model::Chunk) |
//! | `vectors`      | chunk id              | unit-norm f32 LE bytes        |
//! | `hnsw_nodes`   | chunk id              | rkyv [`HnswNode`](crate::model::HnswNode) |
//! | `terms`        | entity/predicate id   | rkyv [`Term`](crate::model::Term) |
//! | `spo`          | (subject, predicate)  | packed object ids             |
//! | `pos`          | (predicate, object)   | packed subject ids            |
//! | `ops`          | (object, predicate)   | packed subject ids (in-edges) |
//! | `triples`      | (s, p, o)             | rkyv [`Triple`](crate::model::Triple) |
//! | `entity_chunks`| entity id             | packed chunk ids              |
//! | `episodes`     | (ts_ms, event id)     | rkyv [`MemoryItem`](crate::model::MemoryItem) |
//! | `meta`         | &str                  | u64                           |
//! | `meta_str`     | &str                  | &str                          |

use crate::error::Result;
use redb::{ReadableTable, TableDefinition, WriteTransaction};

pub type Pair = (u64, u64);
pub type Triplet = (u64, u64, u64);

pub const CHUNKS: TableDefinition<u64, &[u8]> = TableDefinition::new("chunks");
pub const VECTORS: TableDefinition<u64, &[u8]> = TableDefinition::new("vectors");
pub const HNSW_NODES: TableDefinition<u64, &[u8]> = TableDefinition::new("hnsw_nodes");
pub const TERMS: TableDefinition<u64, &[u8]> = TableDefinition::new("terms");
pub const SPO: TableDefinition<Pair, &[u8]> = TableDefinition::new("spo");
pub const POS: TableDefinition<Pair, &[u8]> = TableDefinition::new("pos");
pub const OPS: TableDefinition<Pair, &[u8]> = TableDefinition::new("ops");
pub const TRIPLES: TableDefinition<Triplet, &[u8]> = TableDefinition::new("triples");
pub const ENTITY_CHUNKS: TableDefinition<u64, &[u8]> = TableDefinition::new("entity_chunks");
pub const EPISODES: TableDefinition<Pair, &[u8]> = TableDefinition::new("episodes");
pub const META: TableDefinition<&str, u64> = TableDefinition::new("meta");
pub const META_STR: TableDefinition<&str, &str> = TableDefinition::new("meta_str");

/// Keys of the `meta` / `meta_str` tables.
pub mod keys {
    pub const SCHEMA_VERSION: &str = "schema_version";
    pub const DIM: &str = "vector.dim";
    pub const EMBEDDING_SOURCE: &str = "vector.source";
    pub const HNSW_ENTRY: &str = "hnsw.entry";
    pub const HNSW_MAX_LEVEL: &str = "hnsw.max_level";
    pub const HNSW_COUNT: &str = "hnsw.count";
    pub const HNSW_M: &str = "hnsw.m";
    pub const HNSW_EF_CONSTRUCTION: &str = "hnsw.ef_construction";
    pub const COLLISIONS: &str = "hash.collisions";
    pub const HNSW_SINCE_REPAIR: &str = "hnsw.inserts_since_repair";
}

pub const SCHEMA_VERSION: u64 = 1;

/// Create every table (idempotent) so read transactions never miss one.
pub fn create_all(txn: &WriteTransaction) -> Result<()> {
    txn.open_table(CHUNKS)?;
    txn.open_table(VECTORS)?;
    txn.open_table(HNSW_NODES)?;
    txn.open_table(TERMS)?;
    txn.open_table(SPO)?;
    txn.open_table(POS)?;
    txn.open_table(OPS)?;
    txn.open_table(TRIPLES)?;
    txn.open_table(ENTITY_CHUNKS)?;
    txn.open_table(EPISODES)?;
    txn.open_table(META_STR)?;
    let mut meta = txn.open_table(META)?;
    if meta.get(keys::SCHEMA_VERSION)?.is_none() {
        meta.insert(keys::SCHEMA_VERSION, SCHEMA_VERSION)?;
    }
    Ok(())
}

/// Read a numeric meta value.
pub fn meta_get<T>(table: &T, key: &str) -> Result<Option<u64>>
where
    T: ReadableTable<&'static str, u64>,
{
    Ok(table.get(key)?.map(|g| g.value()))
}

/// Read a string meta value.
pub fn meta_str_get<T>(table: &T, key: &str) -> Result<Option<String>>
where
    T: ReadableTable<&'static str, &'static str>,
{
    Ok(table.get(key)?.map(|g| g.value().to_string()))
}

/// Inclusive key range covering every `(first, *)` pair.
pub fn prefix(first: u64) -> std::ops::RangeInclusive<Pair> {
    (first, 0)..=(first, u64::MAX)
}
