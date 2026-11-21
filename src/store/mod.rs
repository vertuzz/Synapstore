//! `SynapStore`: the public, thread-safe handle tying the three memory layers
//! together. Reads run concurrently on MVCC snapshots from any thread;
//! writes are funnelled to the single writer thread.

pub mod apply;
pub mod explain;
pub mod prepare;
pub mod reads;
pub mod recall;
pub mod space;
pub mod stats;
pub mod tokens;
pub mod types;
pub mod writer;

pub use types::*;

use crate::decay::DecayParams;
use crate::embed::{BUILTIN_DIM, Embedder, EmbedderSpec};
use crate::error::{Error, Result};
use crate::hnsw::HnswParams;
use crate::logger::{Log, LogConfig, Logger};
use crate::metrics::{Metrics, Op};
use crate::retry::{self, RetryPolicy};
use crate::tables::{self, META, META_STR, keys, meta_get, meta_str_get};
use crate::timefmt::now_ms;
use apply::{WriteOp, WriteResult};
use redb::{Database, ReadableDatabase};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use writer::{Writer, WriterCtx};

#[derive(Debug, Clone)]
pub struct Config {
    pub hnsw: HnswParams,
    pub decay: DecayParams,
    pub retry: RetryPolicy,
    /// redb page-cache size.
    pub cache_bytes: usize,
    /// Text embedder (see [`EmbedderSpec`]); `Auto` reuses the store's.
    pub embedder: EmbedderSpec,
    /// Print embedder status (model downloads) to stderr.
    pub embedder_progress: bool,
    /// Dimension of the lexical embedder for a fresh store.
    pub builtin_dim: usize,
    pub max_chunk_chars: usize,
    pub log: Option<LogConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hnsw: HnswParams::default(),
            decay: DecayParams::default(),
            retry: RetryPolicy::default(),
            cache_bytes: 256 * 1024 * 1024,
            embedder: EmbedderSpec::Auto,
            embedder_progress: false,
            builtin_dim: BUILTIN_DIM,
            max_chunk_chars: 1200,
            log: None,
        }
    }
}

struct Inner {
    db: Arc<Database>,
    /// Owns its own `Arc<Database>`; dropping it joins the writer thread.
    writer: Writer,
    metrics: Arc<Metrics>,
    log: Log,
    embedder: Arc<dyn Embedder>,
    cfg: Config,
    path: PathBuf,
}

/// Cheaply clonable, `Send + Sync` handle to one memory file.
#[derive(Clone)]
pub struct SynapStore {
    inner: Arc<Inner>,
}

impl SynapStore {
    /// Open (or create) the memory file at `path`.
    pub fn open(path: impl AsRef<Path>, mut cfg: Config) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let log = Log(cfg.log.clone().map(Logger::open).transpose()?.map(Arc::new));
        let db = Arc::new(retry::open_database(&path, cfg.cache_bytes, &cfg.retry)?);
        // Resolve the embedder before taking the write lock: it may download.
        let rec = space::recorded(&db)?;
        let embedder = space::resolve(&rec, &cfg, &log)?;
        let (txn, _) = retry::begin_write(&db, &cfg.retry)?;
        tables::create_all(&txn)?;
        space::pin(&txn, &rec, &cfg, embedder.as_ref())?;
        {
            // Structural HNSW parameters are fixed at creation time.
            let mut meta = txn.open_table(META)?;
            match meta_get(&meta, keys::HNSW_M)? {
                Some(m) => {
                    cfg.hnsw.m = m as usize;
                    cfg.hnsw.m0 = 2 * m as usize;
                }
                None => {
                    meta.insert(keys::HNSW_M, cfg.hnsw.m as u64)?;
                }
            }
            match meta_get(&meta, keys::HNSW_EF_CONSTRUCTION)? {
                Some(ef) => cfg.hnsw.ef_construction = ef as usize,
                None => {
                    meta.insert(keys::HNSW_EF_CONSTRUCTION, cfg.hnsw.ef_construction as u64)?;
                }
            }
        }
        txn.commit()?;
        let metrics = Arc::new(Metrics::default());
        let writer = Writer::spawn(WriterCtx {
            db: db.clone(),
            hnsw: cfg.hnsw,
            retry: cfg.retry,
            metrics: metrics.clone(),
            log: log.clone(),
        })?;
        log.info("open", &[("path", path.display().to_string()), ("simd", crate::math::backend().into())]);
        Ok(SynapStore { inner: Arc::new(Inner { db, writer, metrics, log, embedder, cfg, path }) })
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }
    pub fn config(&self) -> &Config {
        &self.inner.cfg
    }
    pub fn metrics(&self) -> &Metrics {
        &self.inner.metrics
    }
    pub fn log(&self) -> &Log {
        &self.inner.log
    }
    /// The text embedder used for `memorize`/`recall` without vectors.
    pub fn embedder(&self) -> &dyn Embedder {
        self.inner.embedder.as_ref()
    }
    pub(crate) fn db(&self) -> &Database {
        &self.inner.db
    }

    /// Time an operation, record its latency and log the outcome.
    pub(crate) fn timed<T>(&self, op: Op, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let start = Instant::now();
        let r = f();
        let us = start.elapsed();
        self.inner.metrics.record(op, us);
        let mut fields = vec![("latency_us", us.as_micros().to_string())];
        match &r {
            Ok(_) => self.inner.log.info(op.label(), &fields),
            Err(e) => {
                Metrics::add(&self.inner.metrics.counters.errors, 1);
                fields.push(("error", e.to_string()));
                self.inner.log.warn(op.label(), &fields);
            }
        }
        r
    }

    pub(crate) fn write(&self, op: WriteOp) -> Result<WriteResult> {
        self.inner.writer.submit(op)
    }

    /// The store's vector space: `(dimension, embedding source)`.
    pub fn space(&self) -> Result<(Option<usize>, Option<String>)> {
        let txn = self.db().begin_read()?;
        let dim = meta_get(&txn.open_table(META)?, keys::DIM)?.map(|d| d as usize);
        let src = meta_str_get(&txn.open_table(META_STR)?, keys::EMBEDDING_SOURCE)?;
        Ok((dim, src))
    }

    fn prepare(&self, input: &MemorizeInput) -> Result<WriteOp> {
        let p = prepare::prepare_memorize(input, now_ms(), self.embedder(), self.inner.cfg.max_chunk_chars)?;
        Ok(WriteOp::Memorize(Box::new(p)))
    }

    /// Layer 1+2+3 write: chunk, embed (if needed), index, link entities,
    /// assert facts and append an episode.
    pub fn memorize(&self, input: MemorizeInput) -> Result<MemorizeOutcome> {
        self.timed(Op::Memorize, || {
            let op = self.prepare(&input)?;
            match self.write(op)? {
                WriteResult::Memorize(o) => Ok(o),
                other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
            }
        })
    }

    /// Memorize many items in a single transaction (bulk loading).
    pub fn memorize_batch(&self, inputs: &[MemorizeInput]) -> Result<Vec<MemorizeOutcome>> {
        self.timed(Op::Memorize, || {
            // Embedding is the expensive part: spread inputs over the cores.
            let ops =
                crate::embed::parallel_map(inputs, |i| self.prepare(i)).into_iter().collect::<Result<Vec<_>>>()?;
            match self.write(WriteOp::Batch(ops))? {
                WriteResult::Batch(rs) => Ok(rs
                    .into_iter()
                    .filter_map(|r| if let WriteResult::Memorize(o) = r { Some(o) } else { None })
                    .collect()),
                other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
            }
        })
    }

    /// Assert facts directly. `source` optionally links them to a chunk.
    pub fn add_facts(&self, facts: Vec<FactInput>, source: Option<u64>, ts: Option<u64>) -> Result<usize> {
        self.timed(Op::AddFacts, || {
            for f in &facts {
                crate::graph::validate_name("subject", &f.subject)?;
                crate::graph::validate_name("predicate", &f.predicate)?;
                crate::graph::validate_name("object", &f.object)?;
            }
            let op = WriteOp::Facts { facts, ts: ts.unwrap_or_else(now_ms), source: source.unwrap_or(0) };
            match self.write(op)? {
                WriteResult::Facts(n) => Ok(n),
                other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
            }
        })
    }

    /// Remove one fact. Returns whether it existed.
    pub fn retract_fact(&self, fact: FactInput) -> Result<bool> {
        self.timed(Op::AddFacts, || match self.write(WriteOp::Retract(fact))? {
            WriteResult::Retract(b) => Ok(b),
            other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
        })
    }

    /// Append events to the episodic log. Returns their ids.
    pub fn log_events(&self, events: &[EventInput]) -> Result<Vec<u64>> {
        self.timed(Op::LogEvent, || {
            let now = now_ms();
            let evs = events.iter().map(|e| prepare::prepare_event(e, now)).collect::<Result<Vec<_>>>()?;
            match self.write(WriteOp::Events(evs))? {
                WriteResult::Events(ids) => Ok(ids),
                other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
            }
        })
    }

    /// Reconnect HNSW nodes that became unreachable from the entry point
    /// (also runs automatically, amortized, during inserts).
    pub fn repair_index(&self) -> Result<crate::hnsw::repair::RepairOutcome> {
        let r = self.write(WriteOp::Repair)?;
        match r {
            WriteResult::Repair(o) => {
                self.inner.log.info(
                    "repair_index",
                    &[("unreachable", o.unreachable_before.to_string()), ("links", o.links_added.to_string())],
                );
                Ok(o)
            }
            other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
        }
    }

    /// Delete chunks by id. Returns how many existed.
    pub fn forget(&self, ids: Vec<u64>) -> Result<usize> {
        self.timed(Op::Forget, || match self.write(WriteOp::Forget(ids))? {
            WriteResult::Forget(n) => Ok(n),
            other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
        })
    }
}
