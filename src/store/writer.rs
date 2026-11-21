//! The single writer thread (PRD §6.1). Every mutation is routed through an
//! `std::sync::mpsc` channel to one thread that owns all write
//! transactions — no mutex guards the database. Requests that queue up
//! while a transaction is running are applied together in one transaction
//! (group commit); if any of them fails, the group is rolled back and each
//! request is replayed in its own transaction so errors stay isolated.

use super::apply::{ApplyCtx, WriteOp, WriteResult, apply};
use crate::error::{Error, Result};
use crate::hnsw::HnswParams;
use crate::logger::Log;
use crate::metrics::{Metrics, Op};
use crate::retry::{RetryPolicy, begin_write};
use redb::Database;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

/// Max requests merged into one transaction.
pub const MAX_GROUP: usize = 64;

pub struct Request {
    pub op: WriteOp,
    pub reply: Sender<Result<WriteResult>>,
}

/// Everything the writer thread needs, owned by it.
pub struct WriterCtx {
    pub db: Arc<Database>,
    pub hnsw: HnswParams,
    pub retry: RetryPolicy,
    pub metrics: Arc<Metrics>,
    pub log: Log,
}

pub struct Writer {
    tx: Option<Sender<Request>>,
    handle: Option<JoinHandle<()>>,
}

impl Writer {
    pub fn spawn(ctx: WriterCtx) -> Result<Self> {
        let (tx, rx) = channel::<Request>();
        let handle = std::thread::Builder::new().name("synapstore-writer".into()).spawn(move || run(ctx, rx))?;
        Ok(Writer { tx: Some(tx), handle: Some(handle) })
    }

    /// Submit an operation and block until it is durably committed.
    pub fn submit(&self, op: WriteOp) -> Result<WriteResult> {
        let (reply, rx) = channel();
        let tx = self.tx.as_ref().ok_or_else(|| Error::WriterGone("closed".into()))?;
        tx.send(Request { op, reply }).map_err(|_| Error::WriterGone("writer thread stopped".into()))?;
        rx.recv().map_err(|_| Error::WriterGone("writer dropped the request".into()))?
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(ctx: WriterCtx, rx: Receiver<Request>) {
    while let Ok(first) = rx.recv() {
        let mut batch = vec![first];
        while batch.len() < MAX_GROUP {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }
        if batch.len() == 1 {
            let r = &batch[0];
            let _ = r.reply.send(transact(&ctx, std::slice::from_ref(&r.op)).map(|mut v| v.remove(0)));
            continue;
        }
        let ops: Vec<WriteOp> = batch.iter().map(|r| r.op.clone()).collect();
        match transact(&ctx, &ops) {
            Ok(results) => {
                Metrics::add(&ctx.metrics.counters.group_commits, 1);
                Metrics::add(&ctx.metrics.counters.grouped_ops, batch.len() as u64);
                for (r, res) in batch.iter().zip(results) {
                    let _ = r.reply.send(Ok(res));
                }
            }
            Err(_) => {
                for r in &batch {
                    let res = transact(&ctx, std::slice::from_ref(&r.op)).map(|mut v| v.remove(0));
                    let _ = r.reply.send(res);
                }
            }
        }
    }
}

/// Apply `ops` in one write transaction; commit only if all succeed.
fn transact(ctx: &WriterCtx, ops: &[WriteOp]) -> Result<Vec<WriteResult>> {
    let started = Instant::now();
    let counters = &ctx.metrics.counters;
    let (txn, retries) = match begin_write(&ctx.db, &ctx.retry) {
        Ok(v) => v,
        Err(e) => {
            Metrics::add(&counters.busy_failures, u64::from(matches!(e, Error::Busy(_))));
            Metrics::add(&counters.errors, 1);
            ctx.log.error("write_txn", &[("error", e.to_string())]);
            return Err(e);
        }
    };
    Metrics::add(&counters.write_retries, u64::from(retries));
    let cache = crate::hnsw::cache::WriteCache::default();
    let actx = ApplyCtx { hnsw: &ctx.hnsw, metrics: &ctx.metrics, cache: &cache };
    let mut results = Vec::with_capacity(ops.len());
    for op in ops {
        match apply(&txn, op, &actx) {
            Ok(r) => results.push(r),
            Err(e) => {
                let _ = txn.abort();
                if ops.len() == 1 {
                    Metrics::add(&counters.errors, 1);
                    ctx.log.warn("write_rejected", &[("kind", e.kind().into()), ("error", e.to_string())]);
                }
                return Err(e);
            }
        }
    }
    if let Err(e) = txn.open_table(crate::tables::HNSW_NODES).map_err(Into::into).and_then(|mut t| cache.flush(&mut t))
    {
        let _ = txn.abort();
        return Err(e);
    }
    txn.commit()?;
    let elapsed = started.elapsed();
    ctx.metrics.record(Op::WriteTxn, elapsed);
    ctx.log.debug(
        "write_txn",
        &[
            ("ops", ops.len().to_string()),
            ("retries", retries.to_string()),
            ("latency_us", elapsed.as_micros().to_string()),
        ],
    );
    Ok(results)
}
