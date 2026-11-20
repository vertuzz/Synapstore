//! Exponential backoff around file-lock contention (PRD §6.2).
//!
//! The database is opened in redb's `MultiWriter` mode so several agent
//! processes can share one file. A write transaction that cannot start
//! because another *process* holds the writer lock is retried up to
//! `max_retries` times with exponentially growing sleeps.

use crate::error::{Error, Result};
use redb::{ConcurrencyMode, Database, DatabaseError, TransactionError, WriteTransaction};
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy { max_retries: 5, base_delay: Duration::from_millis(20), max_delay: Duration::from_secs(2) }
    }
}

/// Delay before retry number `attempt` (0-based): `base · 2^attempt`, capped.
pub fn backoff_delay(policy: &RetryPolicy, attempt: u32) -> Duration {
    let factor = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
    policy.base_delay.checked_mul(factor).unwrap_or(policy.max_delay).min(policy.max_delay)
}

/// Run `f` until it yields `Some`, sleeping with backoff on `None`.
/// Returns the value and the number of retries that were needed.
pub fn with_backoff<T>(policy: &RetryPolicy, mut f: impl FnMut() -> Result<Option<T>>) -> Result<(T, u32)> {
    for attempt in 0..=policy.max_retries {
        if let Some(v) = f()? {
            return Ok((v, attempt));
        }
        if attempt < policy.max_retries {
            std::thread::sleep(backoff_delay(policy, attempt));
        }
    }
    Err(Error::Busy(policy.max_retries))
}

/// Begin a write transaction, retrying while another process writes.
pub fn begin_write(db: &Database, policy: &RetryPolicy) -> Result<(WriteTransaction, u32)> {
    with_backoff(policy, || match db.try_begin_write() {
        Ok(txn) => Ok(Some(txn)),
        Err(TransactionError::WouldBlock) => Ok(None),
        Err(e) => Err(e.into()),
    })
}

fn open_with(path: &Path, cache_bytes: usize, mode: ConcurrencyMode) -> Result<Option<Database>> {
    let mut builder = Database::builder();
    builder.set_cache_size(cache_bytes);
    builder.set_concurrency_mode(mode);
    match builder.create(path) {
        Ok(db) => Ok(Some(db)),
        Err(DatabaseError::DatabaseAlreadyOpen) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Open (or create) the database file, retrying while it is locked.
/// Falls back to `ExclusiveWriter` on platforms without byte-range locks.
pub fn open_database(path: &Path, cache_bytes: usize, policy: &RetryPolicy) -> Result<Database> {
    let multi = with_backoff(policy, || open_with(path, cache_bytes, ConcurrencyMode::MultiWriter));
    match multi {
        Ok((db, _)) => Ok(db),
        Err(Error::Busy(n)) => Err(Error::Busy(n)),
        Err(_) => Ok(with_backoff(policy, || open_with(path, cache_bytes, ConcurrencyMode::ExclusiveWriter))?.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_grow_exponentially_and_cap() {
        let p = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
        };
        let d: Vec<u128> = (0..6).map(|a| backoff_delay(&p, a).as_millis()).collect();
        assert_eq!(d, vec![10, 20, 40, 80, 100, 100]);
        assert_eq!(backoff_delay(&p, 200), Duration::from_millis(100));
    }

    #[test]
    fn retries_then_succeeds_or_gives_up() {
        let p =
            RetryPolicy { max_retries: 5, base_delay: Duration::from_micros(1), max_delay: Duration::from_micros(10) };
        let mut calls = 0;
        let (v, retries) = with_backoff(&p, || {
            calls += 1;
            Ok((calls == 3).then_some(42))
        })
        .unwrap();
        assert_eq!((v, retries), (42, 2));
        let mut calls = 0;
        let r: Result<(u8, u32)> = with_backoff(&p, || {
            calls += 1;
            Ok(None)
        });
        assert!(matches!(r, Err(Error::Busy(5))));
        assert_eq!(calls, 6);
    }
}
