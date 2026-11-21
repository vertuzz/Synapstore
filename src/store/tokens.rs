//! API-token administration and verification on a [`SynapStore`].

use super::SynapStore;
use super::apply::{WriteOp, WriteResult};
use crate::auth::{self, AuthOp, AuthResult, Scope, TokenInfo};
use crate::error::{Error, Result};
use crate::model::ApiToken;
use crate::tables::AUTH_TOKENS;
use crate::timefmt::now_ms;
use redb::ReadableDatabase;

const DAY_MS: u64 = 86_400_000;

impl SynapStore {
    fn auth_write(&self, op: AuthOp) -> Result<AuthResult> {
        match self.write(WriteOp::Auth(op))? {
            WriteResult::Auth(r) => Ok(r),
            other => Err(Error::Storage(format!("unexpected writer result {other:?}"))),
        }
    }

    /// Create a token. Returns the secret (shown once) and its record.
    pub fn create_token(&self, name: &str, scope: Scope, expires_days: Option<u64>) -> Result<(String, TokenInfo)> {
        let name = name.trim();
        if name.is_empty() || name.len() > 64 || name.chars().any(char::is_control) {
            return Err(Error::InvalidInput("token name must be 1-64 printable characters".into()));
        }
        let (secret, hash) = auth::generate()?;
        let now = now_ms();
        let record = ApiToken {
            name: name.into(),
            scope: scope as u8,
            created_at: now,
            expires_at: expires_days.map_or(0, |d| now.saturating_add(d.saturating_mul(DAY_MS))),
            last_used_at: 0,
        };
        let info = auth::ops::info(&hash, &record);
        self.auth_write(AuthOp::Create { hash, record })?;
        self.log().info("token_create", &[("id", info.id.clone()), ("name", info.name.clone())]);
        Ok((secret, info))
    }

    pub fn list_tokens(&self) -> Result<Vec<TokenInfo>> {
        let txn = self.db().begin_read()?;
        auth::ops::list(&txn.open_table(AUTH_TOKENS)?)
    }

    /// Revoke every token whose id or name is `target`.
    pub fn revoke_token(&self, target: &str) -> Result<Vec<TokenInfo>> {
        match self.auth_write(AuthOp::Revoke { target: target.trim().into() })? {
            AuthResult::Revoked(v) => {
                for t in &v {
                    self.log().info("token_revoke", &[("id", t.id.clone()), ("name", t.name.clone())]);
                }
                Ok(v)
            }
            other => Err(Error::Storage(format!("unexpected auth result {other:?}"))),
        }
    }

    /// Resolve a presented bearer secret to a live (unexpired) token.
    pub fn authenticate(&self, secret: &str) -> Result<Option<(TokenInfo, [u8; 32])>> {
        if !auth::plausible(secret) {
            return Ok(None);
        }
        let hash = auth::hash_secret(secret);
        let txn = self.db().begin_read()?;
        let found = auth::ops::lookup(&txn.open_table(AUTH_TOKENS)?, &hash)?;
        Ok(found.filter(|t| !t.expired(now_ms())).map(|t| (t, hash)))
    }

    /// Record that a token was used (callers throttle this).
    pub fn touch_token(&self, hash: [u8; 32]) -> Result<()> {
        self.auth_write(AuthOp::Touch { hash, at: now_ms() }).map(drop)
    }
}
