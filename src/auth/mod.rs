//! API tokens for the HTTP transport (RFC 6750 bearer tokens).
//!
//! * Secrets are 256-bit random values, shown once at creation and stored
//!   only as SHA-256 hashes, so a leaked memory file leaks no credentials.
//! * The `syn_` prefix makes them recognizable to secret scanners.
//! * Each token has a scope (read ⊂ write ⊂ admin), an optional expiry and
//!   can be revoked at any time; revocation is effective immediately because
//!   every request is checked against the file.

pub mod ops;

pub use ops::{AuthOp, AuthResult};

use crate::crypto::{base64url, hex, random_bytes, sha256};
use crate::error::{Error, Result};
use serde::Serialize;

pub const TOKEN_PREFIX: &str = "syn_";

/// What a token may do. Ordered: a higher scope includes the lower ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Read = 0,
    Write = 1,
    Admin = 2,
}

impl Scope {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read" | "ro" | "read-only" => Ok(Scope::Read),
            "write" | "rw" | "read-write" => Ok(Scope::Write),
            "admin" => Ok(Scope::Admin),
            other => Err(Error::InvalidInput(format!("unknown scope `{other}` (read, write or admin)"))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Read => "read",
            Scope::Write => "write",
            Scope::Admin => "admin",
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Scope::Read,
            1 => Scope::Write,
            _ => Scope::Admin,
        }
    }
}

/// The scope an MCP tool requires.
pub fn tool_scope(tool: &str) -> Scope {
    match tool {
        "recall" | "graph_query" | "graph_match" | "recent_events" | "explain" | "stats" => Scope::Read,
        "repair_index" => Scope::Admin,
        _ => Scope::Write,
    }
}

/// Public view of a token (never includes the secret).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenInfo {
    /// Short public identifier derived from the hash (safe to log).
    pub id: String,
    pub name: String,
    pub scope: Scope,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub last_used_at: Option<u64>,
}

impl TokenInfo {
    pub fn expired(&self, now: u64) -> bool {
        self.expires_at.is_some_and(|e| e <= now)
    }
}

/// SHA-256 of a presented secret: the table key.
pub fn hash_secret(secret: &str) -> [u8; 32] {
    sha256(secret.as_bytes())
}

/// Public id of a token: the first 6 bytes of its hash, hex.
pub fn token_id(hash: &[u8]) -> String {
    hex(&hash[..6.min(hash.len())])
}

/// A fresh secret and its hash.
pub fn generate() -> Result<(String, [u8; 32])> {
    let mut raw = [0u8; 32];
    random_bytes(&mut raw)?;
    let secret = format!("{TOKEN_PREFIX}{}", base64url(&raw));
    let hash = hash_secret(&secret);
    Ok((secret, hash))
}

/// Cheap shape check before touching the database.
pub fn plausible(secret: &str) -> bool {
    secret.len() == TOKEN_PREFIX.len() + 43
        && secret.starts_with(TOKEN_PREFIX)
        && secret[TOKEN_PREFIX.len()..].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_random_and_well_formed() {
        let (a, ha) = generate().unwrap();
        let (b, _) = generate().unwrap();
        assert_ne!(a, b);
        assert!(plausible(&a) && plausible(&b));
        assert_eq!(hash_secret(&a), ha);
        assert_eq!(token_id(&ha).len(), 12);
        assert!(!plausible("syn_short") && !plausible(&a.replace("syn_", "xyz_")));
    }

    #[test]
    fn scopes_are_ordered() {
        assert!(Scope::Admin > Scope::Write && Scope::Write > Scope::Read);
        assert_eq!(Scope::parse("RW").unwrap(), Scope::Write);
        assert!(Scope::parse("root").is_err());
        assert_eq!(tool_scope("recall"), Scope::Read);
        assert_eq!(tool_scope("memorize"), Scope::Write);
        assert_eq!(tool_scope("forget"), Scope::Write);
        assert_eq!(tool_scope("repair_index"), Scope::Admin);
        for s in [Scope::Read, Scope::Write, Scope::Admin] {
            assert_eq!(Scope::from_u8(s as u8), s);
        }
    }
}
