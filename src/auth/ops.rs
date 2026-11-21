//! Token records in the `auth_tokens` table: writes (applied by the store's
//! writer thread) and reads (on MVCC snapshots).

use super::{Scope, TokenInfo, token_id};
use crate::archive::{decode, encode, view};
use crate::error::{Error, Result};
use crate::model::ApiToken;
use crate::tables::AUTH_TOKENS;
use redb::{ReadableTable, WriteTransaction};

#[derive(Debug, Clone)]
pub enum AuthOp {
    Create {
        hash: [u8; 32],
        record: ApiToken,
    },
    /// Revoke by public id or by name.
    Revoke {
        target: String,
    },
    Touch {
        hash: [u8; 32],
        at: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AuthResult {
    Created,
    Revoked(Vec<TokenInfo>),
    Touched,
}

pub fn info(hash: &[u8], t: &ApiToken) -> TokenInfo {
    TokenInfo {
        id: token_id(hash),
        name: t.name.clone(),
        scope: Scope::from_u8(t.scope),
        created_at: t.created_at,
        expires_at: (t.expires_at != 0).then_some(t.expires_at),
        last_used_at: (t.last_used_at != 0).then_some(t.last_used_at),
    }
}

/// Every token, sorted by creation time.
pub fn list<T: ReadableTable<&'static [u8], &'static [u8]>>(table: &T) -> Result<Vec<TokenInfo>> {
    let mut out = Vec::new();
    for item in table.iter()? {
        let (k, v) = item?;
        out.push(info(k.value(), &decode::<ApiToken>(v.value())?));
    }
    out.sort_by_key(|t| (t.created_at, t.id.clone()));
    Ok(out)
}

/// Look a token up by the hash of its secret.
pub fn lookup<T: ReadableTable<&'static [u8], &'static [u8]>>(table: &T, hash: &[u8; 32]) -> Result<Option<TokenInfo>> {
    let Some(g) = table.get(hash.as_slice())? else { return Ok(None) };
    let t = view::<ApiToken>(g.value())?;
    let record = ApiToken {
        name: t.name.as_str().to_string(),
        scope: t.scope,
        created_at: t.created_at.to_native(),
        expires_at: t.expires_at.to_native(),
        last_used_at: t.last_used_at.to_native(),
    };
    Ok(Some(info(hash, &record)))
}

pub fn apply(txn: &WriteTransaction, op: &AuthOp) -> Result<AuthResult> {
    let mut table = txn.open_table(AUTH_TOKENS)?;
    match op {
        AuthOp::Create { hash, record } => {
            if list(&table)?.iter().any(|t| t.name == record.name) {
                return Err(Error::InvalidInput(format!("a token named `{}` already exists", record.name)));
            }
            table.insert(hash.as_slice(), encode(record)?.as_slice())?;
            Ok(AuthResult::Created)
        }
        AuthOp::Revoke { target } => {
            let mut hit = Vec::new();
            for item in table.iter()? {
                let (k, v) = item?;
                let i = info(k.value(), &decode::<ApiToken>(v.value())?);
                if i.id == *target || i.name == *target {
                    hit.push((k.value().to_vec(), i));
                }
            }
            for (k, _) in &hit {
                table.remove(k.as_slice())?;
            }
            Ok(AuthResult::Revoked(hit.into_iter().map(|(_, i)| i).collect()))
        }
        AuthOp::Touch { hash, at } => {
            let current = table.get(hash.as_slice())?.map(|g| decode::<ApiToken>(g.value())).transpose()?;
            if let Some(mut t) = current {
                t.last_used_at = *at;
                table.insert(hash.as_slice(), encode(&t)?.as_slice())?;
            }
            Ok(AuthResult::Touched)
        }
    }
}
