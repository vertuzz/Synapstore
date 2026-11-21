//! MCP sessions (`Mcp-Session-Id`) and failed-authentication throttling.

use crate::crypto::{hex, random_bytes};
use crate::mcp::Server;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const SESSION_IDLE: Duration = Duration::from_secs(3600);
const MAX_SESSIONS: usize = 4096;

pub struct Session {
    pub server: Mutex<Server>,
    /// Hash of the token that created the session (`None` without auth).
    pub token: Option<[u8; 32]>,
    last_seen: Mutex<Instant>,
}

#[derive(Default)]
pub struct Sessions {
    map: Mutex<HashMap<String, Arc<Session>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Sessions {
    /// Register a new session; evicts idle (or, if full, the oldest) ones.
    pub fn create(&self, server: Server, token: Option<[u8; 32]>) -> Option<String> {
        let mut raw = [0u8; 16];
        random_bytes(&mut raw).ok()?;
        let id = hex(&raw);
        let mut map = lock(&self.map);
        let now = Instant::now();
        map.retain(|_, s| now.duration_since(*lock(&s.last_seen)) < SESSION_IDLE);
        if map.len() >= MAX_SESSIONS {
            let oldest = map.iter().min_by_key(|(_, s)| *lock(&s.last_seen)).map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                map.remove(&k);
            }
        }
        map.insert(id.clone(), Arc::new(Session { server: Mutex::new(server), token, last_seen: Mutex::new(now) }));
        Some(id)
    }

    /// The live session `id`, if it belongs to `token`.
    pub fn get(&self, id: &str, token: Option<[u8; 32]>) -> Option<Arc<Session>> {
        let map = lock(&self.map);
        let s = map.get(id)?;
        let mut seen = lock(&s.last_seen);
        if seen.elapsed() >= SESSION_IDLE || s.token != token {
            return None;
        }
        *seen = Instant::now();
        drop(seen);
        Some(s.clone())
    }

    pub fn remove(&self, id: &str, token: Option<[u8; 32]>) -> bool {
        let mut map = lock(&self.map);
        if map.get(id).is_some_and(|s| s.token == token) {
            map.remove(id);
            return true;
        }
        false
    }

    pub fn len(&self) -> usize {
        lock(&self.map).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Blocks an address after too many failed authentications in a window.
pub struct Throttle {
    map: Mutex<HashMap<IpAddr, (u32, Instant)>>,
    pub max_failures: u32,
    pub window: Duration,
}

impl Default for Throttle {
    fn default() -> Self {
        Throttle { map: Mutex::default(), max_failures: 20, window: Duration::from_secs(300) }
    }
}

impl Throttle {
    /// Seconds until `ip` may retry, if it is currently blocked.
    pub fn blocked(&self, ip: IpAddr) -> Option<u64> {
        let map = lock(&self.map);
        let (n, start) = map.get(&ip)?;
        let left = self.window.checked_sub(start.elapsed())?;
        (*n >= self.max_failures).then(|| left.as_secs().max(1))
    }

    pub fn fail(&self, ip: IpAddr) {
        let mut map = lock(&self.map);
        if map.len() > 100_000 {
            map.retain(|_, (_, t)| t.elapsed() < self.window);
        }
        let e = map.entry(ip).or_insert((0, Instant::now()));
        if e.1.elapsed() >= self.window {
            *e = (0, Instant::now());
        }
        e.0 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_blocks_after_limit() {
        let t = Throttle { max_failures: 3, ..Throttle::default() };
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..2 {
            t.fail(ip);
        }
        assert!(t.blocked(ip).is_none());
        t.fail(ip);
        assert!(t.blocked(ip).is_some_and(|s| s >= 1));
        assert!(t.blocked("10.0.0.2".parse().unwrap()).is_none());
    }
}
