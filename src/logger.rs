//! Minimal file logger with size-based rotation (PRD §9).
//!
//! Lines are `logfmt`: `<rfc3339> <LEVEL> <event> key=value …`, which is easy
//! for humans to read and for agents to parse. When the active file would
//! exceed `max_bytes`, `log` → `log.1` → … → `log.<keep>` (oldest dropped).
//! The logger is an explicit value passed to whoever needs it — no globals.

use crate::timefmt::{now_ms, rfc3339};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
        }
    }

    pub fn parse(s: &str) -> Option<Level> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Some(Level::Error),
            "warn" | "warning" => Some(Level::Warn),
            "info" => Some(Level::Info),
            "debug" => Some(Level::Debug),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogConfig {
    pub path: PathBuf,
    pub max_bytes: u64,
    pub keep: usize,
    pub level: Level,
}

impl LogConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        LogConfig { path: path.into(), max_bytes: 10 * 1024 * 1024, keep: 5, level: Level::Info }
    }
}

struct State {
    file: File,
    size: u64,
}

pub struct Logger {
    cfg: LogConfig,
    state: Mutex<State>,
}

/// Path of the `n`-th rotated file (`n >= 1`).
pub fn rotated_path(path: &Path, n: usize) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{n}"));
    PathBuf::from(s)
}

fn quote(v: &str) -> String {
    if !v.is_empty() && !v.contains(|c: char| c.is_whitespace() || c == '"' || c == '=') {
        return v.to_string();
    }
    format!("{v:?}")
}

/// Render one log line (pure).
pub fn format_line(ts_ms: u64, level: Level, event: &str, fields: &[(&str, String)]) -> String {
    let mut line = format!("{} {} {}", rfc3339(ts_ms), level.as_str(), event);
    for (k, v) in fields {
        line.push(' ');
        line.push_str(k);
        line.push('=');
        line.push_str(&quote(v));
    }
    line.push('\n');
    line
}

fn open_append(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

impl Logger {
    pub fn open(cfg: LogConfig) -> io::Result<Self> {
        if let Some(dir) = cfg.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let file = open_append(&cfg.path)?;
        let size = file.metadata()?.len();
        Ok(Logger { cfg, state: Mutex::new(State { file, size }) })
    }

    pub fn path(&self) -> &Path {
        &self.cfg.path
    }

    fn rotate(&self, st: &mut State) -> io::Result<()> {
        st.file.flush()?;
        let keep = self.cfg.keep.max(1);
        let _ = std::fs::remove_file(rotated_path(&self.cfg.path, keep));
        for n in (1..keep).rev() {
            let from = rotated_path(&self.cfg.path, n);
            if from.exists() {
                std::fs::rename(&from, rotated_path(&self.cfg.path, n + 1))?;
            }
        }
        std::fs::rename(&self.cfg.path, rotated_path(&self.cfg.path, 1))?;
        st.file = open_append(&self.cfg.path)?;
        st.size = 0;
        Ok(())
    }

    /// Append one event. Logging failures are swallowed: memory operations
    /// must never fail because the log disk is full.
    pub fn log(&self, level: Level, event: &str, fields: &[(&str, String)]) {
        if level > self.cfg.level {
            return;
        }
        let line = format_line(now_ms(), level, event, fields);
        let Ok(mut st) = self.state.lock() else {
            return;
        };
        if st.size > 0 && st.size + line.len() as u64 > self.cfg.max_bytes {
            let _ = self.rotate(&mut st);
        }
        if st.file.write_all(line.as_bytes()).is_ok() {
            st.size += line.len() as u64;
        }
    }
}

/// Optional logger handle; every method is a no-op when logging is off.
#[derive(Clone, Default)]
pub struct Log(pub Option<Arc<Logger>>);

impl Log {
    pub fn emit(&self, level: Level, event: &str, fields: &[(&str, String)]) {
        if let Some(l) = &self.0 {
            l.log(level, event, fields);
        }
    }
    pub fn info(&self, event: &str, fields: &[(&str, String)]) {
        self.emit(Level::Info, event, fields);
    }
    pub fn warn(&self, event: &str, fields: &[(&str, String)]) {
        self.emit(Level::Warn, event, fields);
    }
    pub fn error(&self, event: &str, fields: &[(&str, String)]) {
        self.emit(Level::Error, event, fields);
    }
    pub fn debug(&self, event: &str, fields: &[(&str, String)]) {
        self.emit(Level::Debug, event, fields);
    }
}
