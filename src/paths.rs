//! Well-known locations: the default store and per-user config directories.

use std::path::PathBuf;

fn env(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The user's home directory (`HOME`, else `USERPROFILE`).
pub fn home() -> PathBuf {
    env("HOME").or_else(|| env("USERPROFILE")).unwrap_or_else(|| PathBuf::from("."))
}

/// The store used when no path is given: `SYNAPSTORE_DB`, else `~/.synapstore/memory.redb`.
pub fn default_db() -> PathBuf {
    env("SYNAPSTORE_DB").unwrap_or_else(|| home().join(".synapstore").join("memory.redb"))
}

/// Per-user application config directory (`~/.config`, `~/Library/Application Support`, `%APPDATA%`).
pub fn app_config() -> PathBuf {
    if cfg!(windows) {
        env("APPDATA").unwrap_or_else(|| home().join("AppData").join("Roaming"))
    } else if cfg!(target_os = "macos") {
        home().join("Library").join("Application Support")
    } else {
        env("XDG_CONFIG_HOME").unwrap_or_else(|| home().join(".config"))
    }
}

/// `~/.config` on every platform (tools such as OpenCode use it on macOS too).
pub fn xdg_config() -> PathBuf {
    env("XDG_CONFIG_HOME").unwrap_or_else(|| home().join(".config"))
}

/// Find `name` on `PATH` (with Windows executable extensions).
pub fn which(name: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) { &[".exe", ".cmd", ".bat", ""] } else { &[""] };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .find_map(|dir| exts.iter().map(|e| dir.join(format!("{name}{e}"))).find(|p| p.is_file()))
}
