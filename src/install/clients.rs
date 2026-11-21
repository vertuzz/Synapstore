//! Where each agent keeps its MCP servers and skills (user scope).

use crate::paths::{app_config, home, which, xdg_config};
use serde_json::{Value, json};
use std::path::PathBuf;

/// How an MCP server is launched or reached.
#[derive(Debug, Clone, PartialEq)]
pub enum Server {
    Local { command: String, args: Vec<String> },
    Remote { url: String, token: Option<String> },
}

impl Server {
    fn headers(token: &Option<String>) -> Value {
        token.as_ref().map_or_else(|| json!({}), |t| json!({"Authorization": format!("Bearer {t}")}))
    }
}

/// JSON dialects of `mcpServers`-style entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Style {
    /// `{command, args}` / `{type: "http", url, headers}` (Claude Code, Cursor).
    Standard,
    /// Claude Desktop: local servers only.
    LocalOnly,
    Gemini,
    Windsurf,
    VsCode,
    OpenCode,
}

impl Style {
    /// The entry for `server`, or `None` when the client cannot use it.
    pub fn entry(self, server: &Server) -> Option<Value> {
        Some(match (self, server) {
            (Style::OpenCode, Server::Local { command, args }) => {
                let mut cmd = vec![command.clone()];
                cmd.extend(args.iter().cloned());
                json!({"type": "local", "command": cmd, "enabled": true})
            }
            (Style::VsCode, Server::Local { command, args }) => {
                json!({"type": "stdio", "command": command, "args": args})
            }
            (_, Server::Local { command, args }) => json!({"command": command, "args": args}),
            (Style::LocalOnly, Server::Remote { .. }) => return None,
            (Style::OpenCode, Server::Remote { url, token }) => {
                json!({"type": "remote", "url": url, "headers": Server::headers(token), "oauth": false, "enabled": true})
            }
            (Style::Gemini, Server::Remote { url, token }) => {
                json!({"httpUrl": url, "headers": Server::headers(token)})
            }
            (Style::Windsurf, Server::Remote { url, token }) => {
                json!({"serverUrl": url, "headers": Server::headers(token)})
            }
            (Style::VsCode | Style::Standard, Server::Remote { url, token }) => {
                json!({"type": "http", "url": url, "headers": Server::headers(token)})
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum Mcp {
    /// `claude mcp add` when the CLI is on PATH, else `~/.claude.json`.
    ClaudeCode,
    Json {
        file: PathBuf,
        path: &'static [&'static str],
        style: Style,
    },
    Toml {
        file: PathBuf,
    },
    Unsupported(&'static str),
}

#[derive(Debug, Clone)]
pub struct Client {
    pub id: &'static str,
    pub name: &'static str,
    pub mcp: Mcp,
    /// Directory that holds `<skill>/SKILL.md`, if the client reads skills.
    pub skills: Option<PathBuf>,
    pub detected: bool,
}

fn exists(p: PathBuf) -> bool {
    p.exists()
}

fn codex_home() -> PathBuf {
    std::env::var_os("CODEX_HOME").filter(|v| !v.is_empty()).map_or_else(|| home().join(".codex"), PathBuf::from)
}

fn opencode_file() -> PathBuf {
    let dir = xdg_config().join("opencode");
    let jsonc = dir.join("opencode.jsonc");
    if jsonc.is_file() { jsonc } else { dir.join("opencode.json") }
}

/// Every supported client, with detection run against this machine.
pub fn all() -> Vec<Client> {
    let h = home();
    let on_path = |n: &str| which(n).is_some();
    let servers: &'static [&'static str] = &["mcpServers"];
    vec![
        Client {
            id: "claude-code",
            name: "Claude Code",
            mcp: Mcp::ClaudeCode,
            skills: Some(h.join(".claude").join("skills")),
            detected: on_path("claude") || exists(h.join(".claude")),
        },
        Client {
            id: "codex",
            name: "Codex CLI",
            mcp: Mcp::Toml { file: codex_home().join("config.toml") },
            skills: Some(h.join(".agents").join("skills")),
            detected: on_path("codex") || exists(codex_home()),
        },
        Client {
            id: "opencode",
            name: "OpenCode",
            mcp: Mcp::Json { file: opencode_file(), path: &["mcp"], style: Style::OpenCode },
            skills: Some(xdg_config().join("opencode").join("skills")),
            detected: on_path("opencode") || exists(xdg_config().join("opencode")),
        },
        Client {
            id: "gemini",
            name: "Gemini CLI",
            mcp: Mcp::Json { file: h.join(".gemini").join("settings.json"), path: servers, style: Style::Gemini },
            skills: Some(h.join(".gemini").join("skills")),
            detected: on_path("gemini") || exists(h.join(".gemini")),
        },
        Client {
            id: "cursor",
            name: "Cursor",
            mcp: Mcp::Json { file: h.join(".cursor").join("mcp.json"), path: servers, style: Style::Standard },
            skills: Some(h.join(".cursor").join("skills")),
            detected: on_path("cursor") || exists(h.join(".cursor")),
        },
        Client {
            id: "vscode",
            name: "VS Code (Copilot)",
            mcp: Mcp::Json {
                file: app_config().join("Code").join("User").join("mcp.json"),
                path: &["servers"],
                style: Style::VsCode,
            },
            skills: Some(h.join(".copilot").join("skills")),
            detected: on_path("code") || exists(app_config().join("Code").join("User")),
        },
        Client {
            id: "windsurf",
            name: "Windsurf",
            mcp: Mcp::Json {
                file: h.join(".codeium").join("windsurf").join("mcp_config.json"),
                path: servers,
                style: Style::Windsurf,
            },
            skills: Some(h.join(".codeium").join("windsurf").join("skills")),
            detected: exists(h.join(".codeium").join("windsurf")),
        },
        Client {
            id: "claude-desktop",
            name: "Claude Desktop",
            mcp: Mcp::Json {
                file: app_config().join("Claude").join("claude_desktop_config.json"),
                path: servers,
                style: Style::LocalOnly,
            },
            skills: None,
            detected: exists(app_config().join("Claude")),
        },
        Client {
            id: "pi",
            name: "pi",
            mcp: Mcp::Unsupported("pi has no MCP; it uses the skill and the `synapstore` CLI"),
            skills: Some(h.join(".agents").join("skills")),
            detected: on_path("pi") || exists(h.join(".pi")),
        },
    ]
}

/// `~/.claude.json`, Claude Code's own state file (fallback when `claude` is not on PATH).
pub fn claude_json() -> PathBuf {
    home().join(".claude.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_follow_each_dialect() {
        let local = Server::Local { command: "/s".into(), args: vec!["mcp".into()] };
        let remote = Server::Remote { url: "https://m/mcp".into(), token: Some("t".into()) };
        assert_eq!(Style::Standard.entry(&local).unwrap(), json!({"command": "/s", "args": ["mcp"]}));
        assert_eq!(Style::OpenCode.entry(&local).unwrap()["command"], json!(["/s", "mcp"]));
        assert_eq!(Style::Gemini.entry(&remote).unwrap()["httpUrl"], "https://m/mcp");
        assert_eq!(Style::Windsurf.entry(&remote).unwrap()["headers"]["Authorization"], "Bearer t");
        assert_eq!(Style::VsCode.entry(&remote).unwrap()["type"], "http");
        assert!(Style::LocalOnly.entry(&remote).is_none());
        let ids: Vec<&str> = all().iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), 9);
        assert!(ids.contains(&"pi") && ids.contains(&"codex"));
    }
}
