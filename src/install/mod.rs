//! `synapstore install`: register the MCP server and the memory skill with
//! the agents on this machine. Edits keep the user's formatting, back the
//! file up first (`<file>.synapstore.bak`) and are undone by `uninstall`.

pub mod clients;
pub mod jsonc;
pub mod toml;

use crate::paths::{home, which};
use clients::{Client, Mcp, Server, Style};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The agent skill, in the Agent Skills format (`SKILL.md`).
pub const SKILL: &str = include_str!("../../skills/synapstore/SKILL.md");
pub const NAME: &str = "synapstore";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Part {
    Mcp,
    Skill,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Options {
    /// Client ids; empty = the ones found on this machine, `all` = every one.
    pub clients: Vec<String>,
    pub only: Option<Part>,
    pub url: Option<String>,
    pub dry_run: bool,
    pub uninstall: bool,
}

/// The clients `opts` asks for.
pub fn select(opts: &Options) -> Result<Vec<Client>, String> {
    let all = clients::all();
    if opts.clients.is_empty() {
        return Ok(all.into_iter().filter(|c| c.detected).collect());
    }
    if opts.clients.iter().any(|c| c == "all") {
        return Ok(all);
    }
    let mut out = Vec::new();
    for id in &opts.clients {
        let c = all.iter().find(|c| c.id == id || (id == "claude" && c.id == "claude-code"));
        let ids: Vec<&str> = all.iter().map(|c| c.id).collect();
        out.push(c.cloned().ok_or_else(|| format!("unknown client `{id}` (known: {}, all)", ids.join(", ")))?);
    }
    Ok(out)
}

/// `~` instead of the home directory, for messages.
pub fn tilde(p: &Path) -> String {
    let h = home();
    p.strip_prefix(&h)
        .map_or_else(|_| p.display().to_string(), |r| format!("~{}{}", std::path::MAIN_SEPARATOR, r.display()))
}

fn write(path: &Path, text: &str, dry: bool) -> Result<(), String> {
    if dry {
        return Ok(());
    }
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(err)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".synapstore.tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, text).map_err(err)?;
    if path.is_file() {
        let mut bak = path.as_os_str().to_owned();
        bak.push(".synapstore.bak");
        std::fs::copy(path, PathBuf::from(bak)).map_err(err)?;
        // Keep the original's permissions (config files may be private).
        let perms = std::fs::metadata(path).map_err(err)?.permissions();
        std::fs::set_permissions(&tmp, perms).map_err(err)?;
    }
    std::fs::rename(&tmp, path).map_err(err)
}

fn read(path: &Path) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn claude_cli(server: &Server, uninstall: bool, dry: bool) -> Option<Result<String, String>> {
    let bin = which("claude")?;
    let run = |args: &[String]| -> Result<(), String> {
        let out = Command::new(&bin).args(args).output().map_err(|e| format!("claude: {e}"))?;
        if out.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&out.stderr).trim().to_string()) }
    };
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let remove = s(&["mcp", "remove", "--scope", "user", NAME]);
    if uninstall {
        return Some(if dry {
            Ok("would run `claude mcp remove`".into())
        } else {
            run(&remove).map(|_| "removed".into())
        });
    }
    let mut add = s(&["mcp", "add", "--scope", "user"]);
    match server {
        Server::Local { command, args } => {
            add.extend([NAME.to_string(), "--".into(), command.clone()]);
            add.extend(args.iter().cloned());
        }
        Server::Remote { url, token } => {
            add.extend(s(&["--transport", "http", NAME, url]));
            if let Some(t) = token {
                add.extend(["--header".into(), format!("Authorization: Bearer {t}")]);
            }
        }
    }
    if dry {
        return Some(Ok("would run `claude mcp add --scope user synapstore …`".into()));
    }
    let _ = run(&remove);
    Some(run(&add).map(|_| "added with `claude mcp add --scope user`".into()))
}

fn json_edit(file: &Path, path: &[&str], entry: Option<serde_json::Value>, dry: bool) -> Result<String, String> {
    let text = read(file)?;
    let shown = tilde(file);
    let new = match entry {
        Some(e) => Some(jsonc::upsert(&text, path, NAME, &e).map_err(|e| format!("{shown}: {e}"))?),
        None => jsonc::remove(&text, path, NAME).map_err(|e| format!("{shown}: {e}"))?,
    };
    match new {
        Some(t) if t != text => write(file, &t, dry).map(|_| shown),
        Some(_) => Ok(format!("{shown} (already up to date)")),
        None => Ok(format!("{shown} (nothing to remove)")),
    }
}

fn toml_body(server: &Server) -> String {
    match server {
        Server::Local { command, args } => {
            let args: Vec<String> = args.iter().map(|a| toml::string(a)).collect();
            format!("command = {}\nargs = [{}]", toml::string(command), args.join(", "))
        }
        Server::Remote { url, token } => {
            let mut b = format!("url = {}", toml::string(url));
            if let Some(t) = token {
                b.push_str(&format!(
                    "\nhttp_headers = {{ \"Authorization\" = {} }}",
                    toml::string(&format!("Bearer {t}"))
                ));
            }
            b
        }
    }
}

fn mcp(c: &Client, server: &Server, opts: &Options) -> Result<String, String> {
    let (dry, un) = (opts.dry_run, opts.uninstall);
    match &c.mcp {
        Mcp::Unsupported(why) => Ok(format!("skipped ({why})")),
        Mcp::ClaudeCode => match claude_cli(server, un, dry) {
            Some(r) => r,
            None => json_edit(
                &clients::claude_json(),
                &["mcpServers"],
                (!un).then(|| Style::Standard.entry(server)).flatten(),
                dry,
            ),
        },
        Mcp::Json { file, path, style } => match (un, style.entry(server)) {
            (false, None) => Ok("skipped (add remote servers in the app's connector settings)".into()),
            (false, entry) => json_edit(file, path, entry, dry),
            (true, _) => json_edit(file, path, None, dry),
        },
        Mcp::Toml { file } => {
            let text = read(file)?;
            let new = if un { toml::remove(&text, NAME) } else { Some(toml::upsert(&text, NAME, &toml_body(server))) };
            match new {
                Some(t) if t != text => write(file, &t, dry).map(|_| tilde(file)),
                Some(_) => Ok(format!("{} (already up to date)", tilde(file))),
                None => Ok(format!("{} (nothing to remove)", tilde(file))),
            }
        }
    }
}

fn skill(dir: &Path, opts: &Options) -> Result<String, String> {
    let target = dir.join(NAME);
    let file = target.join("SKILL.md");
    if opts.uninstall {
        let ours = read(&file)?.contains("name: synapstore");
        if ours && !opts.dry_run {
            std::fs::remove_dir_all(&target).map_err(|e| format!("{}: {e}", target.display()))?;
        }
        return Ok(if ours { tilde(&target) } else { format!("{} (nothing to remove)", tilde(&target)) });
    }
    if read(&file)? == SKILL {
        return Ok(format!("{} (already up to date)", tilde(&file)));
    }
    write(&file, SKILL, opts.dry_run).map(|_| tilde(&file))
}

/// Install (or uninstall) into the selected clients; one line per action.
pub fn run(opts: &Options, server: &Server) -> Result<Vec<String>, String> {
    let selected = select(opts)?;
    if selected.is_empty() {
        return Err("no supported agents found; name them with --client (or --client all)".into());
    }
    let mut lines = Vec::new();
    let mut skill_dirs = BTreeSet::new();
    let mut failed = false;
    for c in &selected {
        if opts.only != Some(Part::Skill) {
            let r = mcp(c, server, opts);
            failed |= r.is_err();
            lines.push(format!("{:<15} mcp    {}", c.id, r.unwrap_or_else(|e| format!("FAILED: {e}"))));
        }
        if opts.only != Some(Part::Mcp) {
            match &c.skills {
                Some(d) if skill_dirs.insert(d.clone()) => {
                    let r = skill(d, opts);
                    failed |= r.is_err();
                    lines.push(format!("{:<15} skill  {}", c.id, r.unwrap_or_else(|e| format!("FAILED: {e}"))));
                }
                Some(d) => lines.push(format!("{:<15} skill  {} (shared, see above)", c.id, tilde(&d.join(NAME)))),
                None => lines.push(format!("{:<15} skill  skipped (no skills directory)", c.id)),
            }
        }
    }
    if failed {
        lines.push("some steps failed; see above".into());
    }
    Ok(lines)
}

/// Remote server settings the CLI uses when no `--db`/`--url` is given.
pub fn remote_file() -> PathBuf {
    home().join(".synapstore").join("remote.json")
}

pub fn save_remote(url: &str, token: Option<&str>) -> Result<(), String> {
    let path = remote_file();
    let body = serde_json::json!({"url": url, "token": token}).to_string();
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(err)?;
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    std::io::Write::write_all(&mut o.open(&path).map_err(err)?, body.as_bytes()).map_err(err)
}

/// `(url, token)` saved by `install --url`.
pub fn load_remote() -> Option<(String, Option<String>)> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(remote_file()).ok()?).ok()?;
    Some((v["url"].as_str()?.to_string(), v["token"].as_str().map(str::to_string)))
}
