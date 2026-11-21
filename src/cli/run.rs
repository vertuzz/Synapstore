//! Executing a parsed [`Cli`]. Output goes to stdout (data) and stderr
//! (human status lines); `--mcp` keeps stdout for protocol traffic only.

use super::{Cli, Command, TokenCmd};
use crate::auth::{Scope, TokenInfo};
use crate::embed::fetch;
use crate::timefmt::rfc3339;
use crate::{SynapStore, http, synth};
use std::io::{self, BufWriter, Write};

/// `println!` that exits quietly when stdout is closed (`synapstore … | head`).
macro_rules! out {
    ($($t:tt)*) => {{
        if let Err(e) = writeln!(io::stdout(), $($t)*) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }
    }};
}

/// Print a block of text to stdout (quietly ends on a closed pipe).
pub fn print(text: &str) {
    out!("{text}");
}

fn open(c: &Cli) -> Result<SynapStore, String> {
    let db = c.db.clone().unwrap_or_else(crate::paths::default_db);
    if let Some(dir) = db.parent().filter(|d| !d.as_os_str().is_empty() && !d.exists()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let mut cfg = c.config.clone();
    cfg.embedder_progress = true;
    SynapStore::open(&db, cfg).map_err(|e| format!("cannot open {}: {e}", db.display()))
}

fn json<T: serde::Serialize>(v: &T) -> Result<(), String> {
    out!("{}", serde_json::to_string_pretty(v).map_err(|e| e.to_string())?);
    Ok(())
}

fn fmt_time(ms: Option<u64>) -> String {
    ms.map_or_else(|| "-".into(), rfc3339)
}

fn print_tokens(tokens: &[TokenInfo]) {
    out!("{:<14} {:<20} {:<6} {:<24} {:<24} LAST USED", "ID", "NAME", "SCOPE", "CREATED", "EXPIRES");
    for t in tokens {
        out!(
            "{:<14} {:<20} {:<6} {:<24} {:<24} {}",
            t.id,
            t.name,
            t.scope.as_str(),
            rfc3339(t.created_at),
            t.expires_at.map_or_else(|| "never".into(), rfc3339),
            fmt_time(t.last_used_at)
        );
    }
}

fn serve_http(c: &Cli, opts: &http::ServeOptions) -> Result<(), String> {
    let listener = http::bind(opts).map_err(|e| format!("cannot listen on {}: {e}", opts.listen))?;
    let store = open(c)?;
    if !opts.no_auth && store.list_tokens().map_err(|e| e.to_string())?.is_empty() {
        let (secret, info) = store.create_token("default", Scope::Write, None).map_err(|e| e.to_string())?;
        eprintln!("synapstore: no API tokens yet; created token `{}` (id {}, scope write):\n", info.name, info.id);
        eprintln!("    {secret}\n");
        eprintln!("  It is shown only this once. Manage tokens with `synapstore token create|list|revoke`.");
    }
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!(
        "synapstore: serving MCP on http://{addr}/mcp ({})",
        if opts.no_auth { "NO AUTH" } else { "bearer auth" }
    );
    if !addr.ip().is_loopback() {
        eprintln!("synapstore: listening on a public address; terminate TLS in front of it (see README)");
    }
    store.log().info("http_listen", &[("addr", addr.to_string()), ("auth", (!opts.no_auth).to_string())]);
    http::serve(store, opts.clone(), listener).map_err(|e| e.to_string())
}

fn token(c: &Cli, cmd: &TokenCmd) -> Result<(), String> {
    let store = open(c)?;
    match cmd {
        TokenCmd::Create { name, scope, expires_days } => {
            let scope = Scope::parse(scope).map_err(|e| e.to_string())?;
            let (secret, info) = store.create_token(name, scope, *expires_days).map_err(|e| e.to_string())?;
            eprintln!(
                "Created token `{}` (id {}, scope {}). It is shown only once:",
                info.name,
                info.id,
                scope.as_str()
            );
            out!("{secret}");
        }
        TokenCmd::List => print_tokens(&store.list_tokens().map_err(|e| e.to_string())?),
        TokenCmd::Revoke { target } => {
            let gone = store.revoke_token(target).map_err(|e| e.to_string())?;
            if gone.is_empty() {
                return Err(format!("no token with id or name `{target}`"));
            }
            for t in gone {
                eprintln!("Revoked token `{}` (id {})", t.name, t.id);
            }
        }
    }
    Ok(())
}

fn models() {
    out!("{:<13} {:<40} {:>4} {:>7}  {:<11} STATUS", "NAME", "REPOSITORY", "DIM", "SIZE", "LICENSE");
    for m in fetch::REGISTRY {
        let cached = fetch::resolve(m.name).map(|r| r.cache_dir().join(".synapstore-complete").is_file());
        let status = if cached.unwrap_or(false) { "downloaded" } else { "not downloaded" };
        out!("{:<13} {:<40} {:>4} {:>4} MB  {:<11} {status}", m.name, m.repo, m.dim, m.download_mb, m.license);
        out!("{:<13} {} ({})", "", m.about, m.languages);
    }
    out!("\nCache: {}", fetch::models_dir().display());
    out!("Also: lexical (no model), hf:<owner>/<repo>, a local directory, openai:<model>[@<url>]");
}

/// Run the command; `Err` is a message for the user.
pub fn run(c: Cli) -> Result<(), String> {
    match &c.command {
        Command::Help => out!("{}", super::USAGE.trim_end()),
        Command::Version => out!("synapstore {}", env!("CARGO_PKG_VERSION")),
        Command::Models => models(),
        Command::Pull { model } => {
            let e = fetch::load_model(model, &|m| eprintln!("synapstore: {m}")).map_err(|e| e.to_string())?;
            out!("{}", crate::embed::Embedder::source(&e));
        }
        Command::Mcp => {
            let store = open(&c)?;
            crate::mcp::serve(store.clone(), io::stdin().lock(), io::stdout().lock()).map_err(|e| e.to_string())?;
            let snapshot = store.metrics().snapshot().to_string();
            store.log().info("shutdown", &[("metrics", snapshot)]);
        }
        Command::Http(opts) => serve_http(&c, opts)?,
        Command::Agent(op) => {
            let mut op = op.clone();
            let mut token = None;
            if op.url.is_none() && !c.db_explicit && std::env::var_os("SYNAPSTORE_URL").is_none() {
                if let Some((url, saved)) = crate::install::load_remote() {
                    op.url = Some(url);
                    token = saved;
                }
            }
            super::agent::execute(op, token, || open(&c))?
        }
        Command::Install(opts) => super::setup::install(&c, opts)?,
        Command::Skill => out!("{}", crate::install::SKILL.trim_end()),
        Command::Token(cmd) => token(&c, cmd)?,
        Command::Stats { detailed } => json(&open(&c)?.stats(*detailed).map_err(|e| e.to_string())?)?,
        Command::Dump { vectors } => {
            let store = open(&c)?;
            let mut out = BufWriter::new(io::stdout().lock());
            // A closed pipe (`dump | head`) is not an error.
            let quiet_on_pipe = |e: crate::Error| {
                if e.to_string().contains("Broken pipe") {
                    std::process::exit(0);
                }
                e.to_string()
            };
            store.dump(&mut out, *vectors).map_err(quiet_on_pipe)?;
            if let Err(e) = out.flush() {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    std::process::exit(0);
                }
                return Err(e.to_string());
            }
        }
        Command::Repair => json(&open(&c)?.repair_index().map_err(|e| e.to_string())?)?,
        Command::Synth { records, seed } => {
            let store = open(&c)?;
            let spec = synth::Spec::scaled(*records, *seed, c.config.builtin_dim.min(256));
            json(&synth::load(&store, &spec).map_err(|e| e.to_string())?.1)?;
        }
    }
    Ok(())
}
