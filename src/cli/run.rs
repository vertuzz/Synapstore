//! Executing a parsed [`Cli`]. Output goes to stdout (data) and stderr
//! (human status lines); `--mcp` keeps stdout for protocol traffic only.

use super::{Cli, Command, TokenCmd};
use crate::auth::{Scope, TokenInfo};
use crate::embed::fetch;
use crate::timefmt::rfc3339;
use crate::{SynapStore, http, synth};
use std::io::{self, BufWriter, Write};

fn open(c: &Cli) -> Result<SynapStore, String> {
    let db = c.db.clone().unwrap_or_default();
    let mut cfg = c.config.clone();
    cfg.embedder_progress = true;
    SynapStore::open(&db, cfg).map_err(|e| format!("cannot open {}: {e}", db.display()))
}

fn json<T: serde::Serialize>(v: &T) -> Result<(), String> {
    println!("{}", serde_json::to_string_pretty(v).map_err(|e| e.to_string())?);
    Ok(())
}

fn fmt_time(ms: Option<u64>) -> String {
    ms.map_or_else(|| "-".into(), rfc3339)
}

fn print_tokens(tokens: &[TokenInfo]) {
    println!("{:<14} {:<20} {:<6} {:<24} {:<24} LAST USED", "ID", "NAME", "SCOPE", "CREATED", "EXPIRES");
    for t in tokens {
        println!(
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
            println!("{secret}");
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
    println!("{:<10} {:<40} {:>4} {:>7}  {:<11} STATUS", "NAME", "REPOSITORY", "DIM", "SIZE", "LICENSE");
    for m in fetch::REGISTRY {
        let cached = fetch::resolve(m.name).map(|r| r.cache_dir().join(".synapstore-complete").is_file());
        let status = if cached.unwrap_or(false) { "downloaded" } else { "not downloaded" };
        println!("{:<10} {:<40} {:>4} {:>4} MB  {:<11} {status}", m.name, m.repo, m.dim, m.download_mb, m.license);
        println!("{:<10} {} ({})", "", m.about, m.languages);
    }
    println!("\nCache: {}", fetch::models_dir().display());
    println!("Also: lexical (no model), hf:<owner>/<repo>, a local directory, openai:<model>[@<url>]");
}

/// Run the command; `Err` is a message for the user.
pub fn run(c: Cli) -> Result<(), String> {
    match &c.command {
        Command::Help => print!("{}", super::USAGE),
        Command::Version => println!("synapstore {}", env!("CARGO_PKG_VERSION")),
        Command::Models => models(),
        Command::Pull { model } => {
            let e = fetch::load_model(model, &|m| eprintln!("synapstore: {m}")).map_err(|e| e.to_string())?;
            println!("{}", crate::embed::Embedder::source(&e));
        }
        Command::Mcp => {
            let store = open(&c)?;
            crate::mcp::serve(store.clone(), io::stdin().lock(), io::stdout().lock()).map_err(|e| e.to_string())?;
            let snapshot = store.metrics().snapshot().to_string();
            store.log().info("shutdown", &[("metrics", snapshot)]);
        }
        Command::Http(opts) => serve_http(&c, opts)?,
        Command::Token(cmd) => token(&c, cmd)?,
        Command::Stats { detailed } => json(&open(&c)?.stats(*detailed).map_err(|e| e.to_string())?)?,
        Command::Dump { vectors } => {
            let store = open(&c)?;
            let mut out = BufWriter::new(io::stdout().lock());
            store.dump(&mut out, *vectors).map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
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
