//! `synapstore install` / `uninstall`: wire agents to this binary or to a server.

use super::Cli;
use crate::install::{self, Part, clients::Server, tilde};
use crate::paths::{default_db, which};

fn server(c: &Cli, opts: &install::Options) -> Result<Server, String> {
    if let Some(url) = &opts.url {
        let token = std::env::var("SYNAPSTORE_TOKEN").ok().filter(|t| !t.is_empty());
        let url = super::agent::endpoint(url);
        let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|p| url.starts_with(p));
        if token.is_none() && !opts.uninstall && !local {
            return Err("set SYNAPSTORE_TOKEN to the token the agents should use \
                        (on the server: `synapstore token create --name agents`)"
                .into());
        }
        return Ok(Server::Remote { url, token });
    }
    let exe =
        std::env::current_exe().and_then(|p| p.canonicalize()).map_err(|e| format!("cannot locate myself: {e}"))?;
    let mut args = vec!["mcp".to_string()];
    if c.db_explicit {
        let db = c.db.clone().unwrap_or_default();
        let db = std::path::absolute(&db).unwrap_or(db);
        args.extend(["--db".into(), db.display().to_string()]);
    }
    // Windows `canonicalize` yields `\\?\C:\…`; agents expect a plain path.
    let command = exe.display().to_string();
    let command = command.strip_prefix(r"\\?\").map_or(command.clone(), str::to_string);
    Ok(Server::Local { command, args })
}

/// Create the store now (and download its model) so the first agent call is fast.
fn init_store(c: &Cli) -> Result<String, String> {
    let db = c.db.clone().unwrap_or_else(default_db);
    if let Some(dir) = db.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let mut cfg = c.config.clone();
    cfg.embedder_progress = true;
    cfg.log = None;
    let store = crate::SynapStore::open(&db, cfg).map_err(|e| format!("cannot open {}: {e}", db.display()))?;
    Ok(format!("{} (embedder: {})", tilde(&db), store.embedder().source()))
}

pub fn install(c: &Cli, opts: &install::Options) -> Result<(), String> {
    let server = server(c, opts)?;
    let verb = match (opts.uninstall, opts.dry_run) {
        (false, false) => "Installing",
        (true, false) => "Uninstalling",
        (false, true) => "Dry run: would install",
        (true, true) => "Dry run: would uninstall",
    };
    match &server {
        Server::Local { .. } => eprintln!("{verb} SynapStore (local store) for your agents:\n"),
        Server::Remote { url, .. } => eprintln!("{verb} SynapStore (server {url}) for your agents:\n"),
    }
    if !opts.uninstall && !opts.dry_run && opts.only != Some(Part::Skill) && matches!(server, Server::Local { .. }) {
        super::run::print(&format!("{:<15} store  {}", "synapstore", init_store(c)?));
    }
    for line in install::run(opts, &server)? {
        super::run::print(&line);
    }
    let remote = install::remote_file();
    match (&server, opts.uninstall, opts.dry_run) {
        (Server::Remote { url, token }, false, false) => {
            install::save_remote(url, token.as_deref())?;
            super::run::print(&format!("{:<15} cli    {} (CLI and pi use the server)", "synapstore", tilde(&remote)));
        }
        (_, true, false) if remote.is_file() => {
            std::fs::remove_file(&remote).map_err(|e| e.to_string())?;
            super::run::print(&format!("{:<15} cli    {} removed", "synapstore", tilde(&remote)));
        }
        _ => {}
    }
    if opts.uninstall || opts.dry_run {
        if opts.uninstall {
            eprintln!("\nThe memory store itself was not touched.");
        }
        return Ok(());
    }
    if matches!(server, Server::Remote { token: Some(_), .. }) {
        eprintln!("\nThe token was written into the agents' config files (private to your user account).");
    }
    if which("synapstore").is_none() {
        let dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.display().to_string()));
        eprintln!(
            "\nNote: `synapstore` is not on PATH; skills call it by name. Add {} to PATH.",
            dir.unwrap_or_else(|| "its directory".into())
        );
    }
    eprintln!(
        "\nRestart your agents to load it. Try: synapstore remember \"SynapStore is set up\" && synapstore recall setup"
    );
    Ok(())
}
