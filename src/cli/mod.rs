//! Hand-rolled command-line parsing (no CLI crates in the dependency budget).

use crate::decay::DAY_MS;
use crate::embed::EmbedderSpec;
use crate::http::ServeOptions;
use crate::install::{self, Part};
use crate::logger::{Level, LogConfig};
use crate::store::Config;
use std::path::PathBuf;

pub mod agent;
mod setup;
mod usage;
pub mod view;
pub use usage::USAGE;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenCmd {
    Create { name: String, scope: String, expires_days: Option<u64> },
    List,
    Revoke { target: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Mcp,
    Http(ServeOptions),
    Token(TokenCmd),
    Models,
    Pull { model: String },
    Stats { detailed: bool },
    Dump { vectors: bool },
    Synth { records: usize, seed: u64 },
    Repair,
    Agent(agent::Op),
    Install(install::Options),
    Skill,
    Help,
    Version,
}

#[derive(Debug, Clone)]
pub struct Cli {
    pub command: Command,
    /// The store: `--db`, a positional path, `$SYNAPSTORE_DB` or `~/.synapstore/memory.redb`.
    pub db: Option<PathBuf>,
    /// The store path came from the command line (not a default).
    pub db_explicit: bool,
    pub config: Config,
    pub log_disabled: bool,
}

fn value<'a>(it: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<&'a String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn num<T: std::str::FromStr>(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<T, String> {
    let v = value(it, flag)?;
    v.parse().map_err(|_| format!("{flag}: invalid number `{v}`"))
}

#[derive(Default)]
struct Flags {
    detailed: bool,
    vectors: bool,
    records: Option<usize>,
    seed: Option<u64>,
    name: Option<String>,
    scope: Option<String>,
    expires_days: Option<u64>,
    serve: ServeOptions,
    listen: bool,
    agent: agent::AgentFlags,
    install: install::Options,
}

/// Parse `args` (without the program name). Pure.
pub fn parse(args: &[String]) -> Result<Cli, String> {
    let mut cli =
        Cli { command: Command::Help, db: None, db_explicit: false, config: Config::default(), log_disabled: false };
    let mut verb: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut f = Flags::default();
    let mut log: Option<LogConfig> = None;
    let mut level = Level::Info;
    let (mut max_mb, mut keep) = (10u64, 5usize);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let g = &mut f.agent;
        match a.as_str() {
            "-h" | "--help" => return Ok(Cli { command: Command::Help, ..cli }),
            "help" if verb.is_none() => return Ok(Cli { command: Command::Help, ..cli }),
            "-V" | "--version" => return Ok(Cli { command: Command::Version, ..cli }),
            "--mcp" if verb.is_none() => verb = Some("mcp".into()),
            "--db" => cli.db = Some(PathBuf::from(value(&mut it, a)?)),
            "--detailed" => f.detailed = true,
            "--vectors" => f.vectors = true,
            "--records" => f.records = Some(num(&mut it, a)?),
            "--seed" => f.seed = Some(num(&mut it, a)?),
            "--name" => f.name = Some(value(&mut it, a)?.clone()),
            "--scope" => f.scope = Some(value(&mut it, a)?.clone()),
            "--expires-days" => f.expires_days = Some(num::<u64>(&mut it, a)?.max(1)),
            "--fact" => g.facts.push(value(&mut it, a)?.clone()),
            "--tag" => g.tags.push(value(&mut it, a)?.clone()),
            "--source" => g.source = Some(value(&mut it, a)?.clone()),
            "--kind" => g.kind = Some(value(&mut it, a)?.clone()),
            "-k" | "--top-k" => g.top_k = Some(num(&mut it, a)?),
            "--limit" => g.limit = Some(num(&mut it, a)?),
            "--depth" => g.depth = Some(num(&mut it, a)?),
            "--max-age" => g.max_age = Some(value(&mut it, a)?.clone()),
            "--subject" => g.subject = Some(value(&mut it, a)?.clone()),
            "--predicate" => g.predicate = Some(value(&mut it, a)?.clone()),
            "--object" => g.object = Some(value(&mut it, a)?.clone()),
            "--json" => g.json = true,
            "--url" => g.url = Some(value(&mut it, a)?.clone()),
            "--client" | "--clients" => {
                f.install
                    .clients
                    .extend(value(&mut it, a)?.split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()));
            }
            "--only" => {
                f.install.only = Some(match value(&mut it, a)?.as_str() {
                    "mcp" => Part::Mcp,
                    "skill" | "skills" => Part::Skill,
                    o => return Err(format!("--only: expected `mcp` or `skill`, got `{o}`")),
                })
            }
            "--dry-run" => f.install.dry_run = true,
            "--listen" => {
                f.serve.listen = value(&mut it, a)?.clone();
                f.listen = true;
            }
            "--allow-origin" => f.serve.allowed_origins.push(value(&mut it, a)?.trim_end_matches('/').to_string()),
            "--threads" => f.serve.threads = num::<usize>(&mut it, a)?.clamp(1, 1024),
            "--no-auth" => f.serve.no_auth = true,
            "--trust-proxy" => f.serve.trust_proxy = true,
            "--max-body-mb" => f.serve.max_body = num::<usize>(&mut it, a)?.clamp(1, 1024) << 20,
            "--embedder" => cli.config.embedder = EmbedderSpec::parse(value(&mut it, a)?).map_err(|e| e.to_string())?,
            "--log" => log = Some(LogConfig::new(value(&mut it, a)?)),
            "--no-log" => cli.log_disabled = true,
            "--log-level" => {
                let v = value(&mut it, a)?;
                level = Level::parse(v).ok_or_else(|| format!("unknown log level `{v}`"))?;
            }
            "--log-max-mb" => max_mb = num(&mut it, a)?,
            "--log-keep" => keep = num(&mut it, a)?,
            "--ef-search" => cli.config.hnsw.ef_search = num(&mut it, a)?,
            "--m" => {
                let m: usize = num(&mut it, a)?;
                cli.config.hnsw.m = m.clamp(2, 128);
                cli.config.hnsw.m0 = 2 * cli.config.hnsw.m;
            }
            "--ef-construction" => cli.config.hnsw.ef_construction = num::<usize>(&mut it, a)?.max(1),
            "--half-life-days" => {
                let d: f64 = num(&mut it, a)?;
                if !(d.is_finite() && d > 0.0) {
                    return Err("--half-life-days must be > 0".into());
                }
                cli.config.decay.half_life_ms = (d * DAY_MS as f64) as u64;
            }
            "--no-decay" => cli.config.decay.enabled = false,
            "--dim" => cli.config.builtin_dim = num::<usize>(&mut it, a)?.clamp(1, 65_536),
            "--cache-mb" => cli.config.cache_bytes = num::<usize>(&mut it, a)?.max(1) * 1024 * 1024,
            "-" => positional.push("-".into()),
            s if s.starts_with('-') => return Err(format!("unknown option `{s}`")),
            s if verb.is_none() => verb = Some(s.to_string()),
            s => positional.push(s.to_string()),
        }
    }
    let n = positional.len();
    let mut pos = positional.into_iter();
    // An optional leading store path: taken when more positionals than `args` remain.
    let mut db_if = |pos: &mut std::vec::IntoIter<String>, n: usize, args: usize| {
        if n > args {
            cli.db = cli.db.take().or_else(|| pos.next().map(PathBuf::from));
        }
    };
    let command = match verb.as_deref() {
        None => Command::Help,
        Some("mcp" | "serve") => {
            db_if(&mut pos, n, 0);
            if f.listen { Command::Http(f.serve) } else { Command::Mcp }
        }
        Some("token") => {
            let sub = pos.next().ok_or("token needs a subcommand: create, list or revoke")?;
            let wants = usize::from(sub == "revoke");
            db_if(&mut pos, n - 1, wants);
            Command::Token(match sub.as_str() {
                "create" => TokenCmd::Create {
                    name: f.name.ok_or("token create needs --name")?,
                    scope: f.scope.unwrap_or_else(|| "write".into()),
                    expires_days: f.expires_days,
                },
                "list" => TokenCmd::List,
                "revoke" => TokenCmd::Revoke { target: pos.next().ok_or("token revoke needs an id or name")? },
                other => return Err(format!("unknown token subcommand `{other}`")),
            })
        }
        Some("models") => Command::Models,
        Some("skill") => Command::Skill,
        Some("pull") => Command::Pull { model: pos.next().ok_or("pull needs a model name")? },
        Some(v @ ("install" | "uninstall")) => {
            let mut o = f.install;
            o.uninstall = v == "uninstall";
            o.url = f.agent.url.clone();
            Command::Install(o)
        }
        Some(v @ ("stats" | "dump" | "repair" | "synth")) => {
            db_if(&mut pos, n, 0);
            match v {
                "stats" => Command::Stats { detailed: f.detailed },
                "dump" => Command::Dump { vectors: f.vectors },
                "repair" => Command::Repair,
                _ => Command::Synth { records: f.records.unwrap_or(20_000), seed: f.seed.unwrap_or(42) },
            }
        }
        Some(v) if agent::VERBS.contains(&v) => Command::Agent(agent::build(v, pos.by_ref().collect(), f.agent)?),
        Some(other) => return Err(format!("unknown command `{other}`")),
    };
    if let Some(extra) = pos.next() {
        return Err(format!("unexpected argument `{extra}`"));
    }
    cli.command = command;
    cli.db_explicit = cli.db.is_some();
    let server = matches!(cli.command, Command::Mcp | Command::Http(_));
    if server && !cli.log_disabled && log.is_none() {
        let mut p = cli.db.clone().unwrap_or_else(crate::paths::default_db).into_os_string();
        p.push(".log");
        log = Some(LogConfig::new(p));
    }
    cli.config.log = if cli.log_disabled {
        None
    } else {
        log.map(|l| LogConfig { level, max_bytes: max_mb.max(1) * 1024 * 1024, keep: keep.max(1), ..l })
    };
    Ok(cli)
}

pub mod run;

#[cfg(test)]
mod tests;
