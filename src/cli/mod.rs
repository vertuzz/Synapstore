//! Hand-rolled command-line parsing (no CLI crates in the dependency budget).

use crate::decay::DAY_MS;
use crate::embed::EmbedderSpec;
use crate::http::ServeOptions;
use crate::logger::{Level, LogConfig};
use crate::store::Config;
use std::path::PathBuf;

pub const USAGE: &str = "\
SynapStore — single-file ACID memory for AI agents (HNSW + knowledge graph + episodic log).

USAGE:
    synapstore --mcp <DB>                 Serve MCP over stdio (local agents)
    synapstore serve <DB> --listen <ADDR> Serve MCP over HTTP with bearer-token auth
    synapstore token create <DB> --name <NAME> [--scope read|write|admin] [--expires-days N]
    synapstore token list <DB>            List API tokens (secrets are never stored)
    synapstore token revoke <DB> <ID|NAME>
    synapstore models                     List built-in embedding models
    synapstore pull <MODEL>               Download a model ahead of time
    synapstore stats <DB> [--detailed]    Print store statistics as JSON
    synapstore dump <DB> [--vectors]      Export every table as JSON lines to stdout
    synapstore repair <DB>                Reconnect unreachable HNSW nodes
    synapstore synth <DB> [--records N] [--seed S] [--dim D]
                                          Load a synthetic agent-memory dataset
    synapstore --version | --help

EMBEDDINGS:
    --embedder <SPEC>       Text embedder for a new store (remembered in the file):
                              lexical            built-in word hashing, no model (default)
                              minilm             English neural model, downloaded once (~90 MB)
                              bge-small          stronger English model (~130 MB)
                              multilingual       ~100 languages (~470 MB)
                              hf:<owner>/<repo>  any BERT model with safetensors on the Hub
                              <DIR>              a local model directory
                              openai:<model>[@<base-url>]  OpenAI-compatible endpoint
                                                 (Ollama: openai:nomic-embed-text@http://localhost:11434/v1)
    --dim <N>               Lexical embedder dimension for a new store (default 384)

HTTP SERVER (serve --listen):
    --listen <ADDR>         Address to bind, e.g. 127.0.0.1:8765 or 0.0.0.0:8765
    --allow-origin <ORIGIN> Allow browser requests from ORIGIN (repeatable)
    --threads <N>           Worker threads (default 16)
    --max-body-mb <N>       Request size limit (default 8)
    --trust-proxy           Use X-Forwarded-For for the client address (behind a proxy)
    --no-auth               Disable token auth (refused unless bound to 127.0.0.1/::1)

OPTIONS:
    --log <FILE>            Log file (default for servers: <DB>.log)
    --no-log                Disable file logging
    --log-level <LEVEL>     error | warn | info | debug (default info)
    --log-max-mb <N>        Rotate after N MiB (default 10)
    --log-keep <N>          Rotated files to keep (default 5)
    --ef-search <N>         HNSW search beam width (default 64)
    --m <N>                 HNSW links per node, fixed at creation (default 16)
    --ef-construction <N>   HNSW build beam width, fixed at creation (default 100)
    --half-life-days <X>    Time-decay half-life (default 30)
    --no-decay              Disable time-decay
    --cache-mb <N>          redb page cache (default 256)
";

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
    Help,
    Version,
}

#[derive(Debug, Clone)]
pub struct Cli {
    pub command: Command,
    pub db: Option<PathBuf>,
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
}

/// Parse `args` (without the program name). Pure.
pub fn parse(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli { command: Command::Help, db: None, config: Config::default(), log_disabled: false };
    let mut verb: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut f = Flags::default();
    let mut log: Option<LogConfig> = None;
    let mut level = Level::Info;
    let (mut max_mb, mut keep) = (10u64, 5usize);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" | "help" => return Ok(Cli { command: Command::Help, ..cli }),
            "-V" | "--version" => return Ok(Cli { command: Command::Version, ..cli }),
            "--mcp" if verb.is_none() => verb = Some("mcp".into()),
            "--detailed" => f.detailed = true,
            "--vectors" => f.vectors = true,
            "--records" => f.records = Some(num(&mut it, a)?),
            "--seed" => f.seed = Some(num(&mut it, a)?),
            "--name" => f.name = Some(value(&mut it, a)?.clone()),
            "--scope" => f.scope = Some(value(&mut it, a)?.clone()),
            "--expires-days" => f.expires_days = Some(num::<u64>(&mut it, a)?.max(1)),
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
            s if s.starts_with('-') => return Err(format!("unknown option `{s}`")),
            s if verb.is_none() => verb = Some(s.to_string()),
            s => positional.push(s.to_string()),
        }
    }
    let mut pos = positional.into_iter();
    let mut take_db = |pos: &mut std::vec::IntoIter<String>| -> Result<(), String> {
        cli.db = Some(PathBuf::from(pos.next().ok_or("missing database path")?));
        Ok(())
    };
    let command = match verb.as_deref() {
        None => Command::Help,
        Some("mcp" | "serve") => {
            take_db(&mut pos)?;
            if f.listen { Command::Http(f.serve) } else { Command::Mcp }
        }
        Some("token") => {
            let sub = pos.next().ok_or("token needs a subcommand: create, list or revoke")?;
            take_db(&mut pos)?;
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
        Some("pull") => Command::Pull { model: pos.next().ok_or("pull needs a model name")? },
        Some(v @ ("stats" | "dump" | "repair" | "synth")) => {
            take_db(&mut pos)?;
            match v {
                "stats" => Command::Stats { detailed: f.detailed },
                "dump" => Command::Dump { vectors: f.vectors },
                "repair" => Command::Repair,
                _ => Command::Synth { records: f.records.unwrap_or(20_000), seed: f.seed.unwrap_or(42) },
            }
        }
        Some(other) => return Err(format!("unknown command `{other}`")),
    };
    if let Some(extra) = pos.next() {
        return Err(format!("unexpected argument `{extra}`"));
    }
    cli.command = command;
    let server = matches!(cli.command, Command::Mcp | Command::Http(_));
    if server && !cli.log_disabled && log.is_none() {
        let mut p = cli.db.clone().unwrap_or_default().into_os_string();
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
