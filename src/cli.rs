//! Hand-rolled command-line parsing (no CLI crates in the dependency budget).

use crate::decay::DAY_MS;
use crate::logger::{Level, LogConfig};
use crate::store::Config;
use std::path::PathBuf;

pub const USAGE: &str = "\
SynapStore — single-file ACID memory for AI agents (HNSW + knowledge graph + episodic log).

USAGE:
    synapstore --mcp <DB>                 Serve the Model Context Protocol over stdio
    synapstore stats <DB> [--detailed]    Print store statistics as JSON
    synapstore dump <DB> [--vectors]      Export every table as JSON lines to stdout
    synapstore repair <DB>                Reconnect unreachable HNSW nodes
    synapstore synth <DB> [--records N] [--seed S] [--dim D]
                                          Load a synthetic agent-memory dataset
    synapstore --version | --help

OPTIONS:
    --log <FILE>            Log file (default for --mcp: <DB>.log)
    --no-log                Disable file logging
    --log-level <LEVEL>     error | warn | info | debug (default info)
    --log-max-mb <N>        Rotate after N MiB (default 10)
    --log-keep <N>          Rotated files to keep (default 5)
    --ef-search <N>         HNSW search beam width (default 64)
    --m <N>                 HNSW links per node, fixed at creation (default 16)
    --ef-construction <N>   HNSW build beam width, fixed at creation (default 100)
    --half-life-days <X>    Time-decay half-life (default 30)
    --no-decay              Disable time-decay
    --dim <N>               Built-in embedder dimension for a new store (default 384)
    --cache-mb <N>          redb page cache (default 256)
";

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Mcp,
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

/// Parse `args` (without the program name). Pure.
pub fn parse(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli { command: Command::Help, db: None, config: Config::default(), log_disabled: false };
    let mut command: Option<Command> = None;
    let mut log: Option<LogConfig> = None;
    let mut level = Level::Info;
    let (mut max_mb, mut keep) = (10u64, 5usize);
    let (mut detailed, mut vectors) = (false, false);
    let (mut records, mut seed) = (20_000usize, 42u64);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mcp" | "mcp" | "serve" if command.is_none() => command = Some(Command::Mcp),
            "stats" | "dump" | "synth" | "repair" if command.is_none() => {
                command = Some(match a.as_str() {
                    "stats" => Command::Stats { detailed: false },
                    "repair" => Command::Repair,
                    "dump" => Command::Dump { vectors: false },
                    _ => Command::Synth { records: 0, seed: 0 },
                });
            }
            "-h" | "--help" | "help" => {
                return Ok(Cli { command: Command::Help, ..cli });
            }
            "-V" | "--version" => {
                return Ok(Cli { command: Command::Version, ..cli });
            }
            "--detailed" => detailed = true,
            "--vectors" => vectors = true,
            "--records" => records = num(&mut it, a)?,
            "--seed" => seed = num(&mut it, a)?,
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
            s if cli.db.is_none() => cli.db = Some(PathBuf::from(s)),
            s => return Err(format!("unexpected argument `{s}`")),
        }
    }
    cli.command = match command {
        None => Command::Help,
        Some(Command::Stats { .. }) => Command::Stats { detailed },
        Some(Command::Dump { .. }) => Command::Dump { vectors },
        Some(Command::Synth { .. }) => Command::Synth { records, seed },
        Some(c) => c,
    };
    if !matches!(cli.command, Command::Help | Command::Version) && cli.db.is_none() {
        return Err("missing database path".into());
    }
    if cli.command == Command::Mcp && !cli.log_disabled && log.is_none() {
        let db = cli.db.clone().unwrap_or_default();
        let mut p = db.into_os_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Cli, String> {
        parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn parses_mcp_with_defaults() {
        let c = p("--mcp mem.redb --ef-search 99").unwrap();
        assert_eq!(c.command, Command::Mcp);
        assert_eq!(c.db, Some(PathBuf::from("mem.redb")));
        assert_eq!(c.config.hnsw.ef_search, 99);
        assert_eq!(c.config.log.unwrap().path, PathBuf::from("mem.redb.log"));
    }

    #[test]
    fn parses_subcommands() {
        assert_eq!(p("stats x --detailed").unwrap().command, Command::Stats { detailed: true });
        assert_eq!(p("dump x").unwrap().command, Command::Dump { vectors: false });
        assert_eq!(p("synth x --records 5").unwrap().command, Command::Synth { records: 5, seed: 42 });
        assert!(p("stats x").unwrap().config.log.is_none());
        assert!(p("--mcp x --no-log").unwrap().config.log.is_none());
        assert_eq!(p("").unwrap().command, Command::Help);
        assert_eq!(p("--version").unwrap().command, Command::Version);
    }

    #[test]
    fn rejects_garbage() {
        assert!(p("--mcp").is_err());
        assert_eq!(p("--mcp --no-log x").unwrap().db, Some(PathBuf::from("x")));
        assert!(p("--mcp x --bogus").is_err());
        assert!(p("--mcp x --ef-search nope").is_err());
        assert!(p("--mcp x --half-life-days -3").is_err());
        assert!(p("stats a b").is_err());
    }
}
