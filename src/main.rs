//! `synapstore` binary: MCP server and maintenance commands.
#![deny(warnings)]

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;
use synapstore::cli::{self, Command};
use synapstore::{SynapStore, synth};

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = cli::parse(&args)?;
    let open = |c: &cli::Cli| {
        let db = c.db.clone().unwrap_or_default();
        SynapStore::open(&db, c.config.clone()).map_err(|e| format!("cannot open {}: {e}", db.display()))
    };
    match cli.command {
        Command::Help => print!("{}", cli::USAGE),
        Command::Version => println!("synapstore {}", env!("CARGO_PKG_VERSION")),
        Command::Mcp => {
            let store = open(&cli)?;
            let stdin = io::stdin().lock();
            let stdout = io::stdout().lock();
            synapstore::mcp::serve(store.clone(), stdin, stdout).map_err(|e| e.to_string())?;
            let snapshot = store.metrics().snapshot().to_string();
            store.log().info("shutdown", &[("metrics", snapshot)]);
        }
        Command::Stats { detailed } => {
            let store = open(&cli)?;
            let s = store.stats(detailed).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&s).map_err(|e| e.to_string())?);
        }
        Command::Dump { vectors } => {
            let store = open(&cli)?;
            let mut out = BufWriter::new(io::stdout().lock());
            store.dump(&mut out, vectors).map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
        }
        Command::Repair => {
            let store = open(&cli)?;
            let o = store.repair_index().map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&o).map_err(|e| e.to_string())?);
        }
        Command::Synth { records, seed } => {
            let store = open(&cli)?;
            let spec = synth::Spec::scaled(records, seed, cli.config.builtin_dim.min(256));
            let (_, report) = synth::load(&store, &spec).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("synapstore: {e}");
            if !e.starts_with("cannot open") {
                eprintln!("Run `synapstore --help` for usage.");
            }
            ExitCode::FAILURE
        }
    }
}
