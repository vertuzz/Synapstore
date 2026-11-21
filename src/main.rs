//! `synapstore` binary: MCP servers and maintenance commands.
#![deny(warnings)]

use std::process::ExitCode;
use synapstore::cli;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse(&args).and_then(cli::run::run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("synapstore: {e}");
            if !e.starts_with("cannot ") && !e.starts_with("no token") {
                eprintln!("Run `synapstore --help` for usage.");
            }
            ExitCode::FAILURE
        }
    }
}
