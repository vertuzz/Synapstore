//! `synapstore` binary: MCP servers and maintenance commands.
#![deny(warnings)]

use std::process::ExitCode;
use synapstore::cli;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match cli::parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("synapstore: {e}\nRun `synapstore --help` for usage.");
            return ExitCode::from(2);
        }
    };
    match cli::run::run(parsed) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("synapstore: {e}");
            ExitCode::FAILURE
        }
    }
}
