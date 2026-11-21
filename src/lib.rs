//! SynapStore — a flash drive for an agent's brain.
#![deny(warnings)]

pub mod archive;
pub mod auth;
pub mod chunker;
pub mod cli;
pub mod crypto;
pub mod decay;
pub mod embed;
pub mod error;
pub mod extract;
pub mod graph;
pub mod hash;
pub mod hnsw;
pub mod http;
pub mod idlist;
pub mod logger;
pub mod math;
pub mod mcp;
pub mod metrics;
pub mod model;
pub mod retry;
pub mod store;
pub mod synth;
pub mod tables;
pub mod timefmt;

pub use error::{Error, Result};
pub use store::{Config, SynapStore};
