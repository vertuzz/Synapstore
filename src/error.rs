//! Unified error type for SynapStore.

use thiserror::Error;

/// Result alias used across the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every failure the store can report. Variants are coarse on purpose: MCP
/// clients only need a stable, human-readable message plus a category.
#[derive(Debug, Error)]
pub enum Error {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("database is busy (write lock held by another process) after {0} retries")]
    Busy(u32),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("corrupt record in table `{table}` for key {key}: {reason}")]
    Corrupt { table: &'static str, key: String, reason: String },

    #[error("vector dimension mismatch: store uses {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid vector: {0}")]
    InvalidVector(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("hash collision: id {id:016x} already stores different content in `{table}`")]
    HashCollision { table: &'static str, id: u64 },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("embedding source mismatch: store uses `{store}` vectors, request used `{request}`")]
    EmbeddingMismatch { store: String, request: String },

    #[error("writer thread is unavailable: {0}")]
    WriterGone(String),

    #[error("embedding endpoint: {0}")]
    Embedding(String),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Short machine-readable category, used in logs and MCP error payloads.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Storage(_) => "storage",
            Error::Busy(_) => "busy",
            Error::Serialization(_) => "serialization",
            Error::Corrupt { .. } => "corrupt",
            Error::DimensionMismatch { .. } => "dimension_mismatch",
            Error::InvalidVector(_) => "invalid_vector",
            Error::InvalidInput(_) => "invalid_input",
            Error::HashCollision { .. } => "hash_collision",
            Error::NotFound(_) => "not_found",
            Error::EmbeddingMismatch { .. } => "embedding_mismatch",
            Error::WriterGone(_) => "writer_gone",
            Error::Embedding(_) => "embedding_unavailable",
            Error::Io(_) => "io",
        }
    }

    /// True for errors caused by the caller's input (not by the store).
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Error::DimensionMismatch { .. }
                | Error::InvalidVector(_)
                | Error::InvalidInput(_)
                | Error::NotFound(_)
                | Error::EmbeddingMismatch { .. }
        )
    }
}

macro_rules! storage_from {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Error::Storage(e.to_string())
            }
        })*
    };
}

storage_from!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError,
);

impl From<rkyv::rancor::Error> for Error {
    fn from(e: rkyv::rancor::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}
