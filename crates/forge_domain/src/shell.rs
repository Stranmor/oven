use std::fmt;

use serde::{Deserialize, Serialize};

/// Output from a command execution
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

/// Typed identifier for a managed background process.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ProcessId(String);

impl ProcessId {
    /// Creates a process identifier from internally generated storage text.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Parses a process identifier received from an external boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is empty after trimming.
    pub fn parse(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            anyhow::bail!("Process id must not be empty");
        }
        Ok(Self(value))
    }

    /// Returns the stable process identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProcessId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl CommandOutput {
    pub fn success(&self) -> bool {
        self.exit_code.is_none_or(|code| code >= 0)
    }
}

/// Cursor for reading newly captured background process output.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ProcessReadCursor(u64);

impl ProcessReadCursor {
    /// Creates a cursor from a monotonic log sequence number.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the monotonic log sequence number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Lifecycle state of a managed background process.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStatusKind {
    Running,
    Exited { exit_code: Option<i32> },
    Killed,
}

/// Structured status of a managed background process.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessStatus {
    pub process_id: ProcessId,
    pub status: ProcessStatusKind,
    pub command: String,
    pub cwd: String,
}

/// Output stream captured from a managed background process.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStream {
    Stdout,
    Stderr,
}

/// One captured output chunk from a managed background process.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessLogEntry {
    pub cursor: ProcessReadCursor,
    pub stream: ProcessStream,
    pub content: String,
}

/// Handle returned by background process startup.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessStartOutput {
    pub process_id: ProcessId,
    pub status: ProcessStatusKind,
    pub command: String,
    pub cwd: String,
}

/// Read result for a managed background process.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessReadOutput {
    pub process_id: ProcessId,
    pub next_cursor: ProcessReadCursor,
    pub entries: Vec<ProcessLogEntry>,
}
