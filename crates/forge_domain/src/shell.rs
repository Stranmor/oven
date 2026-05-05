use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

use schemars::{JsonSchema, Schema, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Timeout before a shell command is handed off to managed background execution.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShellHandoffTimeoutSeconds(u64);

impl ShellHandoffTimeoutSeconds {
    /// Default synchronous shell handoff timeout in seconds.
    pub const DEFAULT_SECONDS: u64 = 2;

    /// Creates a timeout from externally provided seconds.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is zero.
    pub fn new(value: u64) -> anyhow::Result<Self> {
        if value == 0 {
            anyhow::bail!("handoff_timeout_seconds must be greater than zero");
        }
        Ok(Self(value))
    }

    /// Returns the timeout value as seconds.
    pub const fn seconds(self) -> u64 {
        self.0
    }

    /// Returns the timeout as a standard duration.
    pub const fn duration(self) -> Duration {
        Duration::from_secs(self.0)
    }
}

impl Default for ShellHandoffTimeoutSeconds {
    fn default() -> Self {
        Self(Self::DEFAULT_SECONDS)
    }
}

impl fmt::Display for ShellHandoffTimeoutSeconds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl JsonSchema for ShellHandoffTimeoutSeconds {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("ShellHandoffTimeoutSeconds")
    }

    fn json_schema(_generator: &mut schemars::generate::SchemaGenerator) -> Schema {
        json_schema!({
            "type": "integer",
            "format": "uint64",
            "minimum": 1
        })
    }
}

impl Serialize for ShellHandoffTimeoutSeconds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ShellHandoffTimeoutSeconds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let value = u64::deserialize(deserializer)?;
        Self::new(value).map_err(Error::custom)
    }
}

/// Result of executing a shell command.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CommandExecutionOutput {
    pub output: CommandOutput,
    pub process: Option<ProcessStartOutput>,
}

/// Output from a command execution
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
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
        self.exit_code == Some(0)
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
    pub first_available_cursor: Option<ProcessReadCursor>,
    pub dropped_before_cursor: Option<ProcessReadCursor>,
    pub entries: Vec<ProcessLogEntry>,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use schemars::schema_for;

    use super::*;

    #[test]
    fn test_shell_handoff_timeout_rejects_zero_at_json_boundary() {
        let fixture = "0";

        let actual = serde_json::from_str::<ShellHandoffTimeoutSeconds>(fixture);

        assert!(actual.is_err());
    }

    #[test]
    fn test_shell_handoff_timeout_schema_matches_runtime_boundary() {
        let fixture = schema_for!(ShellHandoffTimeoutSeconds);

        let actual = fixture.to_value();
        let expected = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ShellHandoffTimeoutSeconds",
            "type": "integer",
            "format": "uint64",
            "minimum": 1
        });

        assert_eq!(actual, expected);
    }
}
