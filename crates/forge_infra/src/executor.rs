use std::collections::{HashMap, VecDeque};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bstr::ByteSlice;
use forge_app::CommandInfra;
use forge_domain::{
    CommandOutput, ConsoleWriter as OutputPrinterTrait, Environment, ProcessId, ProcessLogEntry,
    ProcessReadCursor, ProcessReadOutput, ProcessStartOutput, ProcessStatus, ProcessStatusKind,
    ProcessStream,
};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::console::StdConsoleWriter;

const MAX_BACKGROUND_LOG_ENTRIES: usize = 8192;

/// Service for executing shell commands
#[derive(Clone)]
pub struct ForgeCommandExecutorService {
    env: Environment,
    output_printer: Arc<StdConsoleWriter>,

    // Mutex to ensure that only one command is executed at a time
    ready: Arc<Mutex<()>>,
    processes: Arc<Mutex<HashMap<ProcessId, ManagedProcess>>>,
}

struct ManagedProcess {
    command: String,
    cwd: String,
    child: Child,
    status: ProcessStatusKind,
    logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
}

impl ManagedProcess {
    fn refresh_status(&mut self) -> anyhow::Result<()> {
        if matches!(self.status, ProcessStatusKind::Running)
            && let Some(status) = self.child.try_wait()?
        {
            self.status = ProcessStatusKind::Exited { exit_code: status.code() };
        }
        Ok(())
    }

    fn status(&self, process_id: ProcessId) -> ProcessStatus {
        ProcessStatus {
            process_id,
            status: self.status.clone(),
            command: self.command.clone(),
            cwd: self.cwd.clone(),
        }
    }
}

impl ForgeCommandExecutorService {
    pub fn new(env: Environment, output_printer: Arc<StdConsoleWriter>) -> Self {
        Self {
            env,
            output_printer,
            ready: Arc::new(Mutex::new(())),
            processes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn prepare_command(
        &self,
        command_str: &str,
        working_dir: &Path,
        env_vars: Option<Vec<String>>,
    ) -> Command {
        // Create a basic command
        let is_windows = cfg!(target_os = "windows");
        let shell = self.env.shell.as_str();
        let mut command = Command::new(shell);

        // Core color settings for general commands
        command
            .env("CLICOLOR_FORCE", "1")
            .env("FORCE_COLOR", "true")
            .env_remove("NO_COLOR");

        // Language/program specific color settings
        command
            .env("SBT_OPTS", "-Dsbt.color=always")
            .env("JAVA_OPTS", "-Dsbt.color=always");

        // enabled Git colors
        command.env("GIT_CONFIG_PARAMETERS", "'color.ui=always'");

        // Other common tools
        command.env("GREP_OPTIONS", "--color=always"); // GNU grep

        let parameter = if is_windows { "/C" } else { "-c" };
        command.arg(parameter);

        #[cfg(windows)]
        command.raw_arg(command_str);
        #[cfg(unix)]
        command.arg(command_str);

        tracing::info!(command = command_str, "Executing command");

        command.kill_on_drop(true);

        // Set the working directory
        command.current_dir(working_dir);

        // Configure the command for output
        command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set requested environment variables
        if let Some(env_vars) = env_vars {
            for env_var in env_vars {
                if let Ok(value) = std::env::var(&env_var) {
                    command.env(&env_var, value);
                    tracing::debug!(env_var = %env_var, "Set environment variable from system");
                } else {
                    tracing::warn!(env_var = %env_var, "Environment variable not found in system");
                }
            }
        }

        command
    }

    fn next_process_id() -> ProcessId {
        static PROCESS_COUNTER: AtomicU64 = AtomicU64::new(1);
        let sequence = PROCESS_COUNTER.fetch_add(1, Ordering::Relaxed);
        ProcessId::new(format!("process-{sequence}"))
    }

    fn capture_stream<A>(
        mut reader: A,
        stream: ProcessStream,
        logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
        next_cursor: Arc<AtomicU64>,
    ) where
        A: AsyncReadExt + Unpin + Send + 'static,
    {
        tokio::spawn(async move {
            let mut buffer = [0; 1024];
            loop {
                let Ok(count) = reader.read(&mut buffer).await else {
                    break;
                };
                if count == 0 {
                    break;
                }
                let Some(bytes) = buffer.get(..count) else {
                    break;
                };
                let content = bytes.to_str_lossy().into_owned();
                let mut logs = logs.lock().await;
                let cursor = next_cursor.fetch_add(1, Ordering::Relaxed);
                logs.push_back(ProcessLogEntry {
                    cursor: ProcessReadCursor::new(cursor),
                    stream,
                    content,
                });
                while logs.len() > MAX_BACKGROUND_LOG_ENTRIES {
                    logs.pop_front();
                }
            }
        });
    }

    /// Internal method to execute commands with streaming to console
    async fn execute_command_internal(
        &self,
        command: String,
        working_dir: &Path,
        silent: bool,
        env_vars: Option<Vec<String>>,
    ) -> anyhow::Result<CommandOutput> {
        let ready = self.ready.lock().await;

        let mut prepared_command = self.prepare_command(&command, working_dir, env_vars);

        // Spawn the command
        let mut child = prepared_command.spawn()?;

        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        // Stream the output of the command to stdout and stderr concurrently
        let (status, stdout_buffer, stderr_buffer) = if silent {
            tokio::try_join!(
                child.wait(),
                stream(&mut stdout_pipe, io::sink()),
                stream(&mut stderr_pipe, io::sink())
            )?
        } else {
            let stdout_writer = OutputPrinterWriter::stdout(self.output_printer.clone());
            let stderr_writer = OutputPrinterWriter::stderr(self.output_printer.clone());
            let result = tokio::try_join!(
                child.wait(),
                stream(&mut stdout_pipe, stdout_writer),
                stream(&mut stderr_pipe, stderr_writer)
            )?;

            // If the command's stdout did not end with a newline, the terminal
            // cursor is left mid-line. Write a newline so that subsequent output
            // (e.g. the LLM response) starts on a fresh line.
            if result.1.last() != Some(&b'\n') && !result.1.is_empty() {
                let _ = self.output_printer.write(b"\n");
                let _ = self.output_printer.flush();
            }

            result
        };

        // Drop happens after `try_join` due to <https://github.com/tokio-rs/tokio/issues/4309>
        drop(stdout_pipe);
        drop(stderr_pipe);
        drop(ready);

        Ok(CommandOutput {
            stdout: stdout_buffer.to_str_lossy().into_owned(),
            stderr: stderr_buffer.to_str_lossy().into_owned(),
            exit_code: status.code(),
            command,
        })
    }
}

/// Writer that delegates to OutputPrinter for synchronized writes.
struct OutputPrinterWriter {
    printer: Arc<StdConsoleWriter>,
    is_stdout: bool,
}

impl OutputPrinterWriter {
    fn stdout(printer: Arc<StdConsoleWriter>) -> Self {
        Self { printer, is_stdout: true }
    }

    fn stderr(printer: Arc<StdConsoleWriter>) -> Self {
        Self { printer, is_stdout: false }
    }
}

impl Write for OutputPrinterWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.is_stdout {
            self.printer.write(buf)
        } else {
            self.printer.write_err(buf)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.is_stdout {
            self.printer.flush()
        } else {
            self.printer.flush_err()
        }
    }
}

/// reads the output from A and writes it to W
async fn stream<A: AsyncReadExt + Unpin, W: Write>(
    io: &mut Option<A>,
    mut writer: W,
) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    if let Some(io) = io.as_mut() {
        let mut buff = [0; 1024];
        // Carry incomplete trailing UTF-8 codepoint bytes across reads — Windows
        // console stdio rejects even one byte of a split codepoint.
        let mut pending = Vec::<u8>::new();
        loop {
            let n = io.read(&mut buff).await?;
            if n == 0 {
                break;
            }
            let chunk = buff.get(..n).unwrap_or(&[]);
            output.extend_from_slice(chunk);

            let mut working = std::mem::take(&mut pending);
            working.extend_from_slice(chunk);
            pending = write_lossy_utf8(&mut writer, &working)?;
            // note: flush is necessary else we get the cursor could not be found error.
            writer.flush()?;
        }
        // Flush dangling bytes from a stream that ended mid-codepoint.
        if !pending.is_empty() {
            writer.write_all(pending.to_str_lossy().as_bytes())?;
            writer.flush()?;
        }
    }
    Ok(output)
}

/// Writes `buf` as valid UTF-8 (invalid bytes → `U+FFFD`) and returns any
/// incomplete trailing codepoint bytes for the caller to carry into the next
/// chunk.
fn write_lossy_utf8<W: Write>(writer: &mut W, buf: &[u8]) -> io::Result<Vec<u8>> {
    let mut chunks = ByteSlice::utf8_chunks(buf).peekable();

    while let Some(chunk) = chunks.next() {
        writer.write_all(chunk.valid().as_bytes())?;

        if !chunk.invalid().is_empty() {
            if chunk.incomplete() && chunks.peek().is_none() {
                return Ok(chunk.invalid().to_vec());
            }
            writer.write_all("\u{FFFD}".as_bytes())?;
        }
    }

    Ok(Vec::new())
}

/// The implementation for CommandExecutorService
#[async_trait::async_trait]
impl CommandInfra for ForgeCommandExecutorService {
    async fn execute_command(
        &self,
        command: String,
        working_dir: PathBuf,
        silent: bool,
        env_vars: Option<Vec<String>>,
    ) -> anyhow::Result<CommandOutput> {
        self.execute_command_internal(command, &working_dir, silent, env_vars)
            .await
    }

    async fn execute_command_raw(
        &self,
        command: &str,
        working_dir: PathBuf,
        env_vars: Option<Vec<String>>,
    ) -> anyhow::Result<std::process::ExitStatus> {
        let mut prepared_command = self.prepare_command(command, &working_dir, env_vars);
        prepared_command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());
        Ok(prepared_command.spawn()?.wait().await?)
    }

    async fn start_process(
        &self,
        command: String,
        working_dir: PathBuf,
        env_vars: Option<Vec<String>>,
    ) -> anyhow::Result<ProcessStartOutput> {
        let mut prepared_command = self.prepare_command(&command, &working_dir, env_vars);
        prepared_command.stdin(std::process::Stdio::null());
        let mut child = prepared_command.spawn()?;
        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let next_cursor = Arc::new(AtomicU64::new(1));

        if let Some(stdout) = child.stdout.take() {
            Self::capture_stream(
                stdout,
                ProcessStream::Stdout,
                logs.clone(),
                next_cursor.clone(),
            );
        }
        if let Some(stderr) = child.stderr.take() {
            Self::capture_stream(
                stderr,
                ProcessStream::Stderr,
                logs.clone(),
                next_cursor.clone(),
            );
        }

        let process_id = Self::next_process_id();
        let output = ProcessStartOutput {
            process_id: process_id.clone(),
            status: ProcessStatusKind::Running,
            command: command.clone(),
            cwd: working_dir.display().to_string(),
        };
        self.processes.lock().await.insert(
            process_id,
            ManagedProcess {
                command,
                cwd: working_dir.display().to_string(),
                child,
                status: ProcessStatusKind::Running,
                logs,
            },
        );
        Ok(output)
    }

    async fn process_status(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(&process_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"))?;
        process.refresh_status()?;
        Ok(process.status(process_id))
    }

    async fn read_process(
        &self,
        process_id: ProcessId,
        cursor: ProcessReadCursor,
    ) -> anyhow::Result<ProcessReadOutput> {
        let processes = self.processes.lock().await;
        let process = processes
            .get(&process_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"))?;
        let mut entries: Vec<_> = process
            .logs
            .lock()
            .await
            .iter()
            .filter(|entry| entry.cursor > cursor)
            .cloned()
            .collect();
        entries.sort_by_key(|entry| entry.cursor);
        let next = entries.last().map(|entry| entry.cursor).unwrap_or(cursor);
        Ok(ProcessReadOutput { process_id, next_cursor: next, entries })
    }

    async fn list_processes(&self) -> anyhow::Result<Vec<ProcessStatus>> {
        let mut processes = self.processes.lock().await;
        let mut statuses = Vec::with_capacity(processes.len());
        for (process_id, process) in processes.iter_mut() {
            process.refresh_status()?;
            statuses.push(process.status(process_id.clone()));
        }
        Ok(statuses)
    }

    async fn kill_process(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(&process_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"))?;
        if matches!(process.status, ProcessStatusKind::Running) {
            process.child.kill().await?;
            process.status = ProcessStatusKind::Killed;
        }
        Ok(process.status(process_id))
    }
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use super::*;

    fn test_env() -> Environment {
        use fake::{Fake, Faker};
        let fixture: Environment = Faker.fake();
        fixture.shell(
            if cfg!(target_os = "windows") {
                "cmd"
            } else {
                "bash"
            }
            .to_string(),
        )
    }

    fn test_printer() -> Arc<StdConsoleWriter> {
        Arc::new(StdConsoleWriter::default())
    }

    #[tokio::test]
    async fn test_background_process_lifecycle_captures_output() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let actual = fixture
            .start_process(
                "printf ready; sleep 1".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let output = fixture
            .read_process(actual.process_id.clone(), ProcessReadCursor::new(0))
            .await
            .unwrap();
        let status = fixture.kill_process(actual.process_id).await.unwrap();

        assert!(
            output
                .entries
                .iter()
                .any(|entry| entry.content.contains("ready"))
        );
        assert_eq!(status.status, ProcessStatusKind::Killed);
    }

    #[tokio::test]
    async fn test_process_read_next_cursor_does_not_skip_later_output() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(
                "printf first; sleep 0.3; printf second; sleep 2".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let first = fixture
            .read_process(started.process_id.clone(), ProcessReadCursor::new(0))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let actual = fixture
            .read_process(started.process_id.clone(), first.next_cursor)
            .await
            .unwrap();
        let _ = fixture.kill_process(started.process_id).await;

        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("second"))
        );
    }

    #[tokio::test]
    async fn test_process_list_refreshes_exited_status_without_status_probe() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("exit 7".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let actual = fixture.list_processes().await.unwrap();
        let expected = ProcessStatusKind::Exited { exit_code: Some(7) };

        assert_eq!(actual.first().unwrap().process_id, started.process_id);
        assert_eq!(actual.first().unwrap().status, expected);
    }

    #[tokio::test]
    async fn test_process_read_cursor_does_not_advance_before_entry_is_visible() {
        use tokio::io::AsyncWriteExt;

        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let next_cursor = Arc::new(AtomicU64::new(1));
        let (mut writer, reader) = tokio::io::duplex(64);
        let log_guard = logs.lock().await;

        ForgeCommandExecutorService::capture_stream(
            reader,
            ProcessStream::Stdout,
            logs.clone(),
            next_cursor.clone(),
        );
        writer.write_all(b"pending").await.unwrap();
        writer.flush().await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let actual = next_cursor.load(Ordering::Relaxed);
        let expected = 1;
        assert_eq!(log_guard.len(), 0);
        assert_eq!(actual, expected);
        drop(log_guard);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if !logs.lock().await.is_empty() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_background_process_does_not_block_foreground_shell_execution() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(
                "sleep 1; printf background".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        let actual = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            fixture.execute_command(
                "printf foreground".to_string(),
                PathBuf::new().join("."),
                true,
                None,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let _ = fixture.kill_process(started.process_id).await;

        let expected = "foreground";
        assert_eq!(actual.stdout, expected);
    }

    #[tokio::test]
    async fn test_process_read_after_exit_observes_trailing_output() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("printf final".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        let status = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let status = fixture
                    .process_status(started.process_id.clone())
                    .await
                    .unwrap();
                if !matches!(status.status, ProcessStatusKind::Running) {
                    return status;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let actual = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let output = fixture
                    .read_process(started.process_id.clone(), ProcessReadCursor::new(0))
                    .await
                    .unwrap();
                if output
                    .entries
                    .iter()
                    .any(|entry| entry.content.contains("final"))
                {
                    return output;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        assert_eq!(
            status.status,
            ProcessStatusKind::Exited { exit_code: Some(0) }
        );
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("final"))
        );
    }

    #[tokio::test]
    async fn test_command_executor() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = "echo 'hello world'";
        let dir = ".";

        let actual = fixture
            .execute_command(cmd.to_string(), PathBuf::new().join(dir), false, None)
            .await
            .unwrap();

        let mut expected = CommandOutput {
            stdout: "hello world\n".to_string(),
            stderr: "".to_string(),
            command: "echo \"hello world\"".into(),
            exit_code: Some(0),
        };

        if cfg!(target_os = "windows") {
            expected.stdout = format!("'{}'", expected.stdout);
        }

        assert_eq!(actual.stdout.trim(), expected.stdout.trim());
        assert_eq!(actual.stderr, expected.stderr);
        assert_eq!(actual.success(), expected.success());
    }
    #[tokio::test]
    async fn test_command_executor_with_env_vars_success() {
        // Set up test environment variables
        unsafe {
            std::env::set_var("TEST_ENV_VAR", "test_value");
            std::env::set_var("ANOTHER_TEST_VAR", "another_value");
        }

        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = if cfg!(target_os = "windows") {
            "echo %TEST_ENV_VAR%"
        } else {
            "echo $TEST_ENV_VAR"
        };

        let actual = fixture
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join("."),
                false,
                Some(vec!["TEST_ENV_VAR".to_string()]),
            )
            .await
            .unwrap();

        assert!(actual.success());
        assert!(actual.stdout.contains("test_value"));

        // Clean up
        unsafe {
            std::env::remove_var("TEST_ENV_VAR");
            std::env::remove_var("ANOTHER_TEST_VAR");
        }
    }

    #[tokio::test]
    async fn test_command_executor_with_missing_env_vars() {
        unsafe {
            std::env::remove_var("MISSING_ENV_VAR");
        }

        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = if cfg!(target_os = "windows") {
            "echo %MISSING_ENV_VAR%"
        } else {
            "echo ${MISSING_ENV_VAR:-default_value}"
        };

        let actual = fixture
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join("."),
                false,
                Some(vec!["MISSING_ENV_VAR".to_string()]),
            )
            .await
            .unwrap();

        // Should still succeed even with missing env vars
        assert!(actual.success());
    }

    #[tokio::test]
    async fn test_command_executor_with_empty_env_list() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = "echo 'no env vars'";

        let actual = fixture
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join("."),
                false,
                Some(vec![]),
            )
            .await
            .unwrap();

        assert!(actual.success());
        assert!(actual.stdout.contains("no env vars"));
    }

    #[tokio::test]
    async fn test_command_executor_with_multiple_env_vars() {
        unsafe {
            std::env::set_var("FIRST_VAR", "first");
            std::env::set_var("SECOND_VAR", "second");
        }

        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = if cfg!(target_os = "windows") {
            "echo %FIRST_VAR% %SECOND_VAR%"
        } else {
            "echo $FIRST_VAR $SECOND_VAR"
        };

        let actual = fixture
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join("."),
                false,
                Some(vec!["FIRST_VAR".to_string(), "SECOND_VAR".to_string()]),
            )
            .await
            .unwrap();

        assert!(actual.success());
        assert!(actual.stdout.contains("first"));
        assert!(actual.stdout.contains("second"));

        // Clean up
        unsafe {
            std::env::remove_var("FIRST_VAR");
            std::env::remove_var("SECOND_VAR");
        }
    }

    #[tokio::test]
    async fn test_command_executor_silent() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = "echo 'silent test'";
        let dir = ".";

        let actual = fixture
            .execute_command(cmd.to_string(), PathBuf::new().join(dir), true, None)
            .await
            .unwrap();

        let mut expected = CommandOutput {
            stdout: "silent test\n".to_string(),
            stderr: "".to_string(),
            command: "echo \"silent test\"".into(),
            exit_code: Some(0),
        };

        if cfg!(target_os = "windows") {
            expected.stdout = format!("'{}'", expected.stdout);
        }

        // The output should still be captured in the CommandOutput
        assert_eq!(actual.stdout.trim(), expected.stdout.trim());
        assert_eq!(actual.stderr, expected.stderr);
        assert_eq!(actual.success(), expected.success());
    }

    mod write_lossy_utf8 {
        use pretty_assertions::assert_eq;

        use super::super::write_lossy_utf8;

        fn run(buf: &[u8]) -> (Vec<u8>, Vec<u8>) {
            let mut out = Vec::<u8>::new();
            let pending = write_lossy_utf8(&mut out, buf).unwrap();
            (out, pending)
        }

        #[test]
        fn valid_ascii_passes_through() {
            let (out, pending) = run(b"hello");
            assert_eq!(out, b"hello");
            assert!(pending.is_empty());
        }

        #[test]
        fn valid_multibyte_passes_through() {
            // "héllo ✓" — mixed 2-byte and 3-byte codepoints.
            let input = "héllo ✓".as_bytes();
            let (out, pending) = run(input);
            assert_eq!(out, input);
            assert!(pending.is_empty());
        }

        #[test]
        fn incomplete_trailing_codepoint_is_buffered() {
            // "é" is 0xC3 0xA9 — leading byte alone must be held back.
            let (out, pending) = run(&[b'a', 0xC3]);
            assert_eq!(out, b"a");
            assert_eq!(pending, vec![0xC3]);
        }

        #[test]
        fn multibyte_split_across_two_chunks_emits_once_whole() {
            let mut out = Vec::<u8>::new();
            let pending = write_lossy_utf8(&mut out, &[b'a', 0xC3]).unwrap();
            assert_eq!(pending, vec![0xC3]);
            assert_eq!(out, b"a");

            let mut working = pending;
            working.push(0xA9);
            let pending = write_lossy_utf8(&mut out, &working).unwrap();
            assert!(pending.is_empty());
            assert_eq!(out, "aé".as_bytes());
        }

        #[test]
        fn invalid_byte_in_middle_becomes_replacement() {
            let (out, pending) = run(&[b'a', 0xFF, b'b']);
            assert_eq!(out, "a\u{FFFD}b".as_bytes());
            assert!(pending.is_empty());
        }

        #[test]
        fn lone_continuation_byte_becomes_replacement() {
            let (out, pending) = run(&[b'a', 0x80, b'b']);
            assert_eq!(out, "a\u{FFFD}b".as_bytes());
            assert!(pending.is_empty());
        }

        #[test]
        fn windows_1252_smart_quote_becomes_replacement() {
            // Regression: 0x91/0x92 land as bare continuation bytes and broke
            // console stdio on Windows before this fix.
            let (out, pending) = run(b"quote: \x91hi\x92");
            assert_eq!(out, "quote: \u{FFFD}hi\u{FFFD}".as_bytes());
            assert!(pending.is_empty());
        }
    }
}
