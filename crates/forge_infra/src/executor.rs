use std::collections::{HashMap, VecDeque};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bstr::ByteSlice;
use forge_app::CommandInfra;
use forge_domain::{
    CommandExecutionOutput, CommandOutput, ConsoleWriter as OutputPrinterTrait, Environment,
    ProcessId, ProcessLogEntry, ProcessReadCursor, ProcessReadOutput, ProcessStartOutput,
    ProcessStatus, ProcessStatusKind, ProcessStream,
};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::console::StdConsoleWriter;

const MAX_BACKGROUND_LOG_ENTRIES: usize = 8192;
const SHELL_SYNC_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

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
    dropped_before_cursor: Arc<AtomicU64>,
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
}

struct LaunchedProcess {
    process_id: ProcessId,
    child: Child,
    command: String,
    cwd: String,
    logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
    dropped_before_cursor: Arc<AtomicU64>,
    stdout: Arc<Mutex<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
}

impl LaunchedProcess {
    fn start_output(&self) -> ProcessStartOutput {
        ProcessStartOutput {
            process_id: self.process_id.clone(),
            status: ProcessStatusKind::Running,
            command: self.command.clone(),
            cwd: self.cwd.clone(),
        }
    }

    fn into_managed(self) -> ManagedProcess {
        ManagedProcess {
            command: self.command,
            cwd: self.cwd,
            child: self.child,
            status: ProcessStatusKind::Running,
            logs: self.logs,
            dropped_before_cursor: self.dropped_before_cursor,
            stdout_task: self.stdout_task,
            stderr_task: self.stderr_task,
        }
    }
}

impl ManagedProcess {
    async fn refresh_status(&mut self) -> anyhow::Result<()> {
        if matches!(self.status, ProcessStatusKind::Running)
            && let Some(status) = self.child.try_wait()?
        {
            self.status = ProcessStatusKind::Exited { exit_code: status.code() };
            if let Some(stdout_task) = self.stdout_task.take() {
                stdout_task.await?;
            }
            if let Some(stderr_task) = self.stderr_task.take() {
                stderr_task.await?;
            }
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
        #[cfg(unix)]
        command.process_group(0);

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

    fn launch_process(
        &self,
        command: String,
        working_dir: &Path,
        env_vars: Option<Vec<String>>,
        silent: bool,
    ) -> anyhow::Result<LaunchedProcess> {
        let mut prepared_command = self.prepare_command(&command, working_dir, env_vars);
        prepared_command.stdin(std::process::Stdio::null());
        let mut child = prepared_command.spawn()?;
        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let next_cursor = Arc::new(AtomicU64::new(1));
        let dropped_before_cursor = Arc::new(AtomicU64::new(0));
        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(Vec::new()));

        let stdout_task = child.stdout.take().map(|stdout_pipe| {
            Self::capture_stream(
                stdout_pipe,
                ProcessStream::Stdout,
                logs.clone(),
                next_cursor.clone(),
                dropped_before_cursor.clone(),
                stdout.clone(),
                if silent {
                    OutputSink::silent()
                } else {
                    OutputSink::stdout(self.output_printer.clone())
                },
            )
        });
        let stderr_task = child.stderr.take().map(|stderr_pipe| {
            Self::capture_stream(
                stderr_pipe,
                ProcessStream::Stderr,
                logs.clone(),
                next_cursor,
                dropped_before_cursor.clone(),
                stderr.clone(),
                if silent {
                    OutputSink::silent()
                } else {
                    OutputSink::stderr(self.output_printer.clone())
                },
            )
        });

        Ok(LaunchedProcess {
            process_id: Self::next_process_id(),
            child,
            command,
            cwd: working_dir.display().to_string(),
            logs,
            dropped_before_cursor,
            stdout,
            stderr,
            stdout_task,
            stderr_task,
        })
    }

    fn capture_stream<A, W>(
        mut reader: A,
        stream: ProcessStream,
        logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
        next_cursor: Arc<AtomicU64>,
        dropped_before_cursor: Arc<AtomicU64>,
        output: Arc<Mutex<Vec<u8>>>,
        mut writer: W,
    ) -> tokio::task::JoinHandle<()>
    where
        A: AsyncReadExt + Unpin + Send + 'static,
        W: Write + Send + 'static,
    {
        tokio::spawn(async move {
            let mut buffer = [0; 1024];
            let mut pending = Vec::<u8>::new();
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
                output.lock().await.extend_from_slice(bytes);
                let content = bytes.to_str_lossy().into_owned();
                let mut logs = logs.lock().await;
                let cursor = next_cursor.fetch_add(1, Ordering::Relaxed);
                logs.push_back(ProcessLogEntry {
                    cursor: ProcessReadCursor::new(cursor),
                    stream,
                    content,
                });
                trim_background_logs(&mut logs, &dropped_before_cursor);
                drop(logs);

                let mut working = std::mem::take(&mut pending);
                working.extend_from_slice(bytes);
                pending = match write_lossy_utf8(&mut writer, &working) {
                    Ok(pending) => pending,
                    Err(_) => break,
                };
                let _ = writer.flush();
            }

            if !pending.is_empty() {
                let _ = writer.write_all(pending.to_str_lossy().as_bytes());
                let _ = writer.flush();
            }
        })
    }

    async fn snapshot_output(output: &Arc<Mutex<Vec<u8>>>) -> String {
        output.lock().await.to_str_lossy().into_owned()
    }

    async fn register_launched_process(&self, launched: LaunchedProcess) {
        self.processes
            .lock()
            .await
            .insert(launched.process_id.clone(), launched.into_managed());
    }

    async fn execute_command_internal(
        &self,
        command: String,
        working_dir: &Path,
        silent: bool,
        env_vars: Option<Vec<String>>,
    ) -> anyhow::Result<CommandExecutionOutput> {
        let _ready = self.ready.lock().await;
        let mut launched = self.launch_process(command, working_dir, env_vars, silent)?;
        let exit = tokio::time::timeout(SHELL_SYNC_STARTUP_TIMEOUT, launched.child.wait()).await;

        match exit {
            Ok(status) => {
                let status = status?;
                if let Some(stdout_task) = launched.stdout_task.take() {
                    stdout_task.await?;
                }
                if let Some(stderr_task) = launched.stderr_task.take() {
                    stderr_task.await?;
                }
                let stdout = Self::snapshot_output(&launched.stdout).await;
                let stderr = Self::snapshot_output(&launched.stderr).await;

                if !silent && !stdout.ends_with('\n') && !stdout.is_empty() {
                    let _ = self.output_printer.write(b"\n");
                    let _ = self.output_printer.flush();
                }

                Ok(CommandExecutionOutput {
                    output: CommandOutput {
                        stdout,
                        stderr,
                        exit_code: status.code(),
                        command: launched.command,
                    },
                    process: None,
                })
            }
            Err(_) => {
                let process = launched.start_output();
                let stdout = Self::snapshot_output(&launched.stdout).await;
                let captured_stderr = Self::snapshot_output(&launched.stderr).await;
                let stderr = format!(
                    "{captured_stderr}Command exceeded the 2 second synchronous shell window and is running as managed background process {process_id}. Use process_status and process_read with this process_id to observe it.",
                    process_id = process.process_id
                );
                let command = launched.command.clone();
                self.register_launched_process(launched).await;

                Ok(CommandExecutionOutput {
                    output: CommandOutput { stdout, stderr, exit_code: None, command },
                    process: Some(process),
                })
            }
        }
    }
}

enum OutputSink {
    Console(OutputPrinterWriter),
    Silent(io::Sink),
}

impl OutputSink {
    fn stdout(printer: Arc<StdConsoleWriter>) -> Self {
        Self::Console(OutputPrinterWriter::stdout(printer))
    }

    fn stderr(printer: Arc<StdConsoleWriter>) -> Self {
        Self::Console(OutputPrinterWriter::stderr(printer))
    }

    fn silent() -> Self {
        Self::Silent(io::sink())
    }
}

impl Write for OutputSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Console(writer) => writer.write(buf),
            Self::Silent(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Console(writer) => writer.flush(),
            Self::Silent(writer) => writer.flush(),
        }
    }
}

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

fn trim_background_logs(logs: &mut VecDeque<ProcessLogEntry>, dropped_before_cursor: &AtomicU64) {
    while logs.len() > MAX_BACKGROUND_LOG_ENTRIES {
        if let Some(dropped) = logs.pop_front() {
            dropped_before_cursor.store(dropped.cursor.get(), Ordering::Relaxed);
        }
    }
}

#[cfg(unix)]
async fn kill_child_process_group(child: &mut Child) -> anyhow::Result<()> {
    let pid = child
        .id()
        .ok_or_else(|| anyhow::anyhow!("Managed process has no live pid"))?;
    let process_group_id = i32::try_from(pid)?;
    let kill_target = process_group_id
        .checked_neg()
        .ok_or_else(|| anyhow::anyhow!("Invalid managed process group id: {process_group_id}"))?;
    // SAFETY: The child was created with `process_group(0)`, so its process
    // group id is its pid. Passing the negative process group id to POSIX
    // `kill` targets that group instead of an unrelated process.
    let result = unsafe { libc::kill(kill_target, libc::SIGKILL) };
    if result == -1 {
        child.kill().await?;
    } else {
        let _ = child.wait().await?;
    }
    Ok(())
}

#[cfg(not(unix))]
async fn kill_child_process_group(child: &mut Child) -> anyhow::Result<()> {
    child.kill().await?;
    Ok(())
}

fn process_read_output_from_logs(
    process_id: ProcessId,
    cursor: ProcessReadCursor,
    logs: &VecDeque<ProcessLogEntry>,
    dropped_before_cursor: Option<ProcessReadCursor>,
) -> ProcessReadOutput {
    let first_available_cursor = logs.front().map(|entry| entry.cursor);
    let mut entries: Vec<_> = logs
        .iter()
        .filter(|entry| entry.cursor > cursor)
        .cloned()
        .collect();
    entries.sort_by_key(|entry| entry.cursor);
    let next = entries.last().map(|entry| entry.cursor).unwrap_or(cursor);
    ProcessReadOutput {
        process_id,
        next_cursor: next,
        first_available_cursor,
        dropped_before_cursor,
        entries,
    }
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
    ) -> anyhow::Result<CommandExecutionOutput> {
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
        let launched = self.launch_process(command, &working_dir, env_vars, false)?;
        let output = launched.start_output();
        self.register_launched_process(launched).await;
        Ok(output)
    }

    async fn process_status(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(&process_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"))?;
        process.refresh_status().await?;
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
        let logs = process.logs.lock().await;
        let dropped_before_cursor = match process.dropped_before_cursor.load(Ordering::Relaxed) {
            0 => None,
            cursor => Some(ProcessReadCursor::new(cursor)),
        };
        Ok(process_read_output_from_logs(
            process_id,
            cursor,
            &logs,
            dropped_before_cursor,
        ))
    }

    async fn list_processes(&self) -> anyhow::Result<Vec<ProcessStatus>> {
        let mut processes = self.processes.lock().await;
        let mut statuses = Vec::with_capacity(processes.len());
        for (process_id, process) in processes.iter_mut() {
            process.refresh_status().await?;
            statuses.push(process.status(process_id.clone()));
        }
        Ok(statuses)
    }

    async fn kill_process(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(&process_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"))?;
        process.refresh_status().await?;
        if matches!(process.status, ProcessStatusKind::Running) {
            kill_child_process_group(&mut process.child).await?;
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

    #[test]
    fn test_process_read_exposes_dropped_background_log_range() {
        let mut fixture = VecDeque::new();
        let dropped_before_cursor = AtomicU64::new(0);
        let last_cursor = u64::try_from(MAX_BACKGROUND_LOG_ENTRIES).unwrap() + 1;
        for cursor in 1..=last_cursor {
            fixture.push_back(ProcessLogEntry {
                cursor: ProcessReadCursor::new(cursor),
                stream: ProcessStream::Stdout,
                content: format!("entry-{cursor}"),
            });
        }

        trim_background_logs(&mut fixture, &dropped_before_cursor);
        let actual = process_read_output_from_logs(
            ProcessId::new("process-overflow"),
            ProcessReadCursor::new(0),
            &fixture,
            Some(ProcessReadCursor::new(
                dropped_before_cursor.load(Ordering::Relaxed),
            )),
        );

        assert_eq!(
            actual.first_available_cursor,
            Some(ProcessReadCursor::new(2))
        );
        assert_eq!(
            actual.dropped_before_cursor,
            Some(ProcessReadCursor::new(1))
        );
        assert_eq!(actual.entries.len(), MAX_BACKGROUND_LOG_ENTRIES);
        assert_eq!(
            actual.entries.first().unwrap().cursor,
            ProcessReadCursor::new(2)
        );
        assert_eq!(actual.next_cursor, ProcessReadCursor::new(last_cursor));
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
    async fn test_process_kill_refreshes_naturally_exited_status_before_killing() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("exit 9".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let actual = fixture.kill_process(started.process_id).await.unwrap();
        let expected = ProcessStatusKind::Exited { exit_code: Some(9) };

        assert_eq!(actual.status, expected);
    }

    #[tokio::test]
    async fn test_process_read_cursor_does_not_advance_before_entry_is_visible() {
        use tokio::io::AsyncWriteExt;

        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let next_cursor = Arc::new(AtomicU64::new(1));
        let dropped_before_cursor = Arc::new(AtomicU64::new(0));
        let (mut writer, reader) = tokio::io::duplex(64);
        let log_guard = logs.lock().await;

        let output = Arc::new(Mutex::new(Vec::new()));
        let stream_task = ForgeCommandExecutorService::capture_stream(
            reader,
            ProcessStream::Stdout,
            logs.clone(),
            next_cursor.clone(),
            dropped_before_cursor,
            output,
            OutputSink::silent(),
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
        drop(writer);
        stream_task.await.unwrap();
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
        assert_eq!(actual.output.stdout, expected);
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
    async fn test_process_kill_terminates_descendant_background_jobs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let marker_path = temp_dir.path().join("leaked-descendant");
        let command = format!(
            "(sleep 0.3; printf leaked > {}) & wait",
            marker_path.display()
        );
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(command, temp_dir.path().to_path_buf(), None)
            .await
            .unwrap();

        fixture.kill_process(started.process_id).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;

        assert!(!marker_path.exists());
    }

    #[tokio::test]
    async fn test_execute_command_timeout_handoff_preserves_single_process_side_effects() {
        let temp_dir = tempfile::tempdir().unwrap();
        let marker_path = temp_dir.path().join("side-effect-count");
        let command = format!(
            "printf run >> {marker}; printf early-output; sleep 5",
            marker = marker_path.display()
        );
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());

        let actual = fixture
            .execute_command(command, temp_dir.path().to_path_buf(), true, None)
            .await
            .unwrap();
        let process = actual
            .process
            .clone()
            .expect("long-running command should be handed off");
        let read_output = fixture
            .read_process(process.process_id.clone(), ProcessReadCursor::new(0))
            .await
            .unwrap();
        let _ = fixture.kill_process(process.process_id).await;
        let side_effects = std::fs::read_to_string(marker_path).unwrap();

        assert_eq!(actual.output.stdout, "early-output");
        assert_eq!(actual.output.exit_code, None);
        assert!(
            read_output
                .entries
                .iter()
                .any(|entry| entry.content.contains("early-output"))
        );
        assert_eq!(side_effects, "run");
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

        assert_eq!(actual.output.stdout.trim(), expected.stdout.trim());
        assert_eq!(actual.output.stderr, expected.stderr);
        assert_eq!(actual.output.success(), expected.success());
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

        assert!(actual.output.success());
        assert!(actual.output.stdout.contains("test_value"));

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
        assert!(actual.output.success());
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

        assert!(actual.output.success());
        assert!(actual.output.stdout.contains("no env vars"));
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

        assert!(actual.output.success());
        assert!(actual.output.stdout.contains("first"));
        assert!(actual.output.stdout.contains("second"));

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
        assert_eq!(actual.output.stdout.trim(), expected.stdout.trim());
        assert_eq!(actual.output.stderr, expected.stderr);
        assert_eq!(actual.output.success(), expected.success());
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
