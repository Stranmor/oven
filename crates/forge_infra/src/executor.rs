use std::collections::{HashMap, VecDeque};
use std::io::{self, Stderr, Stdout, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bstr::ByteSlice;
use forge_app::CommandInfra;
use forge_domain::{
    CommandExecutionOutput, CommandOutput, ConsoleWriter as OutputPrinterTrait, Environment,
    ProcessId, ProcessLogEntry, ProcessObservationWaitSeconds, ProcessReadCursor,
    ProcessReadOutput, ProcessStartOutput, ProcessStatus, ProcessStatusKind, ProcessStream,
    ShellHandoffTimeoutSeconds,
};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify};

use crate::console::StdConsoleWriter;

const MAX_BACKGROUND_LOG_ENTRIES: usize = 8192;
const MAX_COMPLETED_PROCESS_ARCHIVE: usize = 128;
const PROCESS_OBSERVATION_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Service for executing shell commands
pub struct ForgeCommandExecutorService<O = Stdout, E = Stderr> {
    env: Environment,
    output_printer: Arc<StdConsoleWriter<O, E>>,

    // Mutex to ensure that only one command is executed at a time
    ready: Arc<Mutex<()>>,
    processes: Arc<Mutex<HashMap<ProcessId, ManagedProcess>>>,
    completed_processes: Arc<Mutex<CompletedProcessArchive>>,
}

struct CompletedProcessArchive {
    entries: HashMap<ProcessId, CompletedProcess>,
    order: VecDeque<ProcessId>,
}

struct CompletedProcess {
    command: String,
    cwd: String,
    status: ProcessStatusKind,
    logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
    dropped_before_cursor: Arc<AtomicU64>,
}

impl CompletedProcess {
    fn status(&self, process_id: ProcessId) -> ProcessStatus {
        ProcessStatus {
            process_id,
            status: self.status.clone(),
            command: self.command.clone(),
            cwd: self.cwd.clone(),
        }
    }
}

impl CompletedProcessArchive {
    fn new() -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new() }
    }

    fn insert(&mut self, process_id: ProcessId, process: CompletedProcess) {
        if !self.entries.contains_key(&process_id) {
            self.order.push_back(process_id.clone());
        }
        self.entries.insert(process_id.clone(), process);
        while self.order.len() > MAX_COMPLETED_PROCESS_ARCHIVE {
            if let Some(expired) = self.order.pop_front() {
                self.entries.remove(&expired);
            }
        }
    }

    fn get(&self, process_id: &ProcessId) -> Option<&CompletedProcess> {
        self.entries.get(process_id)
    }
}

struct ManagedProcess {
    command: String,
    cwd: String,
    child: Child,
    process_group_id: ProcessGroupId,
    status: ProcessStatusKind,
    logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
    dropped_before_cursor: Arc<AtomicU64>,
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
    output_finalizing: bool,
    output_finalized: bool,
    output_finalized_notify: Arc<Notify>,
}

#[derive(Clone, Copy)]
struct ProcessGroupId(u32);

impl ProcessGroupId {
    fn from_child(child: &Child) -> anyhow::Result<Self> {
        child
            .id()
            .map(Self)
            .ok_or_else(|| anyhow::anyhow!("Managed process has no live pid"))
    }
}

struct LaunchedProcess {
    process_id: ProcessId,
    child: Child,
    command: String,
    cwd: String,
    logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
    dropped_before_cursor: Arc<AtomicU64>,
    stdout: OutputCapture,
    stderr: OutputCapture,
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
    output_mirror: OutputMirrorControl,
}

#[derive(Clone, Copy)]
enum LaunchCaptureMode {
    ForegroundSnapshot,
    ManagedOnly,
}

impl LaunchCaptureMode {
    fn output_capture(self) -> OutputCapture {
        match self {
            Self::ForegroundSnapshot => OutputCapture::snapshot(),
            Self::ManagedOnly => OutputCapture::managed(),
        }
    }
}

#[derive(Clone)]
struct OutputCapture {
    mode: Arc<OutputCaptureMode>,
}

enum OutputCaptureMode {
    Snapshot {
        bytes: Mutex<Vec<u8>>,
        enabled: AtomicBool,
    },
    Managed,
}

impl OutputCapture {
    fn snapshot() -> Self {
        Self {
            mode: Arc::new(OutputCaptureMode::Snapshot {
                bytes: Mutex::new(Vec::new()),
                enabled: AtomicBool::new(true),
            }),
        }
    }

    fn managed() -> Self {
        Self { mode: Arc::new(OutputCaptureMode::Managed) }
    }

    async fn disable_snapshot_capture(&self) {
        if let OutputCaptureMode::Snapshot { bytes, enabled } = self.mode.as_ref() {
            enabled.store(false, Ordering::Relaxed);
            drop(bytes.lock().await);
        }
    }

    async fn push(&self, bytes: &[u8]) {
        if let OutputCaptureMode::Snapshot { bytes: output, enabled } = self.mode.as_ref() {
            let mut output = output.lock().await;
            if enabled.load(Ordering::Relaxed) {
                output.extend_from_slice(bytes);
            }
        }
    }

    #[cfg(test)]
    fn is_managed(&self) -> bool {
        matches!(self.mode.as_ref(), OutputCaptureMode::Managed)
    }

    #[cfg(test)]
    async fn captured_len(&self) -> usize {
        match self.mode.as_ref() {
            OutputCaptureMode::Snapshot { bytes, .. } => bytes.lock().await.len(),
            OutputCaptureMode::Managed => 0,
        }
    }

    async fn snapshot_output(&self) -> String {
        match self.mode.as_ref() {
            OutputCaptureMode::Snapshot { bytes, .. } => {
                bytes.lock().await.to_str_lossy().into_owned()
            }
            OutputCaptureMode::Managed => String::new(),
        }
    }
}

impl LaunchedProcess {
    async fn disable_foreground_capture(&self) {
        self.stdout.disable_snapshot_capture().await;
        self.stderr.disable_snapshot_capture().await;
    }
    fn disable_output_mirroring(&self) {
        self.output_mirror.disable();
    }

    fn start_output(&self) -> ProcessStartOutput {
        ProcessStartOutput {
            process_id: self.process_id.clone(),
            status: ProcessStatusKind::Running,
            command: self.command.clone(),
            cwd: self.cwd.clone(),
        }
    }

    fn into_managed(self) -> anyhow::Result<ManagedProcess> {
        let process_group_id = ProcessGroupId::from_child(&self.child)?;
        Ok(ManagedProcess {
            command: self.command,
            cwd: self.cwd,
            child: self.child,
            process_group_id,
            status: ProcessStatusKind::Running,
            logs: self.logs,
            dropped_before_cursor: self.dropped_before_cursor,
            stdout_task: self.stdout_task,
            stderr_task: self.stderr_task,
            output_finalizing: false,
            output_finalized: false,
            output_finalized_notify: Arc::new(Notify::new()),
        })
    }
}

impl ManagedProcess {
    async fn refresh_status_without_output_join(&mut self) -> anyhow::Result<()> {
        if matches!(self.status, ProcessStatusKind::Running)
            && let Some(status) = self.child.try_wait()?
        {
            self.status = ProcessStatusKind::Exited { exit_code: status.code() };
        }
        Ok(())
    }

    fn take_output_tasks(
        &mut self,
    ) -> (
        Option<tokio::task::JoinHandle<()>>,
        Option<tokio::task::JoinHandle<()>>,
    ) {
        (self.stdout_task.take(), self.stderr_task.take())
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

impl<O, E> Clone for ForgeCommandExecutorService<O, E> {
    fn clone(&self) -> Self {
        Self {
            env: self.env.clone(),
            output_printer: self.output_printer.clone(),
            ready: self.ready.clone(),
            processes: self.processes.clone(),
            completed_processes: self.completed_processes.clone(),
        }
    }
}

impl<O, E> ForgeCommandExecutorService<O, E>
where
    O: Write + Send + 'static,
    E: Write + Send + 'static,
{
    pub fn new(env: Environment, output_printer: Arc<StdConsoleWriter<O, E>>) -> Self {
        Self {
            env,
            output_printer,
            ready: Arc::new(Mutex::new(())),
            processes: Arc::new(Mutex::new(HashMap::new())),
            completed_processes: Arc::new(Mutex::new(CompletedProcessArchive::new())),
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
        capture_mode: LaunchCaptureMode,
    ) -> anyhow::Result<LaunchedProcess> {
        let mut prepared_command = self.prepare_command(&command, working_dir, env_vars);
        prepared_command.stdin(std::process::Stdio::null());
        let mut child = prepared_command.spawn()?;
        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let next_cursor = Arc::new(AtomicU64::new(1));
        let dropped_before_cursor = Arc::new(AtomicU64::new(0));
        let stdout = capture_mode.output_capture();
        let stderr = capture_mode.output_capture();

        let output_mirror = OutputMirrorControl::new(!silent);
        let stdout_task = child.stdout.take().map(|stdout_pipe| {
            Self::capture_stream(
                stdout_pipe,
                ProcessStream::Stdout,
                logs.clone(),
                next_cursor.clone(),
                dropped_before_cursor.clone(),
                stdout.clone(),
                OutputSink::stdout(self.output_printer.clone(), output_mirror.clone()),
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
                OutputSink::stderr(self.output_printer.clone(), output_mirror.clone()),
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
            output_mirror,
        })
    }

    fn capture_stream<A, W>(
        mut reader: A,
        stream: ProcessStream,
        logs: Arc<Mutex<VecDeque<ProcessLogEntry>>>,
        next_cursor: Arc<AtomicU64>,
        dropped_before_cursor: Arc<AtomicU64>,
        output: OutputCapture,
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
                output.push(bytes).await;
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

    async fn snapshot_output(output: &OutputCapture) -> String {
        output.snapshot_output().await
    }

    async fn register_launched_process(&self, launched: LaunchedProcess) -> anyhow::Result<()> {
        let process_id = launched.process_id.clone();
        let managed = launched.into_managed()?;
        self.processes.lock().await.insert(process_id, managed);
        Ok(())
    }

    async fn archive_completed_process(&self, process_id: ProcessId, process: ManagedProcess) {
        let completed = CompletedProcess {
            command: process.command,
            cwd: process.cwd,
            status: process.status,
            logs: process.logs,
            dropped_before_cursor: process.dropped_before_cursor,
        };
        self.completed_processes
            .lock()
            .await
            .insert(process_id, completed);
    }

    async fn archived_status(&self, process_id: &ProcessId) -> Option<ProcessStatus> {
        self.completed_processes
            .lock()
            .await
            .get(process_id)
            .map(|process| process.status(process_id.clone()))
    }

    async fn archived_read_output(
        &self,
        process_id: &ProcessId,
        cursor: ProcessReadCursor,
    ) -> Option<(ProcessReadOutput, ProcessStatusKind)> {
        let (logs, dropped_before_cursor, status) = {
            let archive = self.completed_processes.lock().await;
            let process = archive.get(process_id)?;
            let dropped_before_cursor = match process.dropped_before_cursor.load(Ordering::Relaxed)
            {
                0 => None,
                cursor => Some(ProcessReadCursor::new(cursor)),
            };
            (
                Arc::clone(&process.logs),
                dropped_before_cursor,
                process.status.clone(),
            )
        };
        let logs = logs.lock().await;
        Some((
            process_read_output_from_logs(process_id.clone(), cursor, &logs, dropped_before_cursor),
            status,
        ))
    }

    async fn finalize_and_archive_process(
        &self,
        process_id: ProcessId,
        stdout_task: Option<tokio::task::JoinHandle<()>>,
        stderr_task: Option<tokio::task::JoinHandle<()>>,
    ) -> anyhow::Result<()> {
        finalize_output_tasks(stdout_task, stderr_task).await?;
        let process = {
            let mut processes = self.processes.lock().await;
            if let Some(process) = processes.get_mut(&process_id) {
                process.output_finalized = true;
                process.output_finalized_notify.notify_waiters();
            }
            processes.remove(&process_id)
        };
        if let Some(process) = process {
            self.archive_completed_process(process_id, process).await;
        }
        Ok(())
    }

    async fn execute_command_internal(
        &self,
        command: String,
        working_dir: &Path,
        silent: bool,
        env_vars: Option<Vec<String>>,
        handoff_timeout: ShellHandoffTimeoutSeconds,
    ) -> anyhow::Result<CommandExecutionOutput> {
        let _guard = self.ready.lock().await;
        let mut launched = self.launch_process(
            command,
            working_dir,
            env_vars,
            silent,
            LaunchCaptureMode::ForegroundSnapshot,
        )?;
        let status = tokio::time::timeout(handoff_timeout.duration(), launched.child.wait()).await;
        match status {
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
                launched.disable_output_mirroring();
                let process = launched.start_output();
                launched.disable_foreground_capture().await;
                let stdout = Self::snapshot_output(&launched.stdout).await;
                let captured_stderr = Self::snapshot_output(&launched.stderr).await;
                let stderr = format!(
                    "{captured_stderr}Command exceeded the {handoff_timeout} second synchronous shell window and is running as managed background process {process_id}. Use process_status and process_read with this process_id to observe it.",
                    process_id = process.process_id
                );
                let command = launched.command.clone();
                self.register_launched_process(launched).await?;

                Ok(CommandExecutionOutput {
                    output: CommandOutput { stdout, stderr, exit_code: None, command },
                    process: Some(process),
                })
            }
        }
    }

    async fn process_status_once(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let (status, stdout_task, stderr_task) = loop {
            let mut processes = self.processes.lock().await;
            let Some(process) = processes.get_mut(&process_id) else {
                drop(processes);
                return self
                    .archived_status(&process_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"));
            };
            process.refresh_status_without_output_join().await?;
            if matches!(process.status, ProcessStatusKind::Running) {
                return Ok(process.status(process_id));
            }
            if process.output_finalized {
                break (process.status(process_id), None, None);
            }
            if process.output_finalizing {
                let notify = Arc::clone(&process.output_finalized_notify);
                let notified = notify.notified();
                drop(processes);
                notified.await;
                continue;
            }
            process.output_finalizing = true;
            let tasks = process.take_output_tasks();
            break (process.status(process_id), tasks.0, tasks.1);
        };
        self.finalize_and_archive_process(status.process_id.clone(), stdout_task, stderr_task)
            .await?;
        Ok(status)
    }

    async fn read_process_once(
        &self,
        process_id: ProcessId,
        cursor: ProcessReadCursor,
    ) -> anyhow::Result<(ProcessReadOutput, ProcessStatusKind)> {
        let (
            logs,
            dropped_before_cursor,
            status,
            stdout_task,
            stderr_task,
            output_finalized_notify,
        ) = loop {
            let mut processes = self.processes.lock().await;
            let Some(process) = processes.get_mut(&process_id) else {
                drop(processes);
                if let Some((output, status)) = self.archived_read_output(&process_id, cursor).await
                {
                    return Ok((output, status));
                }
                anyhow::bail!("Unknown process id: {process_id}");
            };
            process.refresh_status_without_output_join().await?;
            let dropped_before_cursor = match process.dropped_before_cursor.load(Ordering::Relaxed)
            {
                0 => None,
                cursor => Some(ProcessReadCursor::new(cursor)),
            };
            if matches!(process.status, ProcessStatusKind::Running) || process.output_finalized {
                break (
                    Arc::clone(&process.logs),
                    dropped_before_cursor,
                    process.status.clone(),
                    None,
                    None,
                    None,
                );
            }
            if process.output_finalizing {
                let notify = Arc::clone(&process.output_finalized_notify);
                let notified = notify.notified();
                drop(processes);
                notified.await;
                continue;
            }
            process.output_finalizing = true;
            let tasks = process.take_output_tasks();
            break (
                Arc::clone(&process.logs),
                dropped_before_cursor,
                process.status.clone(),
                tasks.0,
                tasks.1,
                Some(Arc::clone(&process.output_finalized_notify)),
            );
        };
        if output_finalized_notify.is_some() {
            self.finalize_and_archive_process(process_id.clone(), stdout_task, stderr_task)
                .await?;
        } else {
            finalize_output_tasks(stdout_task, stderr_task).await?;
        }
        let logs = logs.lock().await;
        Ok((
            process_read_output_from_logs(process_id, cursor, &logs, dropped_before_cursor),
            status,
        ))
    }
    async fn sleep_until_next_observation(deadline: Instant) {
        let now = Instant::now();
        if deadline > now {
            tokio::time::sleep(std::cmp::min(
                PROCESS_OBSERVATION_POLL_INTERVAL,
                deadline.duration_since(now),
            ))
            .await;
        }
    }
}
enum OutputSink<O = Stdout, E = Stderr> {
    Console(OutputPrinterWriter<O, E>, OutputMirrorControl),
}

#[derive(Clone)]
struct OutputMirrorControl {
    enabled: Arc<AtomicBool>,
}

impl OutputMirrorControl {
    fn new(enabled: bool) -> Self {
        Self { enabled: Arc::new(AtomicBool::new(enabled)) }
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

impl<O, E> OutputSink<O, E>
where
    O: Write + Send,
    E: Write + Send,
{
    fn stdout(printer: Arc<StdConsoleWriter<O, E>>, mirror: OutputMirrorControl) -> Self {
        Self::Console(OutputPrinterWriter::stdout(printer), mirror)
    }

    fn stderr(printer: Arc<StdConsoleWriter<O, E>>, mirror: OutputMirrorControl) -> Self {
        Self::Console(OutputPrinterWriter::stderr(printer), mirror)
    }
}

impl<O, E> Write for OutputSink<O, E>
where
    O: Write + Send,
    E: Write + Send,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Console(writer, mirror) if mirror.enabled() => writer.write(buf),
            Self::Console(_, _) => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Console(writer, mirror) if mirror.enabled() => writer.flush(),
            Self::Console(_, _) => Ok(()),
        }
    }
}

struct OutputPrinterWriter<O = Stdout, E = Stderr> {
    printer: Arc<StdConsoleWriter<O, E>>,
    is_stdout: bool,
}

impl<O, E> OutputPrinterWriter<O, E> {
    fn stdout(printer: Arc<StdConsoleWriter<O, E>>) -> Self {
        Self { printer, is_stdout: true }
    }

    fn stderr(printer: Arc<StdConsoleWriter<O, E>>) -> Self {
        Self { printer, is_stdout: false }
    }
}

impl<O, E> Write for OutputPrinterWriter<O, E>
where
    O: Write + Send,
    E: Write + Send,
{
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

async fn join_output_tasks(
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
) -> anyhow::Result<()> {
    if let Some(stdout_task) = stdout_task {
        stdout_task.await?;
    }
    if let Some(stderr_task) = stderr_task {
        stderr_task.await?;
    }
    Ok(())
}

async fn finalize_output_tasks(
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
) -> anyhow::Result<()> {
    join_output_tasks(stdout_task, stderr_task).await
}

#[cfg(unix)]
async fn kill_child_process_group(
    child: &mut Child,
    process_group_id: ProcessGroupId,
) -> anyhow::Result<()> {
    let process_group_id = i32::try_from(process_group_id.0)?;
    let kill_target = process_group_id
        .checked_neg()
        .ok_or_else(|| anyhow::anyhow!("Invalid managed process group id: {process_group_id}"))?;
    // SAFETY: The child was created with `process_group(0)`, so its process
    // group id is its pid. Passing the negative process group id to POSIX
    // `kill` targets that group instead of an unrelated process. The group id
    // is captured at launch so descendants remain targetable after parent exit.
    let result = unsafe { libc::kill(kill_target, libc::SIGKILL) };
    if result == -1 && (child.try_wait()?).is_none() {
        child.kill().await?;
    }
    let _ = child.wait().await?;
    Ok(())
}

#[cfg(not(unix))]
async fn kill_child_process_group(
    child: &mut Child,
    _process_group_id: ProcessGroupId,
) -> anyhow::Result<()> {
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
impl<O, E> CommandInfra for ForgeCommandExecutorService<O, E>
where
    O: Write + Send + 'static,
    E: Write + Send + 'static,
{
    async fn execute_command(
        &self,
        command: String,
        working_dir: PathBuf,
        silent: bool,
        env_vars: Option<Vec<String>>,
        handoff_timeout: ShellHandoffTimeoutSeconds,
    ) -> anyhow::Result<CommandExecutionOutput> {
        self.execute_command_internal(command, &working_dir, silent, env_vars, handoff_timeout)
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
        let launched = self.launch_process(
            command,
            &working_dir,
            env_vars,
            true,
            LaunchCaptureMode::ManagedOnly,
        )?;
        let output = launched.start_output();
        self.register_launched_process(launched).await?;
        Ok(output)
    }

    async fn process_status(
        &self,
        process_id: ProcessId,
        wait: Option<ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessStatus> {
        let deadline = wait.and_then(|wait| Instant::now().checked_add(wait.duration()));
        loop {
            let status = self.process_status_once(process_id.clone()).await?;
            if !matches!(status.status, ProcessStatusKind::Running) {
                return Ok(status);
            }
            let Some(deadline) = deadline else {
                return Ok(status);
            };
            if Instant::now() >= deadline {
                return Ok(status);
            }
            Self::sleep_until_next_observation(deadline).await;
        }
    }

    async fn read_process(
        &self,
        process_id: ProcessId,
        cursor: ProcessReadCursor,
        wait: Option<ProcessObservationWaitSeconds>,
    ) -> anyhow::Result<ProcessReadOutput> {
        let deadline = wait.and_then(|wait| Instant::now().checked_add(wait.duration()));
        loop {
            let (output, status) = self.read_process_once(process_id.clone(), cursor).await?;
            if !output.entries.is_empty() || !matches!(status, ProcessStatusKind::Running) {
                return Ok(output);
            }
            let Some(deadline) = deadline else {
                return Ok(output);
            };
            if Instant::now() >= deadline {
                return Ok(output);
            }
            Self::sleep_until_next_observation(deadline).await;
        }
    }

    async fn list_processes(&self) -> anyhow::Result<Vec<ProcessStatus>> {
        let mut terminal_processes = Vec::new();
        let statuses = {
            let mut processes = self.processes.lock().await;
            let mut statuses = Vec::with_capacity(processes.len());
            for (process_id, process) in processes.iter_mut() {
                process.refresh_status_without_output_join().await?;
                if matches!(process.status, ProcessStatusKind::Running) {
                    statuses.push(process.status(process_id.clone()));
                } else if !process.output_finalizing {
                    process.output_finalizing = true;
                    let (stdout_task, stderr_task) = process.take_output_tasks();
                    terminal_processes.push((process_id.clone(), stdout_task, stderr_task));
                }
            }
            statuses
        };
        for (process_id, stdout_task, stderr_task) in terminal_processes {
            self.finalize_and_archive_process(process_id, stdout_task, stderr_task)
                .await?;
        }
        Ok(statuses)
    }

    async fn kill_process(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
        let (status, stdout_task, stderr_task) = loop {
            let mut processes = self.processes.lock().await;
            let Some(process) = processes.get_mut(&process_id) else {
                drop(processes);
                return self
                    .archived_status(&process_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Unknown process id: {process_id}"));
            };
            process.refresh_status_without_output_join().await?;
            if process.output_finalized {
                let status = process.status(process_id.clone());
                break (status, None, None);
            }
            if process.output_finalizing {
                let notify = Arc::clone(&process.output_finalized_notify);
                let notified = notify.notified();
                drop(processes);
                notified.await;
                continue;
            }
            if !matches!(process.status, ProcessStatusKind::Killed) {
                let status_before_kill = process.status.clone();
                kill_child_process_group(&mut process.child, process.process_group_id).await?;
                process.refresh_status_without_output_join().await?;
                if matches!(status_before_kill, ProcessStatusKind::Running) {
                    process.status = ProcessStatusKind::Killed;
                }
            }
            let status = process.status(process_id.clone());
            process.output_finalizing = true;
            let (stdout_task, stderr_task) = process.take_output_tasks();
            break (status, stdout_task, stderr_task);
        };
        self.finalize_and_archive_process(status.process_id.clone(), stdout_task, stderr_task)
            .await?;
        Ok(status)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Mutex as StdMutex;

    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Clone)]
    struct CapturedWriter(Arc<StdMutex<Vec<u8>>>);

    impl CapturedWriter {
        fn new() -> Self {
            Self(Arc::new(StdMutex::new(Vec::new())))
        }

        fn output(&self) -> String {
            self.0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_slice()
                .to_str_lossy()
                .into_owned()
        }
    }

    impl Write for CapturedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn captured_printer() -> (
        Arc<StdConsoleWriter<CapturedWriter, CapturedWriter>>,
        CapturedWriter,
        CapturedWriter,
    ) {
        let stdout = CapturedWriter::new();
        let stderr = CapturedWriter::new();
        let printer = Arc::new(StdConsoleWriter::with_writers(
            stdout.clone(),
            stderr.clone(),
        ));
        (printer, stdout, stderr)
    }

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
    async fn test_process_start_captures_output_without_console_mirroring() {
        let (printer, stdout, stderr) = captured_printer();
        let fixture = ForgeCommandExecutorService::new(test_env(), printer);
        let started = fixture
            .start_process(
                "printf hidden-stdout; printf hidden-stderr >&2".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        let actual = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let output = fixture
                    .read_process(started.process_id.clone(), ProcessReadCursor::new(0), None)
                    .await
                    .unwrap();
                if output.entries.len() >= 2 {
                    return output;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let expected = (String::new(), String::new());

        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("hidden-stdout"))
        );
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("hidden-stderr"))
        );
        assert_eq!((stdout.output(), stderr.output()), expected);
    }

    #[tokio::test]
    async fn test_shell_timeout_handoff_disables_later_console_mirroring() {
        let (printer, stdout, stderr) = captured_printer();
        let fixture = ForgeCommandExecutorService::new(test_env(), printer);

        let execution = fixture
            .execute_command(
                "printf before-handoff; sleep 1.05; printf after-handoff; printf after-handoff-err >&2; sleep 2".to_string(),
                PathBuf::new().join("."),
                false,
                None,
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let process = execution
            .process
            .clone()
            .expect("long-running command should be handed off");
        let initial_output = fixture
            .read_process(process.process_id.clone(), ProcessReadCursor::new(0), None)
            .await
            .unwrap();
        let actual = fixture
            .read_process(
                process.process_id.clone(),
                initial_output.next_cursor,
                Some(ProcessObservationWaitSeconds::new(3).unwrap()),
            )
            .await
            .unwrap();
        let _ = fixture.kill_process(process.process_id).await;
        let expected_stdout = "before-handoff".to_string();
        let expected_stderr = String::new();

        assert_eq!(stdout.output(), expected_stdout);
        assert_eq!(stderr.output(), expected_stderr);
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("after-handoff"))
        );
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("after-handoff-err"))
        );
    }

    #[tokio::test]
    async fn test_process_start_uses_managed_capture_without_snapshot_buffer() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let mut launched = fixture
            .launch_process(
                "printf managed-output; sleep 1".to_string(),
                &PathBuf::new().join("."),
                None,
                true,
                LaunchCaptureMode::ManagedOnly,
            )
            .unwrap();

        let actual = (launched.stdout.is_managed(), launched.stderr.is_managed());
        let expected = (true, true);

        assert_eq!(actual, expected);
        let _ = launched.child.kill().await;
        let _ = join_output_tasks(launched.stdout_task.take(), launched.stderr_task.take()).await;
    }

    #[tokio::test]
    async fn test_disabled_snapshot_capture_does_not_grow_after_handoff() {
        let fixture = OutputCapture::snapshot();
        fixture.push(b"before").await;
        fixture.disable_snapshot_capture().await;
        let before_late_output = fixture.captured_len().await;

        fixture.push(b"after").await;
        let actual = fixture.captured_len().await;
        let expected = before_late_output;

        assert_eq!(actual, expected);
        assert_eq!(fixture.snapshot_output().await, "before");
    }

    #[tokio::test]
    async fn test_disabled_snapshot_capture_blocks_in_flight_push_after_flag_check() {
        let fixture = OutputCapture::snapshot();
        let OutputCaptureMode::Snapshot { bytes, enabled } = fixture.mode.as_ref() else {
            panic!("fixture should use snapshot capture");
        };
        let guard = bytes.lock().await;
        let push_fixture = fixture.clone();
        let push_task = tokio::spawn(async move {
            push_fixture.push(b"late").await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        enabled.store(false, Ordering::Relaxed);
        drop(guard);
        push_task.await.unwrap();
        let actual = fixture.captured_len().await;
        let expected = 0;

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_shell_handoff_stops_snapshot_capture_after_timeout() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());

        let actual = fixture
            .execute_command(
                "printf before-snapshot; sleep 1.05; printf after-snapshot; sleep 2".to_string(),
                PathBuf::new().join("."),
                true,
                None,
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let process = actual
            .process
            .clone()
            .expect("long-running command should be handed off");
        let initial_output = fixture
            .read_process(process.process_id.clone(), ProcessReadCursor::new(0), None)
            .await
            .unwrap();
        let process_output = fixture
            .read_process(
                process.process_id.clone(),
                initial_output.next_cursor,
                Some(ProcessObservationWaitSeconds::new(3).unwrap()),
            )
            .await
            .unwrap();
        let _ = fixture.kill_process(process.process_id).await;
        let expected = "before-snapshot";

        assert_eq!(actual.output.stdout, expected);
        assert!(
            process_output
                .entries
                .iter()
                .any(|entry| entry.content.contains("after-snapshot"))
        );
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
            .read_process(actual.process_id.clone(), ProcessReadCursor::new(0), None)
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
            .read_process(started.process_id.clone(), ProcessReadCursor::new(0), None)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let actual = fixture
            .read_process(started.process_id.clone(), first.next_cursor, None)
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
    async fn test_process_status_wait_does_not_block_process_read() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(
                "sleep 0.2; printf concurrent; sleep 1".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();
        let process_id = started.process_id.clone();
        let status_fixture = fixture.clone();

        let status_task = tokio::spawn(async move {
            status_fixture
                .process_status(
                    process_id,
                    Some(ProcessObservationWaitSeconds::new(1).unwrap()),
                )
                .await
                .unwrap()
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let actual = fixture
            .read_process(
                started.process_id.clone(),
                ProcessReadCursor::new(0),
                Some(ProcessObservationWaitSeconds::new(1).unwrap()),
            )
            .await
            .unwrap();
        let status = status_task.await.unwrap();
        let _ = fixture.kill_process(started.process_id).await;
        let expected = ProcessStatusKind::Running;

        assert_eq!(status.status, expected);
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("concurrent"))
        );
    }

    #[tokio::test]
    async fn test_process_read_wait_returns_delayed_output_without_external_sleep() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(
                "sleep 0.2; printf delayed; sleep 1".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        let actual = fixture
            .read_process(
                started.process_id.clone(),
                ProcessReadCursor::new(0),
                Some(ProcessObservationWaitSeconds::new(1).unwrap()),
            )
            .await
            .unwrap();
        let _ = fixture.kill_process(started.process_id).await;

        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("delayed"))
        );
    }

    #[tokio::test]
    async fn test_process_read_wait_timeout_preserves_cursor_without_output() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("sleep 2".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        let actual = fixture
            .read_process(
                started.process_id.clone(),
                ProcessReadCursor::new(7),
                Some(ProcessObservationWaitSeconds::new(1).unwrap()),
            )
            .await
            .unwrap();
        let _ = fixture.kill_process(started.process_id).await;
        let expected = ProcessReadOutput {
            process_id: actual.process_id.clone(),
            next_cursor: ProcessReadCursor::new(7),
            first_available_cursor: None,
            dropped_before_cursor: None,
            entries: Vec::new(),
        };

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_process_status_wait_returns_early_when_process_exits() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("sleep 0.2".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        let actual = fixture
            .process_status(
                started.process_id,
                Some(ProcessObservationWaitSeconds::new(1).unwrap()),
            )
            .await
            .unwrap();
        let expected = ProcessStatusKind::Exited { exit_code: Some(0) };

        assert_eq!(actual.status, expected);
    }

    #[test]
    fn test_process_observation_wait_rejects_unbounded_runtime_value() {
        let fixture = 31;

        let actual = ProcessObservationWaitSeconds::new(fixture);

        assert!(actual.is_err());
    }

    #[test]
    fn test_process_observation_wait_accepts_upper_bound() {
        let fixture = 30;

        let actual = ProcessObservationWaitSeconds::new(fixture).unwrap();
        let expected = 30;

        assert_eq!(actual.seconds(), expected);
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
    async fn test_process_list_omits_exited_process_and_archives_status() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("exit 7".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let list = fixture.list_processes().await.unwrap();
        let actual = fixture
            .process_status(started.process_id, None)
            .await
            .unwrap();
        let expected = ProcessStatusKind::Exited { exit_code: Some(7) };

        assert_eq!(list, Vec::new());
        assert_eq!(actual.status, expected);
    }

    #[tokio::test]
    async fn test_process_read_returns_archived_output_after_process_leaves_list() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(
                "printf archived".to_string(),
                PathBuf::new().join("."),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let list = fixture.list_processes().await.unwrap();
        let actual = fixture
            .read_process(started.process_id, ProcessReadCursor::new(0), None)
            .await
            .unwrap();

        assert_eq!(list, Vec::new());
        assert!(
            actual
                .entries
                .iter()
                .any(|entry| entry.content.contains("archived"))
        );
    }

    #[tokio::test]
    async fn test_process_kill_archives_process_after_cleanup() {
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

        let output = OutputCapture::snapshot();
        let stream_task = ForgeCommandExecutorService::<Stdout, Stderr>::capture_stream(
            reader,
            ProcessStream::Stdout,
            logs.clone(),
            next_cursor.clone(),
            dropped_before_cursor,
            output,
            io::sink(),
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
    async fn test_process_read_wait_survives_concurrent_kill() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("sleep 5".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();
        let process_id = started.process_id.clone();
        let read_fixture = fixture.clone();

        let read_task = tokio::spawn(async move {
            read_fixture
                .read_process(
                    process_id,
                    ProcessReadCursor::new(0),
                    Some(ProcessObservationWaitSeconds::new(1).unwrap()),
                )
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let kill_status = fixture.kill_process(started.process_id).await.unwrap();
        let actual = read_task.await.unwrap();

        assert_eq!(kill_status.status, ProcessStatusKind::Killed);
        assert!(actual.is_ok());
    }

    #[tokio::test]
    async fn test_process_kill_waits_for_in_progress_output_finalization() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process("exit 0".to_string(), PathBuf::new().join("."), None)
            .await
            .unwrap();
        {
            let mut processes = fixture.processes.lock().await;
            let process = processes.get_mut(&started.process_id).unwrap();
            process.status = ProcessStatusKind::Exited { exit_code: Some(0) };
            process.output_finalizing = true;
            process.stdout_task = None;
            process.stderr_task = None;
        }
        let process_id = started.process_id.clone();
        let kill_fixture = fixture.clone();

        let actual = tokio::spawn(async move { kill_fixture.kill_process(process_id).await });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let expected = false;

        assert_eq!(actual.is_finished(), expected);
        {
            let mut processes = fixture.processes.lock().await;
            let process = processes.get_mut(&started.process_id).unwrap();
            process.output_finalized = true;
            process.output_finalized_notify.notify_waiters();
        }
        actual.await.unwrap().unwrap();
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
                Default::default(),
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
                    .process_status(started.process_id.clone(), None)
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
                    .read_process(started.process_id.clone(), ProcessReadCursor::new(0), None)
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
    async fn test_process_kill_after_parent_exit_terminates_descendant_group() {
        let temp_dir = tempfile::tempdir().unwrap();
        let marker_path = temp_dir.path().join("late-descendant");
        let command = format!(
            "(sleep 0.3; printf leaked > {}) & sleep 0.05; wait",
            marker_path.display()
        );
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let started = fixture
            .start_process(command, temp_dir.path().to_path_buf(), None)
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let actual = fixture.kill_process(started.process_id).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        let expected = ProcessStatusKind::Killed;

        assert_eq!(actual.status, expected);
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
            .execute_command(
                command,
                temp_dir.path().to_path_buf(),
                true,
                None,
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let process = actual
            .process
            .clone()
            .expect("long-running command should be handed off");
        let read_output = fixture
            .read_process(process.process_id.clone(), ProcessReadCursor::new(0), None)
            .await
            .unwrap();
        let _ = fixture.kill_process(process.process_id).await;
        let side_effects = std::fs::read_to_string(marker_path).unwrap();

        assert_eq!(actual.output.stdout, "early-output");
        assert_eq!(actual.output.exit_code, None);
        assert!(
            actual
                .output
                .stderr
                .contains("exceeded the 1 second synchronous shell window")
        );
        assert!(
            read_output
                .entries
                .iter()
                .any(|entry| entry.content.contains("early-output"))
        );
        assert_eq!(side_effects, "run");
    }

    #[tokio::test]
    async fn test_execute_command_custom_handoff_timeout_is_reported_and_managed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let marker_path = temp_dir.path().join("custom-timeout-side-effect");
        let command = format!(
            "printf custom-run >> {marker}; printf custom-early; sleep 5",
            marker = marker_path.display()
        );
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());

        let actual = fixture
            .execute_command(
                command,
                temp_dir.path().to_path_buf(),
                true,
                None,
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let process = actual
            .process
            .clone()
            .expect("custom timeout should hand off long command");
        let read_output = fixture
            .read_process(process.process_id.clone(), ProcessReadCursor::new(0), None)
            .await
            .unwrap();
        let _ = fixture.kill_process(process.process_id).await;
        let side_effects = std::fs::read_to_string(marker_path).unwrap();

        assert_eq!(actual.output.stdout, "custom-early");
        assert_eq!(actual.output.exit_code, None);
        assert!(
            actual
                .output
                .stderr
                .contains("exceeded the 1 second synchronous shell window")
        );
        assert!(
            read_output
                .entries
                .iter()
                .any(|entry| entry.content.contains("custom-early"))
        );
        assert_eq!(side_effects, "custom-run");
    }

    #[tokio::test]
    async fn test_execute_command_completes_before_timeout_without_managed_entry() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());

        let actual = fixture
            .execute_command(
                "printf immediate".to_string(),
                PathBuf::new().join("."),
                true,
                None,
                ShellHandoffTimeoutSeconds::new(1).unwrap(),
            )
            .await
            .unwrap();
        let processes = fixture.list_processes().await.unwrap();
        let expected = ("immediate".to_string(), None, Vec::new());

        assert_eq!((actual.output.stdout, actual.process, processes), expected);
    }

    #[tokio::test]
    async fn test_execute_command_default_handoff_timeout_is_fifteen_seconds() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());

        let actual = fixture
            .execute_command(
                "printf default-timeout; sleep 1".to_string(),
                PathBuf::new().join("."),
                true,
                None,
                Default::default(),
            )
            .await
            .unwrap();
        let processes = fixture.list_processes().await.unwrap();
        let expected = ("default-timeout".to_string(), Some(0), None, Vec::new());

        assert_eq!(
            (
                actual.output.stdout,
                actual.output.exit_code,
                actual.process,
                processes,
            ),
            expected
        );
    }

    #[tokio::test]
    async fn test_command_executor() {
        let fixture = ForgeCommandExecutorService::new(test_env(), test_printer());
        let cmd = "echo 'hello world'";
        let dir = ".";

        let actual = fixture
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join(dir),
                false,
                None,
                Default::default(),
            )
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
                Default::default(),
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
                Default::default(),
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
                Default::default(),
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
                Default::default(),
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
            .execute_command(
                cmd.to_string(),
                PathBuf::new().join(dir),
                true,
                None,
                Default::default(),
            )
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
