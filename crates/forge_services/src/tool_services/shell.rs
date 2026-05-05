use std::path::PathBuf;
use std::sync::Arc;

use anyhow::bail;
use bstr::ByteSlice;
use forge_app::domain::Environment;
use forge_app::{
    CommandInfra, EnvironmentInfra, ProcessKillServiceOutput, ProcessOutput,
    ProcessReadServiceOutput, ProcessStartServiceOutput, ShellExecuteRequest, ShellOutput,
    ShellService,
};
use forge_domain::{ProcessId, ProcessReadCursor};
use strip_ansi_escapes::strip;

// Strips out the ansi codes from content.
fn strip_ansi(content: String) -> String {
    strip(content.as_bytes()).to_str_lossy().into_owned()
}

/// Prevents potentially harmful operations like absolute path execution and
/// directory changes. Use for file system interaction, running utilities,
/// installing packages, or executing build commands. For operations requiring
/// unrestricted access, advise users to run forge CLI with '-u' flag. Returns
/// complete output including stdout, stderr, and exit code for diagnostic
/// purposes. Commands that are still running after the synchronous startup
/// window return the already-started managed process handle; use process_status
/// and process_read to poll that same process without re-running the command.
pub struct ForgeShell<I> {
    env: Environment,
    infra: Arc<I>,
}

impl<I: EnvironmentInfra> ForgeShell<I> {
    /// Create a new Shell with environment configuration
    pub fn new(infra: Arc<I>) -> Self {
        let env = infra.get_environment();
        Self { env, infra }
    }

    fn validate_command(command: &str) -> anyhow::Result<()> {
        if command.trim().is_empty() {
            bail!("Command string is empty or contains only whitespace");
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl<I: CommandInfra + EnvironmentInfra> ShellService for ForgeShell<I> {
    async fn execute(&self, request: ShellExecuteRequest) -> anyhow::Result<ShellOutput> {
        let ShellExecuteRequest {
            command,
            cwd,
            keep_ansi,
            silent,
            env_vars,
            handoff_timeout,
            description,
        } = request;
        Self::validate_command(&command)?;

        let execution = self
            .infra
            .execute_command(command, cwd, silent, env_vars, handoff_timeout)
            .await?;
        let mut output = execution.output;
        let process = execution.process;

        if !keep_ansi {
            output.stdout = strip_ansi(output.stdout);
            output.stderr = strip_ansi(output.stderr);
        }

        Ok(ShellOutput { output, shell: self.env.shell.clone(), description, process })
    }

    async fn process_start(
        &self,
        command: String,
        cwd: PathBuf,
        env_vars: Option<Vec<String>>,
        description: Option<String>,
    ) -> anyhow::Result<ProcessStartServiceOutput> {
        Self::validate_command(&command)?;
        let output = self.infra.start_process(command, cwd, env_vars).await?;
        Ok(ProcessStartServiceOutput { shell: self.env.shell.clone(), description, output })
    }

    async fn process_status(&self, process_id: ProcessId) -> anyhow::Result<ProcessOutput> {
        let status = self.infra.process_status(process_id).await?;
        Ok(ProcessOutput { shell: self.env.shell.clone(), description: None, status })
    }

    async fn process_read(
        &self,
        process_id: ProcessId,
        cursor: ProcessReadCursor,
    ) -> anyhow::Result<ProcessReadServiceOutput> {
        let output = self.infra.read_process(process_id, cursor).await?;
        Ok(ProcessReadServiceOutput { shell: self.env.shell.clone(), output })
    }

    async fn process_list(&self) -> anyhow::Result<Vec<forge_domain::ProcessStatus>> {
        self.infra.list_processes().await
    }

    async fn process_kill(
        &self,
        process_id: ProcessId,
    ) -> anyhow::Result<ProcessKillServiceOutput> {
        let status = self.infra.kill_process(process_id).await?;
        Ok(ProcessKillServiceOutput { shell: self.env.shell.clone(), description: None, status })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use forge_app::domain::{CommandExecutionOutput, CommandOutput, Environment};
    use forge_app::{CommandInfra, EnvironmentInfra, ShellService};
    use forge_domain::{
        ConfigOperation, ProcessId, ProcessReadCursor, ProcessReadOutput, ProcessStartOutput,
        ProcessStatus, ProcessStatusKind,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    enum MockExecutionMode {
        Immediate,
        Pending,
    }

    struct MockCommandInfra {
        expected_env_vars: Option<Vec<String>>,
        execution_mode: MockExecutionMode,
        process_id: ProcessId,
        side_effect_count: AtomicUsize,
    }

    impl MockCommandInfra {
        fn immediate(expected_env_vars: Option<Vec<String>>) -> Self {
            Self {
                expected_env_vars,
                execution_mode: MockExecutionMode::Immediate,
                process_id: ProcessId::new("process-test"),
                side_effect_count: AtomicUsize::new(0),
            }
        }

        fn pending(expected_env_vars: Option<Vec<String>>) -> Self {
            Self {
                expected_env_vars,
                execution_mode: MockExecutionMode::Pending,
                process_id: ProcessId::new("process-test"),
                side_effect_count: AtomicUsize::new(0),
            }
        }

        fn pending_with_ansi_process_id() -> Self {
            Self {
                expected_env_vars: None,
                execution_mode: MockExecutionMode::Pending,
                process_id: ProcessId::new("\u{1b}[31mprocess-test\u{1b}[0m"),
                side_effect_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl CommandInfra for MockCommandInfra {
        async fn execute_command(
            &self,
            command: String,
            _working_dir: PathBuf,
            _silent: bool,
            env_vars: Option<Vec<String>>,
            _handoff_timeout: forge_domain::ShellHandoffTimeoutSeconds,
        ) -> anyhow::Result<CommandExecutionOutput> {
            assert_eq!(env_vars, self.expected_env_vars);
            match self.execution_mode {
                MockExecutionMode::Immediate => Ok(CommandExecutionOutput {
                    output: CommandOutput {
                        stdout: "Mock output".to_string(),
                        stderr: String::new(),
                        command,
                        exit_code: Some(0),
                    },
                    process: None,
                }),
                MockExecutionMode::Pending => {
                    let process_command = command.clone();
                    Ok(CommandExecutionOutput {
                        output: CommandOutput {
                            stdout: "early stdout".to_string(),
                            stderr: format!(
                                "early stderr\nCommand exceeded the 2 second synchronous shell window and is running as managed background process {process_id}. Use process_status and process_read with this process_id to observe it.",
                                process_id = self.process_id
                            ),
                            command,
                            exit_code: None,
                        },
                        process: Some(ProcessStartOutput {
                            process_id: self.process_id.clone(),
                            status: ProcessStatusKind::Running,
                            command: process_command,
                            cwd: ".".to_string(),
                        }),
                    })
                }
            }
        }

        async fn execute_command_raw(
            &self,
            _command: &str,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> anyhow::Result<std::process::ExitStatus> {
            unimplemented!()
        }

        async fn start_process(
            &self,
            command: String,
            working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> anyhow::Result<ProcessStartOutput> {
            self.side_effect_count.fetch_add(1, Ordering::SeqCst);
            Ok(ProcessStartOutput {
                process_id: self.process_id.clone(),
                status: ProcessStatusKind::Running,
                command,
                cwd: working_dir.display().to_string(),
            })
        }

        async fn process_status(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
            Ok(ProcessStatus {
                process_id,
                status: ProcessStatusKind::Running,
                command: "sleep 60".to_string(),
                cwd: ".".to_string(),
            })
        }

        async fn read_process(
            &self,
            process_id: ProcessId,
            cursor: ProcessReadCursor,
        ) -> anyhow::Result<ProcessReadOutput> {
            Ok(ProcessReadOutput {
                process_id,
                next_cursor: cursor,
                first_available_cursor: None,
                dropped_before_cursor: None,
                entries: Vec::new(),
            })
        }

        async fn list_processes(&self) -> anyhow::Result<Vec<ProcessStatus>> {
            Ok(Vec::new())
        }

        async fn kill_process(&self, process_id: ProcessId) -> anyhow::Result<ProcessStatus> {
            Ok(ProcessStatus {
                process_id,
                status: ProcessStatusKind::Killed,
                command: "sleep 60".to_string(),
                cwd: ".".to_string(),
            })
        }
    }

    impl EnvironmentInfra for MockCommandInfra {
        type Config = forge_config::ForgeConfig;

        fn get_environment(&self) -> Environment {
            use fake::{Fake, Faker};
            Faker.fake()
        }

        fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(&self, _ops: Vec<ConfigOperation>) -> anyhow::Result<()> {
            unimplemented!()
        }

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
            std::collections::BTreeMap::new()
        }
    }

    #[tokio::test]
    async fn test_shell_service_starts_background_process() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(None)));

        let actual = fixture
            .process_start(
                "sleep 60".to_string(),
                PathBuf::from("."),
                None,
                Some("Starts test process".to_string()),
            )
            .await
            .unwrap();

        assert_eq!(actual.output.process_id, ProcessId::new("process-test"));
        assert_eq!(actual.output.status, ProcessStatusKind::Running);
        assert_eq!(actual.description, Some("Starts test process".to_string()));
    }

    fn execute_request(
        command: impl Into<String>,
        env_vars: Option<Vec<String>>,
        description: Option<String>,
    ) -> ShellExecuteRequest {
        ShellExecuteRequest {
            command: command.into(),
            cwd: PathBuf::from("."),
            keep_ansi: false,
            silent: false,
            env_vars,
            handoff_timeout: forge_domain::ShellHandoffTimeoutSeconds::default(),
            description,
        }
    }

    #[tokio::test]
    async fn test_shell_service_rejects_empty_background_command() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(None)));

        let actual = fixture
            .process_start("   ".to_string(), PathBuf::from("."), None, None)
            .await;

        assert!(actual.is_err());
    }

    #[tokio::test]
    async fn test_shell_service_forwards_env_vars() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(Some(vec![
            "PATH".to_string(),
            "HOME".to_string(),
        ]))));

        let actual = fixture
            .execute(execute_request(
                "echo hello",
                Some(vec!["PATH".to_string(), "HOME".to_string()]),
                None,
            ))
            .await
            .unwrap();

        assert_eq!(actual.output.stdout, "Mock output");
        assert_eq!(actual.output.exit_code, Some(0));
        assert_eq!(actual.process, None);
    }

    #[tokio::test]
    async fn test_shell_service_forwards_no_env_vars() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(None)));

        let actual = fixture
            .execute(execute_request("echo hello", None, None))
            .await
            .unwrap();

        assert_eq!(actual.output.stdout, "Mock output");
        assert_eq!(actual.output.exit_code, Some(0));
        assert_eq!(actual.process, None);
    }

    #[tokio::test]
    async fn test_shell_service_forwards_empty_env_vars() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(Some(vec![]))));

        let actual = fixture
            .execute(execute_request("echo hello", Some(vec![]), None))
            .await
            .unwrap();

        assert_eq!(actual.output.stdout, "Mock output");
        assert_eq!(actual.output.exit_code, Some(0));
        assert_eq!(actual.process, None);
    }

    #[tokio::test]
    async fn test_shell_service_with_description() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(None)));

        let actual = fixture
            .execute(execute_request(
                "echo hello",
                None,
                Some("Prints hello to stdout".to_string()),
            ))
            .await
            .unwrap();

        assert_eq!(actual.output.stdout, "Mock output");
        assert_eq!(actual.output.exit_code, Some(0));
        assert_eq!(actual.process, None);
        assert_eq!(
            actual.description,
            Some("Prints hello to stdout".to_string())
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_shell_service_hands_off_timeout_to_managed_process() {
        let infra = Arc::new(MockCommandInfra::pending(None));
        let fixture = ForgeShell::new(infra.clone());

        let actual = fixture
            .execute(execute_request(
                "sleep 60",
                None,
                Some("Starts a long command".to_string()),
            ))
            .await
            .unwrap();

        let process = actual
            .process
            .expect("timeout handoff should start process");
        assert_eq!(actual.output.command, "sleep 60");
        assert_eq!(actual.output.stdout, "early stdout");
        assert_eq!(actual.output.exit_code, None);
        assert!(
            actual
                .output
                .stderr
                .contains("managed background process process-test")
        );
        assert_eq!(process.process_id, ProcessId::new("process-test"));
        assert_eq!(process.status, ProcessStatusKind::Running);
        assert_eq!(process.cwd, ".");
        assert_eq!(infra.side_effect_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_shell_service_handoff_strips_ansi_from_process_id() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::pending_with_ansi_process_id()));

        let actual = fixture
            .execute(execute_request(
                "sleep 60",
                None,
                Some("Starts a long command".to_string()),
            ))
            .await
            .unwrap();

        let process = actual
            .process
            .expect("timeout handoff should start process");
        assert_eq!(
            process.process_id,
            ProcessId::new("\u{1b}[31mprocess-test\u{1b}[0m")
        );
        assert_eq!(actual.output.command, "sleep 60");
        assert_eq!(actual.output.exit_code, None);
        assert_eq!(actual.output.stdout, "early stdout");
        assert!(
            actual
                .output
                .stderr
                .contains("managed background process process-test")
        );
        assert!(!actual.output.stderr.contains("\u{1b}[31m"));
        assert_eq!(process.status, ProcessStatusKind::Running);
        assert_eq!(process.cwd, ".");
    }

    #[tokio::test]
    async fn test_shell_service_without_description() {
        let fixture = ForgeShell::new(Arc::new(MockCommandInfra::immediate(None)));

        let actual = fixture
            .execute(execute_request("echo hello", None, None))
            .await
            .unwrap();

        assert_eq!(actual.output.stdout, "Mock output");
        assert_eq!(actual.output.exit_code, Some(0));
        assert_eq!(actual.process, None);
        assert_eq!(actual.description, None);
    }
}
