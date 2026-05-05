use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use forge_domain::{
    CodebaseQueryResult, ToolCallContext, ToolCatalog, ToolOutput, resolve_execution_cwd,
};

use crate::fmt::content::FormatContent;
use crate::operation::{TempContentFiles, ToolOperation};
use crate::services::Services;
use crate::{
    AgentRegistry, ConversationService, EnvironmentInfra, FollowUpService, FsPatchService,
    FsReadService, FsRemoveService, FsSearchService, FsUndoService, FsWriteService,
    ImageReadService, NetFetchService, PlanCreateService, ProviderService, ShellExecuteRequest,
    ShellService, SkillFetchService, WorkspaceService,
};

pub struct ToolExecutor<S> {
    services: Arc<S>,
}

fn resolve_tool_execution_cwd(requested_cwd: Option<&PathBuf>, environment_cwd: &Path) -> PathBuf {
    resolve_execution_cwd(requested_cwd, environment_cwd)
}

impl<
    S: FsReadService
        + ImageReadService
        + FsWriteService
        + FsSearchService
        + WorkspaceService
        + NetFetchService
        + FsRemoveService
        + FsPatchService
        + FsUndoService
        + ShellService
        + FollowUpService
        + ConversationService
        + EnvironmentInfra<Config = forge_config::ForgeConfig>
        + PlanCreateService
        + SkillFetchService
        + AgentRegistry
        + ProviderService
        + Services,
> ToolExecutor<S>
{
    pub fn new(services: Arc<S>) -> Self {
        Self { services }
    }

    fn require_prior_read(
        &self,
        context: &ToolCallContext,
        raw_path: &str,
        action: &str,
    ) -> anyhow::Result<()> {
        let target_path = self.normalize_path(raw_path.to_string());
        let has_read = context.with_metrics(|metrics| {
            metrics.files_accessed.contains(&target_path)
                || metrics.files_accessed.contains(raw_path)
        })?;

        if has_read {
            Ok(())
        } else {
            Err(anyhow!(
                "You must read the file with the read tool before attempting to {action}.",
                action = action
            ))
        }
    }

    async fn dump_operation(&self, operation: &ToolOperation) -> anyhow::Result<TempContentFiles> {
        match operation {
            ToolOperation::NetFetch { input: _, output } => {
                let config = self.services.get_config()?;
                let original_length = output.content.len();
                let is_truncated = original_length > config.max_fetch_chars;
                let mut files = TempContentFiles::default();

                if is_truncated {
                    files = files.stdout(
                        self.create_temp_file("forge_fetch_", ".txt", &output.content)
                            .await?,
                    );
                }

                Ok(files)
            }
            ToolOperation::Shell { output } => {
                let config = self.services.get_config()?;
                let stdout_lines = output.output.stdout.lines().count();
                let stderr_lines = output.output.stderr.lines().count();
                let stdout_truncated =
                    stdout_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;
                let stderr_truncated =
                    stderr_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;

                let mut files = TempContentFiles::default();

                if stdout_truncated {
                    files = files.stdout(
                        self.create_temp_file("forge_shell_stdout_", ".txt", &output.output.stdout)
                            .await?,
                    );
                }
                if stderr_truncated {
                    files = files.stderr(
                        self.create_temp_file("forge_shell_stderr_", ".txt", &output.output.stderr)
                            .await?,
                    );
                }

                Ok(files)
            }
            _ => Ok(TempContentFiles::default()),
        }
    }

    /// Converts a path to absolute by joining it with the current working
    /// directory if it's relative
    fn normalize_path(&self, path: String) -> String {
        let env = self.services.get_environment();
        let path_buf = PathBuf::from(&path);

        if path_buf.is_absolute() {
            path
        } else {
            PathBuf::from(&env.cwd).join(path_buf).display().to_string()
        }
    }

    /// Resolves command execution cwd to the same physical path used for
    /// permission checks.
    fn resolve_execution_cwd(&self, cwd: Option<&PathBuf>) -> PathBuf {
        resolve_tool_execution_cwd(cwd, self.services.get_environment().cwd.as_path())
    }

    async fn create_temp_file(
        &self,
        prefix: &str,
        ext: &str,
        content: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let path = tempfile::Builder::new()
            .disable_cleanup(true)
            .prefix(prefix)
            .suffix(ext)
            .tempfile()?
            .into_temp_path()
            .to_path_buf();
        self.services
            .write(
                path.to_string_lossy().to_string(),
                content.to_string(),
                true,
            )
            .await?;
        Ok(path)
    }

    async fn call_internal(
        &self,
        input: ToolCatalog,
        context: &ToolCallContext,
    ) -> anyhow::Result<ToolOperation> {
        Ok(match input {
            ToolCatalog::Read(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .read(
                        normalized_path,
                        input
                            .range
                            .as_ref()
                            .and_then(|r| r.start_line)
                            .map(|i| i as u64),
                        input
                            .range
                            .as_ref()
                            .and_then(|r| r.end_line)
                            .map(|i| i as u64),
                    )
                    .await?;

                (input, output).into()
            }
            ToolCatalog::Write(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .write(normalized_path, input.content.clone(), input.overwrite)
                    .await?;
                (input, output).into()
            }
            ToolCatalog::FsSearch(input) => {
                let mut params = input.clone();
                // Normalize path if provided
                if let Some(ref path) = params.path {
                    params.path = Some(self.normalize_path(path.clone()));
                }
                let output = self.services.search(params).await?;
                (input, output).into()
            }
            ToolCatalog::SemSearch(input) => {
                let config = self.services.get_config()?;
                let env = self.services.get_environment();
                let services = self.services.clone();
                let cwd = env.cwd.clone();
                let limit = config.max_sem_search_results;
                let top_k = config.sem_search_top_k as u32;
                let params: Vec<_> = input
                    .queries
                    .iter()
                    .map(|search_query| {
                        forge_domain::SearchParams::new(&search_query.query, &search_query.use_case)
                            .limit(limit)
                            .top_k(top_k)
                    })
                    .collect();

                // Execute all queries in parallel
                let futures: Vec<_> = params
                    .into_iter()
                    .map(|param| services.query_workspace(cwd.clone(), param))
                    .collect();

                let mut results = futures::future::try_join_all(futures).await?;

                // Deduplicate results across queries
                crate::search_dedup::deduplicate_results(&mut results);

                let output = input
                    .queries
                    .into_iter()
                    .zip(results)
                    .map(|(query, results)| CodebaseQueryResult {
                        query: query.query,
                        use_case: query.use_case,
                        results,
                    })
                    .collect::<Vec<_>>();

                let output = forge_domain::CodebaseSearchResults { queries: output };
                ToolOperation::CodebaseSearch { output }
            }
            ToolCatalog::Remove(input) => {
                let normalized_path = self.normalize_path(input.path.clone());
                let output = self.services.remove(normalized_path).await?;
                (input, output).into()
            }
            ToolCatalog::Patch(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .patch(
                        normalized_path,
                        input.old_string.clone(),
                        input.new_string.clone(),
                        input.replace_all,
                    )
                    .await?;
                (input, output).into()
            }
            ToolCatalog::MultiPatch(input) => {
                let normalized_path = self.normalize_path(input.file_path.clone());
                let output = self
                    .services
                    .multi_patch(normalized_path, input.edits.clone())
                    .await?;
                (input, output).into()
            }
            ToolCatalog::Undo(input) => {
                let normalized_path = self.normalize_path(input.path.clone());
                let output = self.services.undo(normalized_path).await?;
                (input, output).into()
            }
            ToolCatalog::Shell(input) => {
                let execution_cwd = self.resolve_execution_cwd(input.cwd.as_ref());
                let output = self
                    .services
                    .execute(ShellExecuteRequest {
                        command: input.command.clone(),
                        cwd: execution_cwd,
                        keep_ansi: input.keep_ansi,
                        silent: false,
                        env_vars: input.env.clone(),
                        handoff_timeout: input.handoff_timeout_seconds.unwrap_or_default(),
                        description: input.description.clone(),
                    })
                    .await?;
                output.into()
            }
            ToolCatalog::ProcessStart(input) => {
                let execution_cwd = self.resolve_execution_cwd(input.cwd.as_ref());
                let output = self
                    .services
                    .process_start(
                        input.command.clone(),
                        execution_cwd,
                        input.env.clone(),
                        input.description.clone(),
                    )
                    .await?;
                ToolOperation::ProcessStart { output }
            }
            ToolCatalog::ProcessStatus(input) => {
                let output = self
                    .services
                    .process_status(forge_domain::ProcessId::parse(input.process_id.clone())?)
                    .await?;
                ToolOperation::ProcessStatus { output }
            }
            ToolCatalog::ProcessRead(input) => {
                let output = self
                    .services
                    .process_read(
                        forge_domain::ProcessId::parse(input.process_id.clone())?,
                        forge_domain::ProcessReadCursor::new(input.cursor),
                    )
                    .await?;
                ToolOperation::ProcessRead { output }
            }
            ToolCatalog::ProcessList(_input) => {
                let output = self.services.process_list().await?;
                ToolOperation::ProcessList { output }
            }
            ToolCatalog::ProcessKill(input) => {
                let output = self
                    .services
                    .process_kill(forge_domain::ProcessId::parse(input.process_id.clone())?)
                    .await?;
                ToolOperation::ProcessKill { output }
            }
            ToolCatalog::Fetch(input) => {
                let output = self.services.fetch(input.url.clone(), input.raw).await?;
                (input, output).into()
            }
            ToolCatalog::Followup(input) => {
                let output = self
                    .services
                    .follow_up(
                        input.question.clone(),
                        input
                            .option1
                            .clone()
                            .into_iter()
                            .chain(input.option2.clone())
                            .chain(input.option3.clone())
                            .chain(input.option4.clone())
                            .chain(input.option5.clone())
                            .collect(),
                        input.multiple,
                    )
                    .await?;
                output.into()
            }
            ToolCatalog::Plan(input) => {
                let output = self
                    .services
                    .create_plan(
                        input.plan_name.clone(),
                        input.version.clone(),
                        input.content.clone(),
                    )
                    .await?;
                (input, output).into()
            }
            ToolCatalog::Skill(input) => {
                let skill = self.services.fetch_skill(input.name.clone()).await?;
                ToolOperation::Skill { output: skill }
            }
            ToolCatalog::TodoWrite(input) => {
                let before = context.get_todos()?;
                context.update_todos(input.todos.clone())?;
                let after = context.get_todos()?;
                ToolOperation::TodoWrite { before, after }
            }
            ToolCatalog::TodoRead(_input) => {
                let todos = context.get_todos()?;
                ToolOperation::TodoRead { output: todos }
            }
            ToolCatalog::Task(_) => {
                // Task tools are handled in ToolRegistry before reaching here
                unreachable!("Task tool should be handled in ToolRegistry")
            }
        })
    }

    pub async fn execute(
        &self,
        tool_input: ToolCatalog,
        context: &ToolCallContext,
    ) -> anyhow::Result<ToolOutput> {
        let tool_kind = tool_input.kind();
        let env = self.services.get_environment();
        let config = self.services.get_config()?;

        // Enforce read-before-edit for patch operations
        let file_path = match &tool_input {
            ToolCatalog::Patch(input) => Some(&input.file_path),
            ToolCatalog::MultiPatch(input) => Some(&input.file_path),
            _ => None,
        };

        if let Some(path) = file_path {
            self.require_prior_read(context, path, "edit it")?;
        }

        // Enforce read-before-edit for overwrite writes
        if let ToolCatalog::Write(input) = &tool_input
            && input.overwrite
        {
            self.require_prior_read(context, &input.file_path, "overwrite it")?;
        }

        let execution_result = self.call_internal(tool_input.clone(), context).await;

        if let Err(ref error) = execution_result {
            tracing::error!(error = ?error, "Tool execution failed");
        }

        let operation = execution_result?;

        // Send formatted output message
        if let Some(output) = operation.to_content(&env) {
            context.send(output).await?;
        }

        let truncation_path = self.dump_operation(&operation).await?;

        context.with_metrics(|metrics| {
            operation.into_tool_output(tool_kind, truncation_path, &env, &config, metrics)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_domain::{PermissionOperation, ProcessStart, Shell};
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;
    fn create_directory_symlink(physical: &PathBuf, alias: &PathBuf) -> anyhow::Result<()> {
        #[cfg(unix)]
        std::os::unix::fs::symlink(physical, alias)?;
        #[cfg(windows)]
        if let Err(error) = std::os::windows::fs::symlink_dir(physical, alias) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                return Ok(());
            }
            return Err(error.into());
        }
        Ok(())
    }

    fn symlink_fixture() -> anyhow::Result<Option<(tempfile::TempDir, PathBuf, PathBuf)>> {
        let fixture = tempfile::tempdir()?;
        let workspace = fixture.path().join("workspace");
        let physical = fixture.path().join("physical");
        let alias = workspace.join("alias");
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(&physical)?;
        create_directory_symlink(&physical, &alias)?;
        if !alias.exists() {
            return Ok(None);
        }
        let physical = std::fs::canonicalize(&physical)?;
        Ok(Some((fixture, workspace, physical)))
    }

    #[test]
    fn test_shell_execution_cwd_matches_policy_physical_symlink_resolution() -> anyhow::Result<()> {
        let Some((_fixture, workspace, physical)) = symlink_fixture()? else {
            return Ok(());
        };
        let cwd = PathBuf::from("alias");
        let tool = ToolCatalog::Shell(Shell {
            command: "pwd".to_string(),
            cwd: Some(cwd.clone()),
            ..Default::default()
        });

        let actual = resolve_tool_execution_cwd(Some(&cwd), workspace.as_path());
        let expected = match tool.to_policy_operation(workspace).unwrap() {
            PermissionOperation::Execute { cwd, .. } => cwd,
            _ => unreachable!("shell policy operation must be execute"),
        };
        assert_eq!(actual, physical);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_process_start_execution_cwd_matches_policy_physical_symlink_resolution()
    -> anyhow::Result<()> {
        let Some((_fixture, workspace, physical)) = symlink_fixture()? else {
            return Ok(());
        };
        let cwd = PathBuf::from("alias");
        let tool = ToolCatalog::ProcessStart(ProcessStart {
            command: "pwd".to_string(),
            cwd: Some(cwd.clone()),
            ..Default::default()
        });

        let actual = resolve_tool_execution_cwd(Some(&cwd), workspace.as_path());
        let expected = match tool.to_policy_operation(workspace).unwrap() {
            PermissionOperation::Execute { cwd, .. } => cwd,
            _ => unreachable!("process_start policy operation must be execute"),
        };
        assert_eq!(actual, physical);
        assert_eq!(actual, expected);
        Ok(())
    }
}
