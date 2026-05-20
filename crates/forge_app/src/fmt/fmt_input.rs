use std::path::{Path, PathBuf};

use forge_domain::{
    ChatResponseContent, Environment, ProcessRead, ProcessStatusInput, TitleFormat, ToolCatalog,
};

use crate::fmt::content::FormatContent;
use crate::utils::format_display_path;

fn format_process_status_subtitle(input: &ProcessStatusInput) -> String {
    match input.wait_seconds {
        Some(wait_seconds) => format!("{} (wait_seconds: {wait_seconds})", input.process_id),
        None => input.process_id.clone(),
    }
}

fn format_process_read_subtitle(input: &ProcessRead) -> String {
    let mut parts = vec![
        input.process_id.clone(),
        format!("cursor: {}", input.cursor),
    ];

    if let Some(wait_seconds) = input.wait_seconds {
        parts.push(format!("wait_seconds: {wait_seconds}"));
    }

    parts.join(" · ")
}

fn format_task_subtitle(agent_id: &str, tasks: &[String]) -> String {
    if tasks.is_empty() {
        agent_id.to_string()
    } else {
        format!("{}\n{}", agent_id, tasks.join("\n"))
    }
}

impl FormatContent for ToolCatalog {
    fn to_content(&self, env: &Environment) -> Option<ChatResponseContent> {
        let display_path_for = |path: &str| format_display_path(Path::new(path), env.cwd.as_path());

        match self {
            ToolCatalog::Read(input) => {
                let display_path = display_path_for(&input.file_path);
                let is_explicit_range = input.range.is_some();
                let mut subtitle = display_path;
                if is_explicit_range && let Some(range) = &input.range {
                    match (range.start_line, range.end_line) {
                        (Some(start), Some(end)) => {
                            subtitle.push_str(&format!(":{start}-{end}"));
                        }
                        (Some(start), None) => {
                            subtitle.push_str(&format!(":{start}"));
                        }
                        (None, Some(end)) => {
                            subtitle.push_str(&format!(":1-{end}"));
                        }
                        (None, None) => {}
                    }
                };
                Some(TitleFormat::debug("Read").sub_title(subtitle).into())
            }
            ToolCatalog::Write(input) => {
                let path = PathBuf::from(&input.file_path);
                let display_path = display_path_for(&input.file_path);
                let title = match (path.exists(), input.overwrite) {
                    (true, true) => "Overwrite",
                    (true, false) => {
                        // Case: file exists but overwrite is false then we throw error from tool,
                        // so it's good idea to not print anything on CLI.
                        return None;
                    }
                    (false, _) => "Create",
                };
                Some(TitleFormat::debug(title).sub_title(display_path).into())
            }
            ToolCatalog::FsSearch(input) => {
                let formatted_dir = input.path.as_deref().unwrap_or(".");
                let formatted_dir = display_path_for(formatted_dir);

                let title = match (&input.glob, &input.file_type) {
                    (Some(glob), _) => {
                        format!(
                            "Search for '{}' in '{}' files at {}",
                            input.pattern, glob, formatted_dir
                        )
                    }
                    (None, Some(file_type)) => {
                        format!(
                            "Search for '{}' in {} files at {}",
                            input.pattern, file_type, formatted_dir
                        )
                    }
                    (None, None) => {
                        format!("Search for '{}' at {}", input.pattern, formatted_dir)
                    }
                };
                Some(TitleFormat::debug(title).into())
            }
            ToolCatalog::SemSearch(input) => {
                let pairs: Vec<_> = input
                    .queries
                    .iter()
                    .map(|item| item.query.as_str())
                    .collect();
                Some(
                    TitleFormat::debug("Codebase Search")
                        .sub_title(format!("[{}]", pairs.join(" · ")))
                        .into(),
                )
            }
            ToolCatalog::WorkspaceVectorIndexBuildContinuation(input) => Some(
                TitleFormat::debug("Workspace Vector Index Build Continuation")
                    .sub_title(display_path_for(
                        &input.workspace_path.display().to_string(),
                    ))
                    .into(),
            ),
            ToolCatalog::Remove(input) => {
                let display_path = display_path_for(&input.path);
                Some(TitleFormat::debug("Remove").sub_title(display_path).into())
            }
            ToolCatalog::Patch(input) => {
                let display_path = display_path_for(&input.file_path);
                let operation_name = if input.replace_all {
                    "Replace All"
                } else {
                    "Replace"
                };
                Some(
                    TitleFormat::debug(operation_name)
                        .sub_title(display_path)
                        .into(),
                )
            }
            ToolCatalog::MultiPatch(input) => {
                let display_path = display_path_for(&input.file_path);
                Some(
                    TitleFormat::debug("Replace")
                        .sub_title(format!("{} ({} edits)", display_path, input.edits.len()))
                        .into(),
                )
            }
            ToolCatalog::Undo(input) => {
                let display_path = display_path_for(&input.path);
                Some(TitleFormat::debug("Undo").sub_title(display_path).into())
            }
            ToolCatalog::Shell(input) => Some(
                TitleFormat::debug(format!("Execute [{}]", env.shell))
                    .sub_title(&input.command)
                    .into(),
            ),
            ToolCatalog::ProcessStatus(input) => Some(
                TitleFormat::debug("Process Status")
                    .sub_title(format_process_status_subtitle(input))
                    .into(),
            ),
            ToolCatalog::ProcessRead(input) => Some(
                TitleFormat::debug("Read Process")
                    .sub_title(format_process_read_subtitle(input))
                    .into(),
            ),
            ToolCatalog::ProcessList(_) => Some(TitleFormat::debug("List Processes").into()),
            ToolCatalog::ProcessKill(input) => Some(
                TitleFormat::debug("Kill Process")
                    .sub_title(&input.process_id)
                    .into(),
            ),
            ToolCatalog::Fetch(input) => {
                Some(TitleFormat::debug("GET").sub_title(&input.url).into())
            }
            ToolCatalog::Followup(input) => Some(
                TitleFormat::debug("Follow-up")
                    .sub_title(&input.question)
                    .into(),
            ),
            ToolCatalog::Plan(_) => None,
            ToolCatalog::Skill(input) => Some(
                TitleFormat::debug("Skill")
                    .sub_title(input.name.to_lowercase())
                    .into(),
            ),
            ToolCatalog::TodoWrite(input) => Some(
                TitleFormat::debug("Update Todos")
                    .sub_title(format!("{} item(s)", input.todos.len()))
                    .into(),
            ),
            ToolCatalog::TodoRead(_) => Some(TitleFormat::debug("Read Todos").into()),
            ToolCatalog::Task(input) => Some(
                TitleFormat::debug("Task")
                    .sub_title(format_task_subtitle(&input.agent_id, &input.tasks))
                    .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use forge_domain::{
        Category, ChatResponseContent, Environment, ProcessObservationWaitSeconds, TaskInput,
        TitleFormat, ToolCatalog,
    };
    use pretty_assertions::assert_eq;

    use super::FormatContent;

    fn fixture_environment() -> Environment {
        Environment {
            os: "linux".to_string(),
            cwd: PathBuf::from("/workspace"),
            home: Some(PathBuf::from("/home/user")),
            shell: "/usr/bin/zsh".to_string(),
            base_path: PathBuf::from("/home/user/.forge"),
        }
    }

    #[test]
    fn test_process_status_formats_wait_seconds() {
        let fixture = ToolCatalog::ProcessStatus(forge_domain::ProcessStatusInput {
            process_id: "process-174".to_string(),
            wait_seconds: Some(ProcessObservationWaitSeconds::new(5).unwrap()),
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected_title = "Process Status";
        let expected_sub_title = Some("process-174 (wait_seconds: 5)".to_string());
        let expected_category = Category::Debug;

        assert_eq!(actual.title, expected_title);
        assert_eq!(actual.sub_title, expected_sub_title);
        assert_eq!(actual.category, expected_category);
    }

    #[test]
    fn test_process_read_formats_zero_cursor_without_wait_seconds() {
        let fixture = ToolCatalog::ProcessRead(forge_domain::ProcessRead {
            process_id: "process-174".to_string(),
            cursor: 0,
            wait_seconds: None,
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected_title = "Read Process";
        let expected_sub_title = Some("process-174 · cursor: 0".to_string());
        let expected_category = Category::Debug;

        assert_eq!(actual.title, expected_title);
        assert_eq!(actual.sub_title, expected_sub_title);
        assert_eq!(actual.category, expected_category);
    }

    #[test]
    fn test_process_read_formats_zero_cursor_and_wait_seconds() {
        let fixture = ToolCatalog::ProcessRead(forge_domain::ProcessRead {
            process_id: "process-174".to_string(),
            cursor: 0,
            wait_seconds: Some(ProcessObservationWaitSeconds::new(7).unwrap()),
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected_title = "Read Process";
        let expected_sub_title = Some("process-174 · cursor: 0 · wait_seconds: 7".to_string());
        let expected_category = Category::Debug;

        assert_eq!(actual.title, expected_title);
        assert_eq!(actual.sub_title, expected_sub_title);
        assert_eq!(actual.category, expected_category);
    }

    #[test]
    fn test_process_read_formats_nonzero_cursor_without_wait_seconds() {
        let fixture = ToolCatalog::ProcessRead(forge_domain::ProcessRead {
            process_id: "process-174".to_string(),
            cursor: 42,
            wait_seconds: None,
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected = TitleFormat {
            title: "Read Process".to_string(),
            sub_title: Some("process-174 · cursor: 42".to_string()),
            category: Category::Debug,
            timestamp: actual.timestamp,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_process_read_formats_cursor_and_wait_seconds() {
        let fixture = ToolCatalog::ProcessRead(forge_domain::ProcessRead {
            process_id: "process-174".to_string(),
            cursor: 42,
            wait_seconds: Some(ProcessObservationWaitSeconds::new(7).unwrap()),
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected_title = "Read Process";
        let expected_sub_title = Some("process-174 · cursor: 42 · wait_seconds: 7".to_string());
        let expected_category = Category::Debug;

        assert_eq!(actual.title, expected_title);
        assert_eq!(actual.sub_title, expected_sub_title);
        assert_eq!(actual.category, expected_category);
    }

    #[test]
    fn test_task_formats_multiline_tasks_for_live_display() {
        let fixture = ToolCatalog::Task(TaskInput {
            tasks: vec![
                "first line\nsecond line".to_string(),
                "another task".to_string(),
            ],
            agent_id: "agi-dev".to_string(),
            session_id: None,
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected = TitleFormat {
            title: "Task".to_string(),
            sub_title: Some("agi-dev\nfirst line\nsecond line\nanother task".to_string()),
            category: Category::Debug,
            timestamp: actual.timestamp,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_process_status_formats_without_wait_seconds() {
        let fixture = ToolCatalog::ProcessStatus(forge_domain::ProcessStatusInput {
            process_id: "process-174".to_string(),
            wait_seconds: None,
        });

        let actual = fixture.to_content(&fixture_environment()).unwrap();
        let ChatResponseContent::ToolInput(actual) = actual else {
            panic!("expected tool input content");
        };
        let expected_title = "Process Status";
        let expected_sub_title = Some("process-174".to_string());
        let expected_category = Category::Debug;

        assert_eq!(actual.title, expected_title);
        assert_eq!(actual.sub_title, expected_sub_title);
        assert_eq!(actual.category, expected_category);
    }
}
