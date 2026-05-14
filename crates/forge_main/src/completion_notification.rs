use std::fmt;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use forge_config::{CompletionNotification, TelegramChatId, TelegramCompletionNotification};
use serde::Serialize;

const TERMINAL_BELL: u8 = b'\x07';
const DESKTOP_NOTIFICATION_TITLE: &str = "Forge";
const TELEGRAM_NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(5);
const TELEGRAM_FIELD_MAX_CHARS: usize = 96;

/// Safe summary included in completion notifications.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionNotificationContext {
    title: String,
    project: Option<String>,
    conversation_id: Option<String>,
}

impl CompletionNotificationContext {
    /// Creates a completion notification context with a required title.
    ///
    /// # Arguments
    ///
    /// * `title` - Short status/title string for the completed session.
    pub fn new(title: impl Into<String>) -> Self {
        Self { title: title.into(), project: None, conversation_id: None }
    }

    /// Attaches a project label derived from the workspace path.
    ///
    /// # Arguments
    ///
    /// * `project` - Optional project label.
    pub fn project(mut self, project: Option<String>) -> Self {
        self.project = project;
        self
    }

    /// Attaches a conversation identifier.
    ///
    /// # Arguments
    ///
    /// * `conversation_id` - Optional conversation identifier.
    pub fn conversation_id(mut self, conversation_id: Option<String>) -> Self {
        self.conversation_id = conversation_id;
        self
    }

    fn desktop_message(&self) -> String {
        sanitize_notification_field(&self.title)
    }

    fn telegram_message(&self) -> String {
        let mut lines = vec!["Forge: completed".to_string()];
        let title = sanitize_notification_field(&self.title);
        if !title.is_empty() {
            lines.push(format!("Title: {title}"));
        }
        if let Some(project) = self.project.as_deref().map(sanitize_notification_field)
            && !project.is_empty()
        {
            lines.push(format!("Project: {project}"));
        }
        if let Some(conversation_id) = self
            .conversation_id
            .as_deref()
            .map(sanitize_notification_field)
            && !conversation_id.is_empty()
        {
            lines.push(format!("Conversation: {conversation_id}"));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq)]
struct NotificationCommand {
    program: &'static str,
    args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct TelegramNotificationPayload {
    chat_id: String,
    text: String,
    disable_web_page_preview: bool,
}

#[derive(Clone, PartialEq)]
struct TelegramNotificationRequest {
    token: String,
    payload: TelegramNotificationPayload,
}

impl fmt::Debug for TelegramNotificationRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelegramNotificationRequest")
            .field("token", &"[REDACTED]")
            .field("payload", &self.payload)
            .finish()
    }
}

trait TelegramNotificationSender {
    async fn send(&mut self, request: TelegramNotificationRequest);
}

#[derive(Clone, Copy)]
struct ReqwestTelegramNotificationSender;

impl TelegramNotificationSender for ReqwestTelegramNotificationSender {
    async fn send(&mut self, request: TelegramNotificationRequest) {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", request.token);
        let client = reqwest::Client::new();
        let send = client.post(url).json(&request.payload).send();
        let _ = tokio::time::timeout(TELEGRAM_NOTIFICATION_TIMEOUT, send).await;
    }
}

trait NotificationCommandRunner {
    fn run(&mut self, command: NotificationCommand) -> std::io::Result<()>;
}

struct ProcessNotificationCommandRunner;

impl NotificationCommandRunner for ProcessNotificationCommandRunner {
    fn run(&mut self, command: NotificationCommand) -> std::io::Result<()> {
        Command::new(command.program).args(command.args).status()?;
        Ok(())
    }
}

/// Emits the configured main-session completion notification.
///
/// # Arguments
///
/// * `notification` - The optional configured notification backend.
/// * `output` - The stream that receives terminal notification bytes.
/// * `message` - The completion message displayed in desktop notifications.
pub async fn play_completion_notification(
    notification: Option<&CompletionNotification>,
    output: &mut impl Write,
    context: &CompletionNotificationContext,
) {
    let mut command_runner = ProcessNotificationCommandRunner;
    let mut telegram_sender = ReqwestTelegramNotificationSender;
    play_completion_notification_with_runner(
        notification,
        output,
        context,
        &mut command_runner,
        &mut telegram_sender,
    )
    .await;
}

/// Returns a project label suitable for safe completion notifications.
///
/// # Arguments
///
/// * `path` - Workspace path used to derive the project label.
pub fn project_label(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(sanitize_notification_field)
        .filter(|name| !name.is_empty())
}

async fn play_completion_notification_with_runner(
    notification: Option<&CompletionNotification>,
    output: &mut impl Write,
    context: &CompletionNotificationContext,
    command_runner: &mut impl NotificationCommandRunner,
    telegram_sender: &mut impl TelegramNotificationSender,
) {
    match notification {
        Some(CompletionNotification::Bell) => {
            let _ = output.write_all(&[TERMINAL_BELL]);
            let _ = output.flush();
        }
        Some(CompletionNotification::Desktop) => {
            let message = context.desktop_message();
            if let Some(command) =
                desktop_notification_command(DESKTOP_NOTIFICATION_TITLE, &message)
            {
                let _ = command_runner.run(command);
            }
        }
        Some(CompletionNotification::Telegram(config)) => {
            if let Some(request) = telegram_notification_request(config, context) {
                telegram_sender.send(request).await;
            }
        }
        None => {}
    }
}

fn telegram_notification_request(
    config: &TelegramCompletionNotification,
    context: &CompletionNotificationContext,
) -> Option<TelegramNotificationRequest> {
    telegram_notification_request_with_env(config, context, |name| std::env::var(name).ok())
}

fn telegram_notification_request_with_env(
    config: &TelegramCompletionNotification,
    context: &CompletionNotificationContext,
    mut env_var: impl FnMut(&str) -> Option<String>,
) -> Option<TelegramNotificationRequest> {
    let token = env_var(config.token_env_var.as_str())?;
    let chat_id = resolve_telegram_chat_id(config, &mut env_var)?;
    Some(TelegramNotificationRequest {
        token,
        payload: TelegramNotificationPayload {
            chat_id: chat_id.as_str().to_string(),
            text: context.telegram_message(),
            disable_web_page_preview: true,
        },
    })
}

fn resolve_telegram_chat_id(
    config: &TelegramCompletionNotification,
    env_var: &mut impl FnMut(&str) -> Option<String>,
) -> Option<TelegramChatId> {
    config.chat_id.clone().or_else(|| {
        env_var(config.chat_id_env_var.as_str()).and_then(|value| TelegramChatId::new(value).ok())
    })
}

fn sanitize_notification_field(value: &str) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut truncated = collapsed
        .chars()
        .take(TELEGRAM_FIELD_MAX_CHARS)
        .collect::<String>();
    if collapsed.chars().count() > TELEGRAM_FIELD_MAX_CHARS {
        truncated.push('…');
    }
    truncated
}

#[cfg(target_os = "linux")]
fn desktop_notification_command(title: &str, body: &str) -> Option<NotificationCommand> {
    Some(NotificationCommand {
        program: "notify-send",
        args: vec![title.to_string(), body.to_string()],
    })
}

#[cfg(target_os = "macos")]
fn desktop_notification_command(title: &str, body: &str) -> Option<NotificationCommand> {
    Some(NotificationCommand {
        program: "osascript",
        args: vec![
            "-e".to_string(),
            format!(
                "display notification {} with title {}",
                apple_script_string_literal(body),
                apple_script_string_literal(title)
            ),
        ],
    })
}

#[cfg(target_os = "macos")]
fn apple_script_string_literal(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn desktop_notification_command(_: &str, _: &str) -> Option<NotificationCommand> {
    None
}

#[cfg(test)]
mod tests {
    use std::io;

    use pretty_assertions::assert_eq;

    use super::*;

    fn notification_context() -> CompletionNotificationContext {
        CompletionNotificationContext::new("Finished: Task")
    }

    #[derive(Debug, Default)]
    struct RecordingRunner {
        commands: Vec<NotificationCommand>,
        fail: bool,
    }

    #[derive(Debug, Default)]
    struct RecordingTelegramSender {
        requests: Vec<TelegramNotificationRequest>,
    }

    impl NotificationCommandRunner for RecordingRunner {
        fn run(&mut self, command: NotificationCommand) -> io::Result<()> {
            self.commands.push(command);
            if self.fail {
                Err(io::Error::other("notification failed"))
            } else {
                Ok(())
            }
        }
    }

    impl TelegramNotificationSender for RecordingTelegramSender {
        async fn send(&mut self, request: TelegramNotificationRequest) {
            self.requests.push(request);
        }
    }

    #[tokio::test]
    async fn test_play_completion_notification_writes_bell_when_enabled() {
        let fixture = CompletionNotification::Bell;
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();
        let mut telegram_sender = RecordingTelegramSender::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            &notification_context(),
            &mut runner,
            &mut telegram_sender,
        )
        .await;

        let expected = vec![TERMINAL_BELL];
        assert_eq!(output, expected);
        assert_eq!(runner.commands, Vec::<NotificationCommand>::new());
        assert_eq!(
            telegram_sender.requests,
            Vec::<TelegramNotificationRequest>::new()
        );
    }

    #[tokio::test]
    async fn test_play_completion_notification_is_silent_when_disabled() {
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();
        let mut telegram_sender = RecordingTelegramSender::default();

        play_completion_notification_with_runner(
            None,
            &mut output,
            &notification_context(),
            &mut runner,
            &mut telegram_sender,
        )
        .await;

        let expected = Vec::<u8>::new();
        assert_eq!(output, expected);
        assert_eq!(runner.commands, Vec::<NotificationCommand>::new());
        assert_eq!(
            telegram_sender.requests,
            Vec::<TelegramNotificationRequest>::new()
        );
    }

    #[tokio::test]
    async fn test_play_completion_notification_ignores_write_error() {
        struct BrokenOutput;

        impl io::Write for BrokenOutput {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("write failed"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let fixture = CompletionNotification::Bell;
        let mut output = BrokenOutput;
        let mut runner = RecordingRunner::default();
        let mut telegram_sender = RecordingTelegramSender::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            &notification_context(),
            &mut runner,
            &mut telegram_sender,
        )
        .await;

        let expected = Vec::<NotificationCommand>::new();
        assert_eq!(runner.commands, expected);
        assert_eq!(
            telegram_sender.requests,
            Vec::<TelegramNotificationRequest>::new()
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_play_completion_notification_runs_linux_desktop_command() {
        let fixture = CompletionNotification::Desktop;
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();
        let mut telegram_sender = RecordingTelegramSender::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            &notification_context(),
            &mut runner,
            &mut telegram_sender,
        )
        .await;

        let expected = vec![NotificationCommand {
            program: "notify-send",
            args: vec!["Forge".to_string(), "Finished: Task".to_string()],
        }];
        assert_eq!(runner.commands, expected);
        assert_eq!(output, Vec::<u8>::new());
        assert_eq!(
            telegram_sender.requests,
            Vec::<TelegramNotificationRequest>::new()
        );
    }

    #[tokio::test]
    async fn test_play_completion_notification_ignores_desktop_command_error() {
        let fixture = CompletionNotification::Desktop;
        let mut output = Vec::new();
        let mut runner = RecordingRunner { commands: Vec::new(), fail: true };
        let mut telegram_sender = RecordingTelegramSender::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            &notification_context(),
            &mut runner,
            &mut telegram_sender,
        )
        .await;

        let expected = desktop_notification_command("Forge", "Finished: Task")
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(runner.commands, expected);
        assert_eq!(output, Vec::<u8>::new());
        assert_eq!(
            telegram_sender.requests,
            Vec::<TelegramNotificationRequest>::new()
        );
    }

    #[test]
    fn test_telegram_notification_request_uses_configured_chat_id_without_leaking_token_to_payload()
    {
        let fixture = TelegramCompletionNotification {
            chat_id: Some(TelegramChatId::new("432567587").unwrap()),
            ..Default::default()
        };
        let setup = CompletionNotificationContext::new("Finished: Task")
            .project(Some("oven".to_string()))
            .conversation_id(Some("conv_123".to_string()));

        let actual = telegram_notification_request_with_env(&fixture, &setup, |name| {
            (name == "FORGE_TELEGRAM_BOT_TOKEN").then(|| "secret-token".to_string())
        })
        .expect("request should be created");

        let expected = TelegramNotificationPayload {
            chat_id: "432567587".to_string(),
            text: "Forge: completed\nTitle: Finished: Task\nProject: oven\nConversation: conv_123"
                .to_string(),
            disable_web_page_preview: true,
        };
        assert_eq!(actual.payload, expected);
        assert_eq!(actual.token, "secret-token");
    }

    #[test]
    fn test_telegram_notification_request_reads_chat_id_from_environment() {
        let fixture = TelegramCompletionNotification::default();
        let setup = CompletionNotificationContext::new("  ");

        let actual = telegram_notification_request_with_env(&fixture, &setup, |name| match name {
            "FORGE_TELEGRAM_BOT_TOKEN" => Some("secret-token".to_string()),
            "FORGE_TELEGRAM_CHAT_ID" => Some("432567587".to_string()),
            _ => None,
        })
        .expect("request should be created");

        let expected = TelegramNotificationPayload {
            chat_id: "432567587".to_string(),
            text: "Forge: completed".to_string(),
            disable_web_page_preview: true,
        };
        assert_eq!(actual.payload, expected);
    }

    #[test]
    fn test_telegram_notification_request_is_absent_without_token() {
        let fixture = TelegramCompletionNotification {
            chat_id: Some(TelegramChatId::new("432567587").unwrap()),
            ..Default::default()
        };

        let setup = CompletionNotificationContext::new("Finished: Task");

        let actual = telegram_notification_request_with_env(&fixture, &setup, |_| None);

        assert!(actual.is_none());
    }

    #[test]
    fn test_telegram_notification_request_debug_redacts_token() {
        let setup = TelegramNotificationRequest {
            token: "secret-token".to_string(),
            payload: TelegramNotificationPayload {
                chat_id: "432567587".to_string(),
                text: "Forge: completed".to_string(),
                disable_web_page_preview: true,
            },
        };

        let actual = format!("{setup:?}");

        assert!(!actual.contains("secret-token"));
    }

    #[test]
    fn test_project_label_uses_last_path_component() {
        let fixture = Path::new("/home/stranmor/Documents/project/_mycelium/oven");

        let actual = project_label(fixture);

        let expected = Some("oven".to_string());
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_telegram_message_sanitizes_and_truncates_fields() {
        let setup = CompletionNotificationContext::new(format!("{} secret", "x".repeat(120)))
            .project(Some("oven\nrepo".to_string()))
            .conversation_id(Some("conv\t123".to_string()));

        let actual = setup.telegram_message();

        let expected = format!(
            "Forge: completed\nTitle: {}…\nProject: oven repo\nConversation: conv 123",
            "x".repeat(96)
        );
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_apple_script_string_literal_escapes_quotes_and_backslashes() {
        let fixture = "Finished: \\\"Task\\\"";

        let actual = apple_script_string_literal(fixture);

        let expected = "\"Finished: \\\\\\\"Task\\\\\\\"\"".to_string();
        assert_eq!(actual, expected);
    }
}
