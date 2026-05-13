use std::io::Write;
use std::process::Command;

use forge_config::CompletionNotification;

const TERMINAL_BELL: u8 = b'\x07';
const DESKTOP_NOTIFICATION_TITLE: &str = "Forge";

#[derive(Debug, Clone, PartialEq)]
struct NotificationCommand {
    program: &'static str,
    args: Vec<String>,
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
pub fn play_completion_notification(
    notification: Option<&CompletionNotification>,
    output: &mut impl Write,
    message: &str,
) {
    let mut runner = ProcessNotificationCommandRunner;
    play_completion_notification_with_runner(notification, output, message, &mut runner);
}

fn play_completion_notification_with_runner(
    notification: Option<&CompletionNotification>,
    output: &mut impl Write,
    message: &str,
    runner: &mut impl NotificationCommandRunner,
) {
    match notification {
        Some(CompletionNotification::Bell) => {
            let _ = output.write_all(&[TERMINAL_BELL]);
            let _ = output.flush();
        }
        Some(CompletionNotification::Desktop) => {
            if let Some(command) = desktop_notification_command(DESKTOP_NOTIFICATION_TITLE, message)
            {
                let _ = runner.run(command);
            }
        }
        None => {}
    }
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

    #[derive(Debug, Default)]
    struct RecordingRunner {
        commands: Vec<NotificationCommand>,
        fail: bool,
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

    #[test]
    fn test_play_completion_notification_writes_bell_when_enabled() {
        let fixture = CompletionNotification::Bell;
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            "Finished: Task",
            &mut runner,
        );

        let expected = vec![TERMINAL_BELL];
        assert_eq!(output, expected);
        assert_eq!(runner.commands, Vec::<NotificationCommand>::new());
    }

    #[test]
    fn test_play_completion_notification_is_silent_when_disabled() {
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();

        play_completion_notification_with_runner(None, &mut output, "Finished: Task", &mut runner);

        let expected = Vec::<u8>::new();
        assert_eq!(output, expected);
        assert_eq!(runner.commands, Vec::<NotificationCommand>::new());
    }

    #[test]
    fn test_play_completion_notification_ignores_write_error() {
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

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            "Finished: Task",
            &mut runner,
        );

        let expected = Vec::<NotificationCommand>::new();
        assert_eq!(runner.commands, expected);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_play_completion_notification_runs_linux_desktop_command() {
        let fixture = CompletionNotification::Desktop;
        let mut output = Vec::new();
        let mut runner = RecordingRunner::default();

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            "Finished: Task",
            &mut runner,
        );

        let expected = vec![NotificationCommand {
            program: "notify-send",
            args: vec!["Forge".to_string(), "Finished: Task".to_string()],
        }];
        assert_eq!(runner.commands, expected);
        assert_eq!(output, Vec::<u8>::new());
    }

    #[test]
    fn test_play_completion_notification_ignores_desktop_command_error() {
        let fixture = CompletionNotification::Desktop;
        let mut output = Vec::new();
        let mut runner = RecordingRunner { commands: Vec::new(), fail: true };

        play_completion_notification_with_runner(
            Some(&fixture),
            &mut output,
            "Finished: Task",
            &mut runner,
        );

        let expected = desktop_notification_command("Forge", "Finished: Task")
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(runner.commands, expected);
        assert_eq!(output, Vec::<u8>::new());
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
