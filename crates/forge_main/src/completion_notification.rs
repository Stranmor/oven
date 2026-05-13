use std::io::Write;

use forge_config::CompletionNotification;

const TERMINAL_BELL: u8 = b'\x07';

/// Emits the configured main-session completion notification.
///
/// # Arguments
///
/// * `notification` - The optional configured notification backend.
/// * `output` - The stream that receives terminal notification bytes.
///
/// # Errors
///
/// Returns an error when writing to or flushing the output stream fails.
pub fn play_completion_notification(
    notification: Option<&CompletionNotification>,
    output: &mut impl Write,
) -> std::io::Result<()> {
    if let Some(CompletionNotification::Bell) = notification {
        output.write_all(&[TERMINAL_BELL])?;
        output.flush()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_play_completion_notification_writes_bell_when_enabled() {
        let fixture = CompletionNotification::Bell;
        let mut output = Vec::new();

        let actual = play_completion_notification(Some(&fixture), &mut output)
            .map(|_| ())
            .map_err(|error| error.kind());

        let expected = Ok(());
        assert_eq!(actual, expected);
        assert_eq!(output, vec![TERMINAL_BELL]);
    }

    #[test]
    fn test_play_completion_notification_is_silent_when_disabled() {
        let mut output = Vec::new();

        let actual = play_completion_notification(None, &mut output)
            .map(|_| ())
            .map_err(|error| error.kind());

        let expected = Ok(());
        assert_eq!(actual, expected);
        assert_eq!(output, Vec::<u8>::new());
    }

    #[test]
    fn test_play_completion_notification_propagates_write_error() {
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

        let actual = play_completion_notification(Some(&fixture), &mut output)
            .map(|_| ())
            .map_err(|error| error.kind());

        let expected = Err(io::ErrorKind::Other);
        assert_eq!(actual, expected);
    }
}
