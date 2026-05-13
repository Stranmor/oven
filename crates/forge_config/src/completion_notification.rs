use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Session completion sound notification backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, fake::Dummy)]
#[serde(rename_all = "snake_case")]
pub enum CompletionNotification {
    /// Emits a terminal bell character when the main session completes.
    Bell,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_completion_notification_deserializes_bell_field() {
        #[derive(Deserialize)]
        struct Fixture {
            notification: CompletionNotification,
        }

        let fixture = r#"
notification = "bell"
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture)
            .unwrap()
            .notification;

        let expected = CompletionNotification::Bell;
        assert_eq!(actual, expected);
    }
}
