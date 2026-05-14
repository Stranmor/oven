use std::borrow::Cow;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

const DEFAULT_TELEGRAM_BOT_TOKEN_ENV_VAR: &str = "FORGE_TELEGRAM_BOT_TOKEN";
const DEFAULT_TELEGRAM_CHAT_ID_ENV_VAR: &str = "FORGE_TELEGRAM_CHAT_ID";

/// Session completion notification backend.
#[derive(Debug, Clone, PartialEq, fake::Dummy)]
pub enum CompletionNotification {
    /// Emits a terminal bell character when the main session completes.
    Bell,
    /// Emits an operating-system desktop notification when the main session
    /// completes.
    Desktop,
    /// Sends a Telegram Bot API message when the main session completes.
    Telegram(TelegramCompletionNotification),
}

/// Telegram completion notification settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, fake::Dummy)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct TelegramCompletionNotification {
    /// Telegram backend discriminator for table-style configuration.
    #[serde(default = "telegram_completion_notification_backend")]
    pub backend: TelegramCompletionNotificationBackend,
    /// Telegram chat identifier that receives completion messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<TelegramChatId>,
    /// Environment variable containing the Telegram bot token.
    #[serde(default = "default_telegram_bot_token_env_var")]
    pub token_env_var: EnvVarName,
    /// Environment variable containing the Telegram chat id when `chat_id` is
    /// absent.
    #[serde(default = "default_telegram_chat_id_env_var")]
    pub chat_id_env_var: EnvVarName,
}

impl Default for TelegramCompletionNotification {
    fn default() -> Self {
        Self {
            backend: TelegramCompletionNotificationBackend::Telegram,
            chat_id: None,
            token_env_var: default_telegram_bot_token_env_var(),
            chat_id_env_var: default_telegram_chat_id_env_var(),
        }
    }
}

/// Telegram completion notification backend discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, fake::Dummy)]
#[serde(rename_all = "snake_case")]
pub enum TelegramCompletionNotificationBackend {
    /// Telegram Bot API backend.
    Telegram,
}

/// Telegram chat identifier.
#[derive(Debug, Clone, PartialEq, Eq, fake::Dummy)]
pub struct TelegramChatId(String);

impl TelegramChatId {
    /// Creates a Telegram chat identifier after validating it is non-empty.
    ///
    /// # Arguments
    ///
    /// * `value` - Raw Telegram chat identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is empty after trimming whitespace.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            Err("Telegram chat id must not be empty".to_string())
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the Telegram chat identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for TelegramChatId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TelegramChatId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Environment variable name.
#[derive(Debug, Clone, PartialEq, Eq, fake::Dummy)]
pub struct EnvVarName(String);

impl EnvVarName {
    /// Creates an environment variable name after validating it is non-empty,
    /// does not contain `=`, and does not contain NUL.
    ///
    /// # Arguments
    ///
    /// * `value` - Raw environment variable name.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is empty, contains `=`, or contains NUL.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            Err("environment variable name must not be empty".to_string())
        } else if value.contains('=') {
            Err("environment variable name must not contain '='".to_string())
        } else if value.contains('\0') {
            Err("environment variable name must not contain NUL".to_string())
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the environment variable name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for EnvVarName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EnvVarName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

impl schemars::JsonSchema for TelegramChatId {
    fn schema_name() -> Cow<'static, str> {
        "TelegramChatId".into()
    }

    fn json_schema(_gen: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "Telegram chat identifier.",
            "type": "string",
            "minLength": 1
        })
    }
}

impl schemars::JsonSchema for EnvVarName {
    fn schema_name() -> Cow<'static, str> {
        "EnvVarName".into()
    }

    fn json_schema(_gen: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "Environment variable name.",
            "type": "string",
            "minLength": 1,
            "pattern": "^[^=\\u0000]+$"
        })
    }
}

impl schemars::JsonSchema for CompletionNotification {
    fn schema_name() -> Cow<'static, str> {
        "CompletionNotification".into()
    }

    fn json_schema(r#gen: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
        let _ = r#gen.subschema_for::<TelegramCompletionNotification>();
        schemars::json_schema!({
            "description": "Session completion notification backend.",
            "anyOf": [
                {
                    "description": "Emits a terminal bell character when the main session completes.",
                    "type": "string",
                    "const": "bell"
                },
                {
                    "description": "Emits an operating-system desktop notification when the main session completes.",
                    "type": "string",
                    "const": "desktop"
                },
                {
                    "description": "Sends a Telegram Bot API message when the main session completes using default environment variable names.",
                    "type": "string",
                    "const": "telegram"
                },
                {
                    "description": "Sends a Telegram Bot API message using table-style settings.",
                    "$ref": "#/$defs/TelegramCompletionNotification"
                }
            ]
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompletionNotificationBackend {
    Bell,
    Desktop,
    Telegram,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionNotificationWire {
    Backend(CompletionNotificationBackend),
    Telegram(TelegramCompletionNotification),
}

impl Serialize for CompletionNotification {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Bell => CompletionNotificationBackend::Bell.serialize(serializer),
            Self::Desktop => CompletionNotificationBackend::Desktop.serialize(serializer),
            Self::Telegram(config) => config.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for CompletionNotification {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match CompletionNotificationWire::deserialize(deserializer)? {
            CompletionNotificationWire::Backend(CompletionNotificationBackend::Bell) => {
                Ok(Self::Bell)
            }
            CompletionNotificationWire::Backend(CompletionNotificationBackend::Desktop) => {
                Ok(Self::Desktop)
            }
            CompletionNotificationWire::Backend(CompletionNotificationBackend::Telegram) => {
                Ok(Self::Telegram(TelegramCompletionNotification::default()))
            }
            CompletionNotificationWire::Telegram(config) => Ok(Self::Telegram(config)),
        }
    }
}

fn telegram_completion_notification_backend() -> TelegramCompletionNotificationBackend {
    TelegramCompletionNotificationBackend::Telegram
}

fn default_telegram_bot_token_env_var() -> EnvVarName {
    EnvVarName::new(DEFAULT_TELEGRAM_BOT_TOKEN_ENV_VAR)
        .expect("default Telegram bot token env var should be valid")
}

fn default_telegram_chat_id_env_var() -> EnvVarName {
    EnvVarName::new(DEFAULT_TELEGRAM_CHAT_ID_ENV_VAR)
        .expect("default Telegram chat id env var should be valid")
}

impl fmt::Display for TelegramChatId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl fmt::Display for EnvVarName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
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
            .expect("config fixture should deserialize from TOML")
            .notification;

        let expected = CompletionNotification::Bell;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_completion_notification_deserializes_desktop_field() {
        #[derive(Deserialize)]
        struct Fixture {
            notification: CompletionNotification,
        }

        let fixture = r#"
notification = "desktop"
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture)
            .expect("config fixture should deserialize from TOML")
            .notification;

        let expected = CompletionNotification::Desktop;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_completion_notification_deserializes_telegram_string_field() {
        #[derive(Deserialize)]
        struct Fixture {
            notification: CompletionNotification,
        }

        let fixture = r#"
notification = "telegram"
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture)
            .expect("config fixture should deserialize from TOML")
            .notification;

        let expected = CompletionNotification::Telegram(TelegramCompletionNotification::default());
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_completion_notification_deserializes_telegram_table_field() {
        #[derive(Deserialize)]
        struct Fixture {
            notification: CompletionNotification,
        }

        let fixture = r#"
[notification]
backend = "telegram"
chat_id = "432567587"
token_env_var = "CUSTOM_TELEGRAM_TOKEN"
chat_id_env_var = "CUSTOM_TELEGRAM_CHAT_ID"
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture)
            .expect("config fixture should deserialize from TOML")
            .notification;

        let expected = CompletionNotification::Telegram(TelegramCompletionNotification {
            backend: TelegramCompletionNotificationBackend::Telegram,
            chat_id: Some(
                TelegramChatId::new("432567587").expect("telegram chat id should be valid"),
            ),
            token_env_var: EnvVarName::new("CUSTOM_TELEGRAM_TOKEN")
                .expect("token env var name should be valid"),
            chat_id_env_var: EnvVarName::new("CUSTOM_TELEGRAM_CHAT_ID")
                .expect("chat id env var name should be valid"),
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_completion_notification_serializes_legacy_backends_as_strings() {
        #[derive(Serialize)]
        struct Fixture {
            notification: CompletionNotification,
        }

        let setup = Fixture { notification: CompletionNotification::Bell };

        let actual = toml_edit::ser::to_string_pretty(&setup)
            .expect("config fixture should serialize to TOML");

        let expected = "notification = \"bell\"\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_completion_notification_rejects_empty_telegram_chat_id() {
        #[derive(Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            notification: CompletionNotification,
        }

        let fixture = r#"
[notification]
backend = "telegram"
chat_id = " "
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture).is_err();

        assert!(actual);
    }

    #[test]
    fn test_completion_notification_rejects_unknown_telegram_field() {
        #[derive(Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            notification: CompletionNotification,
        }

        let fixture = r#"
[notification]
backend = "telegram"
token_env_vra = "CUSTOM_TELEGRAM_TOKEN"
"#;

        let actual = toml_edit::de::from_str::<Fixture>(fixture).is_err();

        assert!(actual);
    }

    #[test]
    fn test_completion_notification_rejects_nul_env_var_name() {
        #[derive(Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            notification: CompletionNotification,
        }

        let fixture = "\n[notification]\nbackend = \"telegram\"\ntoken_env_var = \"FORGE\\u0000TELEGRAM_BOT_TOKEN\"\n";

        let actual = toml_edit::de::from_str::<Fixture>(fixture).is_err();

        assert!(actual);
    }
}
