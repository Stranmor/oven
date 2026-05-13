use std::collections::HashMap;
use std::fmt::Write;

use derive_more::{Deref, From};
use derive_setters::Setters;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Attachment, NamedTool, Template, TerminalContext, ToolName};

/// Represents a partial event structure used for CLI event dispatching
///
/// This is an intermediate structure for parsing event JSON from the CLI
/// before converting it to a full Event type.
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct UserCommand {
    pub name: String,
    pub template: Template<Value>,
    pub parameters: Vec<String>,
}

impl UserCommand {
    pub fn new<V: Into<Template<Value>>>(
        name: impl ToString,
        value: V,
        parameters: Vec<String>,
    ) -> Self {
        Self { name: name.to_string(), template: value.into(), parameters }
    }
}

impl From<UserCommand> for Event {
    fn from(value: UserCommand) -> Self {
        Event::new(EventValue::Command(value))
    }
}

impl<T: AsRef<str>> From<T> for EventValue {
    fn from(value: T) -> Self {
        EventValue::Text(UserPrompt(value.as_ref().to_owned()))
    }
}

// We'll use simple strings for JSON schema compatibility
#[derive(Debug, Deserialize, Serialize, Clone, Setters)]
#[setters(into, strip_option)]
pub struct Event {
    pub id: String,
    pub value: Option<EventValue>,
    pub timestamp: String,
    pub attachments: Vec<Attachment>,

    /// Contains additional context about the prompt that should typically be
    /// included after the `value` as a user message.
    pub additional_context: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub enum EventValue {
    Text(UserPrompt),
    Command(UserCommand),
}

impl EventValue {
    pub fn as_user_prompt(&self) -> Option<&UserPrompt> {
        match self {
            EventValue::Text(user_prompt) => Some(user_prompt),
            EventValue::Command(_) => None,
        }
    }

    pub fn as_command(&self) -> Option<&UserCommand> {
        match self {
            EventValue::Text(_user_prompt) => None,
            EventValue::Command(user_command) => Some(user_command),
        }
    }

    pub fn text(str: impl ToString) -> Self {
        EventValue::Text(UserPrompt(str.to_string()))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, From, Deref)]
#[serde(transparent)]
pub struct UserPrompt(String);

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Setters)]
pub struct EventContext {
    event: EventContextValue,
    suggestions: Vec<String>,
    variables: HashMap<String, Value>,
    current_date: String,
    #[serde(default)]
    current_datetime: String,
    #[serde(default)]
    timezone_offset: String,
    #[serde(default)]
    unix_timestamp: i64,
    /// Structured terminal context injected by [`TerminalContextService`],
    /// or `None` when terminal context is unavailable or disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    terminal_context: Option<TerminalContext>,
}

/// Request-scoped live runtime time context rendered as an uncached prompt message.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct LiveRuntimeContext {
    current_date: String,
    current_datetime: String,
    timezone_offset: String,
    unix_timestamp: i64,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Setters)]
pub struct EventContextValue {
    pub name: String,
    pub value: String,
}

impl EventContextValue {
    pub fn new<S: Into<String>>(value: S) -> Self {
        Self { name: String::new(), value: value.into() }
    }
}

impl EventContext {
    pub fn new(event: impl Into<EventContextValue>) -> Self {
        Self::from_runtime_context(event, LiveRuntimeContext::now())
    }

    /// Creates an event context using a request-scoped runtime timestamp.
    ///
    /// # Arguments
    /// * `event` - User event value exposed to prompt templates.
    /// * `runtime_context` - Live time fields captured for the current request.
    pub fn from_runtime_context(
        event: impl Into<EventContextValue>,
        runtime_context: LiveRuntimeContext,
    ) -> Self {
        Self {
            event: event.into(),
            suggestions: Default::default(),
            variables: Default::default(),
            current_date: runtime_context.current_date,
            current_datetime: runtime_context.current_datetime,
            timezone_offset: runtime_context.timezone_offset,
            unix_timestamp: runtime_context.unix_timestamp,
            terminal_context: None,
        }
    }

    /// Converts this EventContext into a feedback event by setting the event
    /// name to "feedback". This should be used when the context already
    /// contains user messages.
    pub fn into_feedback(mut self) -> Self {
        self.event.name = "feedback".to_string();
        self
    }

    /// Converts this EventContext into a new task event by setting the event
    /// name to "task". This should be used when this is a new task without
    /// prior user messages.
    pub fn into_task(mut self) -> Self {
        self.event.name = "task".to_string();
        self
    }
}

impl LiveRuntimeContext {
    /// Creates live runtime context from a request-scoped local timestamp.
    ///
    /// # Arguments
    /// * `current_time` - Timestamp captured once for the current chat request.
    pub fn from_local(current_time: chrono::DateTime<chrono::Local>) -> Self {
        Self {
            current_date: current_time.format("%Y-%m-%d").to_string(),
            current_datetime: current_time.to_rfc3339(),
            timezone_offset: current_time.format("%:z").to_string(),
            unix_timestamp: current_time.timestamp(),
        }
    }

    /// Creates live runtime context from the current local clock.
    pub fn now() -> Self {
        Self::from_local(chrono::Local::now())
    }

    /// Returns the date-only compatibility value for existing templates.
    pub fn current_date(&self) -> &str {
        &self.current_date
    }

    /// Returns the RFC3339 local datetime with offset.
    pub fn current_datetime(&self) -> &str {
        &self.current_datetime
    }

    /// Returns the timezone offset for the runtime timestamp.
    pub fn timezone_offset(&self) -> &str {
        &self.timezone_offset
    }

    /// Returns the Unix timestamp for the runtime timestamp.
    pub fn unix_timestamp(&self) -> i64 {
        self.unix_timestamp
    }

    /// Renders this context as a compact XML prompt payload.
    pub fn render_prompt_xml(&self) -> String {
        let mut output = String::new();
        writeln!(
            output,
            "<runtime_context freshness=\"live\" cache=\"uncached\">"
        )
        .expect("Writing to String should not fail");
        writeln!(output, "<current_date>{}</current_date>", self.current_date)
            .expect("Writing to String should not fail");
        writeln!(
            output,
            "<current_datetime>{}</current_datetime>",
            self.current_datetime
        )
        .expect("Writing to String should not fail");
        writeln!(
            output,
            "<timezone_offset>{}</timezone_offset>",
            self.timezone_offset
        )
        .expect("Writing to String should not fail");
        writeln!(
            output,
            "<unix_timestamp>{}</unix_timestamp>",
            self.unix_timestamp
        )
        .expect("Writing to String should not fail");
        output.push_str("</runtime_context>");
        output
    }
}

impl NamedTool for Event {
    fn tool_name() -> ToolName {
        ToolName::new("forge_tool_event_dispatch")
    }
}

impl Event {
    pub fn new<V: Into<EventValue>>(value: V) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();

        Self {
            id,
            value: Some(value.into()),
            timestamp,
            attachments: Vec::new(),
            additional_context: None,
        }
    }

    pub fn empty() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();

        Self {
            id,
            value: None,
            timestamp,
            attachments: Vec::new(),
            additional_context: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_feedback() {
        let event = EventContextValue::new("");
        let context = EventContext::new(event);

        let feedback_context = context.into_feedback();

        assert_eq!(feedback_context.event.name, "feedback");
    }

    #[test]
    fn test_into_task() {
        let event = EventContextValue::new("");
        let context = EventContext::new(event);

        let task_context = context.into_task();

        assert_eq!(task_context.event.name, "task");
    }

    #[test]
    fn test_into_feedback_idempotent() {
        let event = EventContextValue::new("");
        let context = EventContext::new(event);

        // Call into_feedback twice
        let feedback_context = context.into_feedback().into_feedback();

        assert_eq!(feedback_context.event.name, "feedback");
    }

    #[test]
    fn test_into_task_idempotent() {
        let event = EventContextValue::new("");
        let context = EventContext::new(event);

        // Call into_task twice
        let task_context = context.into_task().into_task();

        assert_eq!(task_context.event.name, "task");
    }

    #[test]
    fn test_live_runtime_context_from_local_uses_request_timestamp() {
        let current_time = chrono::DateTime::parse_from_rfc3339("2026-05-13T12:34:56+03:00")
            .unwrap()
            .with_timezone(&chrono::Local);
        let actual = LiveRuntimeContext::from_local(current_time);
        let expected = (
            "2026-05-13",
            "2026-05-13T12:34:56+03:00",
            "+03:00",
            1778664896,
        );
        assert_eq!(
            (
                actual.current_date(),
                actual.current_datetime(),
                actual.timezone_offset(),
                actual.unix_timestamp(),
            ),
            expected
        );
    }

    #[test]
    fn test_live_runtime_context_renders_prompt_xml() {
        let current_time = chrono::DateTime::parse_from_rfc3339("2026-05-13T12:34:56+03:00")
            .unwrap()
            .with_timezone(&chrono::Local);
        let actual = LiveRuntimeContext::from_local(current_time).render_prompt_xml();
        let expected = "<runtime_context freshness=\"live\" cache=\"uncached\">\n<current_date>2026-05-13</current_date>\n<current_datetime>2026-05-13T12:34:56+03:00</current_datetime>\n<timezone_offset>+03:00</timezone_offset>\n<unix_timestamp>1778664896</unix_timestamp>\n</runtime_context>";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_chaining_methods() {
        let event = EventContextValue::new("initial content");
        let context = EventContext::new(event).into_task();

        assert_eq!(context.event.name, "task");
        assert_eq!(context.event.value, "initial content");
    }
}
