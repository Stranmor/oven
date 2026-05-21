use std::ops::Deref;
use std::sync::Arc;

use forge_domain::{Agent, *};
use serde_json::json;
use tracing::debug;

use crate::{AttachmentService, EnvironmentInfra, TemplateEngine, TerminalContextService};

/// Service responsible for setting user prompts in the conversation context
#[derive(Clone)]
pub struct UserPromptGenerator<S> {
    services: Arc<S>,
    agent: Agent,
    event: Event,
    current_time: chrono::DateTime<chrono::Local>,
}

impl<S: AttachmentService + EnvironmentInfra<Config = forge_config::ForgeConfig>>
    UserPromptGenerator<S>
{
    /// Creates a new UserPromptService
    pub fn new(
        service: Arc<S>,
        agent: Agent,
        event: Event,
        current_time: chrono::DateTime<chrono::Local>,
    ) -> Self {
        Self { services: service, agent, event, current_time }
    }

    /// Sets the user prompt in the context based on agent configuration and
    /// event data
    pub async fn add_user_prompt(
        &self,
        conversation: Conversation,
    ) -> anyhow::Result<Conversation> {
        // Check if this is a resume BEFORE adding new messages
        let is_resume = conversation
            .context
            .as_ref()
            .map(|ctx| ctx.messages.iter().any(|msg| msg.has_role(Role::User)))
            .unwrap_or(false);

        let (conversation, content) = self.add_rendered_message(conversation).await?;
        let conversation = self.add_runtime_context(conversation);
        let conversation = self.add_goal_context(conversation);
        let conversation = if is_resume {
            self.add_todos_on_resume(conversation)?
        } else {
            conversation
        };
        let conversation = self.add_additional_context(conversation).await?;
        let conversation = if let Some(content) = content {
            self.add_attachments(conversation, &content).await?
        } else {
            conversation
        };

        Ok(conversation)
    }

    fn runtime_context(&self) -> LiveRuntimeContext {
        LiveRuntimeContext::from_local(self.current_time)
    }

    /// Adds request-scoped runtime context as a small cache-ineligible message.
    fn add_runtime_context(&self, mut conversation: Conversation) -> Conversation {
        let mut context = conversation.context.take().unwrap_or_default();
        context
            .messages
            .retain(|message| !Self::is_stale_runtime_context_message(message));
        let message = TextMessage::new(Role::User, self.runtime_context().render_prompt_xml())
            .model(self.agent.model.clone())
            .runtime_context()
            .cacheable(false);
        context = context.add_message(ContextMessage::Text(message));
        conversation.context(context)
    }

    fn is_stale_runtime_context_message(message: &MessageEntry) -> bool {
        matches!(&message.message, ContextMessage::Text(text) if text.is_runtime_context())
    }

    /// Adds an active thread goal from typed conversation state.
    fn add_goal_context(&self, mut conversation: Conversation) -> Conversation {
        let mut context = conversation.context.take().unwrap_or_default();
        context
            .messages
            .retain(|message| !Self::is_stale_goal_context_message(message));
        if let Some(goal) = context.active_goal.clone().filter(ActiveGoal::is_active) {
            let message = TextMessage::goal_context(Role::User, goal.render_prompt_xml())
                .model(self.agent.model.clone());
            context = context.add_message(ContextMessage::Text(message));
        }
        conversation.context(context)
    }

    fn is_stale_goal_context_message(message: &MessageEntry) -> bool {
        matches!(&message.message, ContextMessage::Text(text) if text.is_goal_context())
    }

    /// Adds existing todos as a user message when resuming a conversation
    fn add_todos_on_resume(&self, mut conversation: Conversation) -> anyhow::Result<Conversation> {
        let mut context = conversation.context.take().unwrap_or_default();

        // Load existing todos from session metrics
        let todos = conversation.metrics.todos.clone();

        if !todos.is_empty() {
            // Format todos as markdown checklist
            let todo_content = self.format_todos_as_markdown(&todos);

            // Add as a droppable user message after the new task
            let todo_message = TextMessage {
                role: Role::User,
                content: todo_content,
                raw_content: None,
                tool_calls: None,
                thought_signature: None,
                reasoning_details: None,
                model: Some(self.agent.model.clone()),
                droppable: true, // Droppable so it can be removed during context compression
                phase: None,
                cacheable: Some(false),
                cache_class: Some(MessageCacheClass::Uncached),
                kind: None,
            };
            context = context.add_message(ContextMessage::Text(todo_message));
        }

        Ok(conversation.context(context))
    }

    /// Formats todos as a markdown checklist
    fn format_todos_as_markdown(&self, todos: &[Todo]) -> String {
        use std::fmt::Write;

        let mut content = String::from("**Current task list:**\n\n");

        for todo in todos {
            let checkbox = match todo.status {
                TodoStatus::Completed => "[DONE]",
                TodoStatus::InProgress => "[IN_PROGRESS]",
                TodoStatus::Pending => "[PENDING]",
                TodoStatus::Cancelled => "[CANCELLED]",
            };

            writeln!(content, "- {} {}", checkbox, todo.content)
                .expect("Writing to String should not fail");
        }

        content
    }

    /// Adds additional context (piped input) as a droppable user message
    async fn add_additional_context(
        &self,
        mut conversation: Conversation,
    ) -> anyhow::Result<Conversation> {
        let mut context = conversation.context.take().unwrap_or_default();

        if let Some(piped_input) = &self.event.additional_context {
            let piped_message = TextMessage {
                role: Role::User,
                content: piped_input.clone(),
                raw_content: None,
                tool_calls: None,
                thought_signature: None,
                reasoning_details: None,
                model: Some(self.agent.model.clone()),
                droppable: true, // Piped input is droppable
                phase: None,
                cacheable: Some(false),
                cache_class: Some(MessageCacheClass::Uncached),
                kind: None,
            };
            context = context.add_message(ContextMessage::Text(piped_message));
        }

        Ok(conversation.context(context))
    }

    /// Renders the user message content and adds it to the conversation
    /// Returns the conversation and the rendered content for attachment parsing
    async fn add_rendered_message(
        &self,
        mut conversation: Conversation,
    ) -> anyhow::Result<(Conversation, Option<String>)> {
        let mut context = conversation.context.take().unwrap_or_default();
        let event_value = self.event.value.clone();
        let template_engine = TemplateEngine::default();

        let content = if let Some(user_prompt) = &self.agent.user_prompt
            && self.event.value.is_some()
        {
            let user_input = self
                .event
                .value
                .as_ref()
                .and_then(|v| v.as_user_prompt().map(|u| u.as_str().to_string()))
                .unwrap_or_default();
            let runtime_context = self.runtime_context();
            let mut event_context = EventContext::from_runtime_context(
                EventContextValue::new(user_input),
                runtime_context,
            );

            // Check if context already contains user messages to determine if it's feedback
            let has_user_messages = context.messages.iter().any(|msg| msg.has_role(Role::User));

            if has_user_messages {
                event_context = event_context.into_feedback();
            } else {
                event_context = event_context.into_task();
            }

            debug!(event_context = ?event_context, "Event context");

            // Render the command first.
            let event_context = match self.event.value.as_ref().and_then(|v| v.as_command()) {
                Some(command) => {
                    let rendered_prompt = template_engine.render_template(
                        command.template.clone(),
                        &json!({"parameters": command.parameters.join(" ")}),
                    )?;
                    event_context.event(EventContextValue::new(rendered_prompt))
                }
                None => event_context,
            };

            // Inject terminal context into the event context when available.
            let event_context =
                match TerminalContextService::new(self.services.clone()).get_terminal_context() {
                    Some(ctx) => event_context.terminal_context(Some(ctx)),
                    None => event_context,
                };

            // Render the event value into agent's user prompt template.
            Some(
                template_engine.render_template(
                    Template::new(user_prompt.template.as_str()),
                    &event_context,
                )?,
            )
        } else {
            // Use the raw event value as content if no user_prompt is provided
            event_value
                .as_ref()
                .and_then(|v| v.as_user_prompt().map(|p| p.deref().to_owned()))
        };

        if let Some(content) = &content {
            // Create User Message
            let message = TextMessage {
                role: Role::User,
                content: content.clone(),
                raw_content: event_value,
                tool_calls: None,
                thought_signature: None,
                reasoning_details: None,
                model: Some(self.agent.model.clone()),
                droppable: false,
                phase: None,
                cacheable: None,
                cache_class: None,
                kind: None,
            };
            context = context.add_message(ContextMessage::Text(message));
        }

        Ok((conversation.context(context), content))
    }

    /// Parses and adds attachments to the conversation based on the provided
    /// content
    async fn add_attachments(
        &self,
        mut conversation: Conversation,
        content: &str,
    ) -> anyhow::Result<Conversation> {
        let mut context = conversation.context.take().unwrap_or_default();

        // Parse Attachments (do NOT parse piped input for attachments)
        let attachments = self.services.attachments(content).await?;

        // Track file attachments as read operations in metrics
        let mut metrics = conversation.metrics.clone();
        for attachment in &attachments {
            // Only track file content attachments (not images or directory listings).
            // Use the raw content_hash (computed before line-numbering) so that the
            // external-change detector, which hashes the raw file on disk, sees a
            // matching hash and does not raise a false "modified externally" warning.
            if let AttachmentContent::FileContent { info, .. } = &attachment.content {
                metrics = metrics.insert(
                    attachment.path.clone(),
                    FileOperation::new(ToolKind::Read)
                        .content_hash(Some(info.content_hash.clone())),
                );
            }
        }
        conversation.metrics = metrics;

        context = context.add_attachments(attachments, Some(self.agent.model.clone()));

        Ok(conversation.context(context))
    }
}

#[cfg(test)]
mod tests {
    use forge_domain::{
        AgentId, AttachmentContent, Context, ContextMessage, ConversationId, FileInfo, ModelId,
        ProviderId, ToolKind,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    struct MockService;

    #[async_trait::async_trait]
    impl AttachmentService for MockService {
        async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
            Ok(Vec::new())
        }
    }

    impl crate::EnvironmentInfra for MockService {
        type Config = forge_config::ForgeConfig;

        fn get_environment(&self) -> forge_domain::Environment {
            use fake::{Fake, Faker};
            Faker.fake()
        }

        fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
            Default::default()
        }
    }

    fn fixture_agent_without_user_prompt() -> Agent {
        Agent::new(
            AgentId::from("test_agent"),
            ProviderId::OPENAI,
            ModelId::from("test-model"),
        )
    }

    fn fixture_conversation() -> Conversation {
        Conversation::new(ConversationId::default()).context(Context::default())
    }

    fn fixture_generator(agent: Agent, event: Event) -> UserPromptGenerator<MockService> {
        let current_time = chrono::DateTime::parse_from_rfc3339("2026-05-13T12:34:56+03:00")
            .unwrap()
            .with_timezone(&chrono::Local);
        UserPromptGenerator::new(Arc::new(MockService), agent, event, current_time)
    }

    #[tokio::test]
    async fn test_adds_context_as_droppable_message() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("First Message").additional_context("Second Message");
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        assert_eq!(
            messages.len(),
            3,
            "Should have task, runtime context, and context message"
        );

        // First message should be the context (droppable)
        let task_message = messages.first().unwrap();
        assert_eq!(task_message.content().unwrap(), "First Message");
        assert!(
            !task_message.is_droppable(),
            "Context message should be droppable"
        );

        // Third message should be the additional context and should be droppable
        let context_message = messages.last().unwrap();
        assert_eq!(context_message.content().unwrap(), "Second Message");
        assert!(
            context_message.is_droppable(),
            "Additional context message should be droppable"
        );
        assert_eq!(context_message.is_cache_eligible(), false);
    }

    #[tokio::test]
    async fn test_context_added_before_main_message() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("First Message").additional_context("Second Message");
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        assert_eq!(messages.len(), 3);

        // Verify order: main message first, runtime context second, then additional
        // context
        assert_eq!(messages[0].content().unwrap(), "First Message");
        assert!(messages[1].content().unwrap().contains("<runtime_context"));
        assert_eq!(messages[2].content().unwrap(), "Second Message");
    }

    #[tokio::test]
    async fn test_no_context_only_main_message() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Simple task");
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        assert_eq!(
            messages.len(),
            2,
            "Should have main and runtime context messages"
        );
        assert_eq!(messages[0].content().unwrap(), "Simple task");
        assert!(messages[1].content().unwrap().contains("<runtime_context"));
    }

    #[tokio::test]
    async fn test_empty_event_no_message_added() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::empty();
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        assert_eq!(
            messages.len(),
            1,
            "Should add runtime context even when there is no user event message"
        );
        assert!(messages[0].content().unwrap().contains("<runtime_context"));
        assert_eq!(messages[0].is_cache_eligible(), false);
    }

    #[tokio::test]
    async fn test_raw_content_preserved_in_message() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Task text");
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        let message = messages.first().unwrap();

        if let ContextMessage::Text(text_msg) = &**message {
            assert!(
                text_msg.raw_content.is_some(),
                "Raw content should be preserved"
            );
            let raw = text_msg.raw_content.as_ref().unwrap();
            assert_eq!(raw.as_user_prompt().unwrap().as_str(), "Task text");
        } else {
            panic!("Expected TextMessage");
        }
    }

    #[tokio::test]
    async fn test_runtime_context_added_as_cache_ineligible_message() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Simple task");
        let conversation = fixture_conversation();
        let generator = fixture_generator(agent.clone(), event);

        let expected_runtime_context = generator.runtime_context();
        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        let runtime_message = &messages[1];
        let runtime_content = runtime_message.content().unwrap();
        assert!(
            runtime_content.contains("<runtime_context freshness=\"live\" cache=\"uncached\">")
        );
        assert!(runtime_content.contains(&format!(
            "<current_date>{}</current_date>",
            expected_runtime_context.current_date()
        )));
        assert!(runtime_content.contains(&format!(
            "<current_datetime>{}</current_datetime>",
            expected_runtime_context.current_datetime()
        )));
        assert!(runtime_content.contains(&format!(
            "<timezone_offset>{}</timezone_offset>",
            expected_runtime_context.timezone_offset()
        )));
        assert!(runtime_content.contains(&format!(
            "<unix_timestamp>{}</unix_timestamp>",
            expected_runtime_context.unix_timestamp()
        )));
        assert!(matches!(
            &runtime_message.message,
            ContextMessage::Text(text) if text.is_runtime_context()
        ));
        assert_eq!(runtime_message.is_cache_eligible(), false);
        assert_eq!(runtime_message.is_droppable(), false);
    }

    #[tokio::test]
    async fn test_active_goal_is_injected_from_typed_state() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("First task");
        let conversation = fixture_conversation().context(
            Context::default().set_active_goal(ActiveGoal::new("finish the slice").unwrap()),
        );
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        let goal_message = messages
            .iter()
            .find(|message| {
                matches!(&message.message, ContextMessage::Text(text) if text.is_goal_context())
            })
            .expect("goal context should be injected");
        let actual = goal_message.content().unwrap();
        let expected = "<conversation_goal\n  status=\"active\"\n>\n<objective>finish the slice</objective>\n</conversation_goal>";
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_paused_goal_is_persisted_but_not_injected() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("First task");
        let mut goal = ActiveGoal::new("finish the slice").unwrap();
        goal.pause();
        let conversation = fixture_conversation().context(Context::default().set_active_goal(goal));
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let context = actual.context.unwrap();
        let actual = (
            context.active_goal.is_some(),
            context.messages.iter().any(|message| {
                matches!(&message.message, ContextMessage::Text(text) if text.is_goal_context())
            }),
        );
        let expected = (true, false);
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_runtime_context_replaces_previous_request_context() {
        let agent = fixture_agent_without_user_prompt();
        let first_event = Event::new("First task");
        let first_conversation = fixture_conversation();
        let first_generator = fixture_generator(agent.clone(), first_event);
        let first_actual = first_generator
            .add_user_prompt(first_conversation)
            .await
            .unwrap();
        let second_event = Event::new("Second task");
        let second_generator = fixture_generator(agent.clone(), second_event);

        let actual = second_generator
            .add_user_prompt(first_actual)
            .await
            .unwrap();

        let runtime_messages = actual
            .context
            .unwrap()
            .messages
            .into_iter()
            .filter(|message| {
                matches!(&message.message, ContextMessage::Text(text) if text.is_runtime_context())
            })
            .count();
        let expected = 1;
        assert_eq!(runtime_messages, expected);
    }

    #[tokio::test]
    async fn test_runtime_context_preserves_untyped_legacy_context_without_kind() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Second task");
        let literal_user_text = "<runtime_context freshness=\"live\" cache=\"uncached\">\n<current_date>2026-05-12</current_date>\n</runtime_context>";
        let conversation = fixture_conversation().context(
            Context::default()
                .add_message(ContextMessage::user("First task", None))
                .add_message(ContextMessage::Text(
                    TextMessage::new(Role::User, literal_user_text)
                        .model(ModelId::new("test-model"))
                        .cacheable(false),
                )),
        );
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        let untyped_literal_messages = messages
            .iter()
            .filter(|message| message.content() == Some(literal_user_text))
            .count();
        let runtime_messages = messages
            .iter()
            .filter(|message| {
                matches!(&message.message, ContextMessage::Text(text) if text.is_runtime_context())
            })
            .count();
        assert_eq!(untyped_literal_messages, 1);
        assert_eq!(runtime_messages, 1);
    }

    #[tokio::test]
    async fn test_runtime_context_replacement_preserves_user_text_that_mentions_runtime_context() {
        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Second task");
        let literal_user_text = "<runtime_context freshness=\"live\" cache=\"uncached\">\nthis is user supplied text, not internal runtime context\n</runtime_context>";
        let conversation = fixture_conversation().context(
            Context::default()
                .add_message(ContextMessage::user("First task", None))
                .add_message(ContextMessage::Text(
                    TextMessage::new(Role::User, literal_user_text)
                        .model(ModelId::new("test-model"))
                        .cacheable(false),
                )),
        );
        let generator = fixture_generator(agent.clone(), event);

        let actual = generator.add_user_prompt(conversation).await.unwrap();

        let messages = actual.context.unwrap().messages;
        let user_literal_messages = messages
            .iter()
            .filter(|message| message.content() == Some(literal_user_text))
            .count();
        let expected = 1;
        assert_eq!(user_literal_messages, expected);
    }

    #[tokio::test]
    async fn test_attachments_tracked_as_read_operations() {
        // Setup - Create a service that returns file attachments
        struct MockServiceWithFiles;

        impl crate::EnvironmentInfra for MockServiceWithFiles {
            type Config = forge_config::ForgeConfig;
            fn get_environment(&self) -> forge_domain::Environment {
                use fake::{Fake, Faker};
                Faker.fake()
            }
            fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
                Ok(forge_config::ForgeConfig::default())
            }
            async fn update_environment(
                &self,
                _ops: Vec<forge_domain::ConfigOperation>,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn get_env_var(&self, _key: &str) -> Option<String> {
                None
            }
            fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
                Default::default()
            }
        }

        #[async_trait::async_trait]
        impl AttachmentService for MockServiceWithFiles {
            async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
                Ok(vec![
                    Attachment {
                        path: "/test/file1.rs".to_string(),
                        content: AttachmentContent::FileContent {
                            content: "fn main() {}".to_string(),
                            info: FileInfo::new(1, 1, 1, "hash1".to_string()),
                        },
                    },
                    Attachment {
                        path: "/test/file2.rs".to_string(),
                        content: AttachmentContent::FileContent {
                            content: "fn test() {}".to_string(),
                            info: FileInfo::new(1, 1, 1, "hash2".to_string()),
                        },
                    },
                ])
            }
        }

        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Task with @[/test/file1.rs] and @[/test/file2.rs]");
        let conversation = Conversation::new(ConversationId::default());
        let generator = UserPromptGenerator::new(
            Arc::new(MockServiceWithFiles),
            agent.clone(),
            event,
            chrono::Local::now(),
        );

        // Execute
        let actual = generator.add_user_prompt(conversation).await.unwrap();

        // Assert - Both files should be tracked as read operations
        let file1_op = actual.metrics.file_operations.get("/test/file1.rs");
        let file2_op = actual.metrics.file_operations.get("/test/file2.rs");

        assert!(file1_op.is_some(), "file1.rs should be tracked in metrics");
        assert!(file2_op.is_some(), "file2.rs should be tracked in metrics");

        // Verify the operation is marked as Read
        let file1_metrics = file1_op.unwrap();
        assert_eq!(
            file1_metrics.tool,
            ToolKind::Read,
            "file1.rs should be tracked as Read operation"
        );
        assert!(
            file1_metrics.content_hash.is_some(),
            "file1.rs should have content hash"
        );

        let file2_metrics = file2_op.unwrap();
        assert_eq!(
            file2_metrics.tool,
            ToolKind::Read,
            "file2.rs should be tracked as Read operation"
        );
        assert!(
            file2_metrics.content_hash.is_some(),
            "file2.rs should have content hash"
        );

        // Verify both files are in files_accessed (since they are Read operations)
        assert!(
            actual.metrics.files_accessed.contains("/test/file1.rs"),
            "file1.rs should be in files_accessed"
        );
        assert!(
            actual.metrics.files_accessed.contains("/test/file2.rs"),
            "file2.rs should be in files_accessed"
        );
    }

    #[tokio::test]
    async fn test_todos_injected_on_resume() {
        // Setup - Simple mock that returns no attachments
        struct MockServiceWithTodos;

        impl crate::EnvironmentInfra for MockServiceWithTodos {
            type Config = forge_config::ForgeConfig;
            fn get_environment(&self) -> forge_domain::Environment {
                use fake::{Fake, Faker};
                Faker.fake()
            }
            fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
                Ok(forge_config::ForgeConfig::default())
            }
            async fn update_environment(
                &self,
                _ops: Vec<forge_domain::ConfigOperation>,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn get_env_var(&self, _key: &str) -> Option<String> {
                None
            }
            fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
                Default::default()
            }
        }

        #[async_trait::async_trait]
        impl AttachmentService for MockServiceWithTodos {
            async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
                Ok(Vec::new())
            }
        }

        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("Continue working");

        // Create a conversation with existing context (simulating resume) and todos
        // stored in metrics
        let conversation = Conversation::new(ConversationId::generate())
            .context(
                Context::default()
                    .add_message(ContextMessage::system("System message"))
                    .add_message(ContextMessage::user("Previous task", None)),
            )
            .metrics(Metrics::default().todos(vec![
                Todo::new("Task 1").status(TodoStatus::Completed),
                Todo::new("Task 2").status(TodoStatus::InProgress),
                Todo::new("Task 3").status(TodoStatus::Pending),
            ]));

        let generator = UserPromptGenerator::new(
            Arc::new(MockServiceWithTodos),
            agent.clone(),
            event,
            chrono::Local::now(),
        );

        // Execute
        let actual = generator.add_user_prompt(conversation).await.unwrap();

        // Assert - Should have system, previous user, new user message, runtime
        // context, and todo list
        let messages = actual.context.unwrap().messages;
        assert_eq!(messages.len(), 5, "Should have 5 messages");

        // First is system message
        assert_eq!(messages[0].content().unwrap(), "System message");

        // Second is previous user task
        assert_eq!(messages[1].content().unwrap(), "Previous task");

        // Third is the new user message
        assert_eq!(messages[2].content().unwrap(), "Continue working");

        // Fourth is runtime context
        assert!(messages[3].content().unwrap().contains("<runtime_context"));
        assert_eq!(messages[3].is_cache_eligible(), false);

        // Fifth should be the todo list (droppable)
        let todo_message = &messages[4];
        assert!(
            todo_message.is_droppable(),
            "Todo message should be droppable"
        );
        let todo_content = todo_message.content().unwrap();
        assert!(
            todo_content.contains("Current task list:"),
            "Should contain task list header"
        );
        assert!(
            todo_content.contains("[DONE] Task 1"),
            "Should contain completed task"
        );
        assert!(
            todo_content.contains("[IN_PROGRESS] Task 2"),
            "Should contain in-progress task"
        );
        assert!(
            todo_content.contains("[PENDING] Task 3"),
            "Should contain pending task"
        );
    }

    #[tokio::test]
    async fn test_todos_not_injected_on_new_conversation() {
        // Setup - Simple mock with no attachments
        struct MockServiceNoTodos;

        impl crate::EnvironmentInfra for MockServiceNoTodos {
            type Config = forge_config::ForgeConfig;
            fn get_environment(&self) -> forge_domain::Environment {
                use fake::{Fake, Faker};
                Faker.fake()
            }
            fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
                Ok(forge_config::ForgeConfig::default())
            }
            async fn update_environment(
                &self,
                _ops: Vec<forge_domain::ConfigOperation>,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn get_env_var(&self, _key: &str) -> Option<String> {
                None
            }
            fn get_env_vars(&self) -> std::collections::BTreeMap<String, String> {
                Default::default()
            }
        }

        #[async_trait::async_trait]
        impl AttachmentService for MockServiceNoTodos {
            async fn attachments(&self, _url: &str) -> anyhow::Result<Vec<Attachment>> {
                Ok(Vec::new())
            }
        }

        let agent = fixture_agent_without_user_prompt();
        let event = Event::new("First task");

        // Create a new conversation (no existing context, no todos)
        let conversation = Conversation::new(ConversationId::generate());

        let generator = UserPromptGenerator::new(
            Arc::new(MockServiceNoTodos),
            agent.clone(),
            event,
            chrono::Local::now(),
        );

        // Execute
        let actual = generator.add_user_prompt(conversation).await.unwrap();

        // Assert - Should have user message and runtime context, no todos
        let messages = actual.context.unwrap().messages;
        assert_eq!(
            messages.len(),
            2,
            "Should have user message and runtime context"
        );
        assert_eq!(messages[0].content().unwrap(), "First task");
        assert!(messages[1].content().unwrap().contains("<runtime_context"));
    }
}
