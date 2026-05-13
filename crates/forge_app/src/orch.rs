use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_recursion::async_recursion;
use derive_setters::Setters;
use forge_domain::{Agent, *};
use forge_template::Element;
use futures::future::join_all;
use tokio::sync::Notify;
use tracing::warn;

use crate::agent::AgentService;
use crate::compact::Compactor;
use crate::transformers::{DropReasoningOnlyMessages, ModelSpecificReasoning};
use crate::{EnvironmentInfra, TemplateEngine};

#[derive(Clone, Setters)]
#[setters(into)]
pub struct Orchestrator<S> {
    services: Arc<S>,
    sender: Option<ArcSender>,
    conversation: Conversation,
    tool_definitions: Vec<ToolDefinition>,
    models: Vec<Model>,
    agent: Agent,
    error_tracker: ToolErrorTracker,
    hook: Arc<Hook>,
    config: forge_config::ForgeConfig,
}

impl<S: AgentService + EnvironmentInfra<Config = forge_config::ForgeConfig>> Orchestrator<S> {
    pub fn new(
        services: Arc<S>,
        conversation: Conversation,
        agent: Agent,
        config: forge_config::ForgeConfig,
    ) -> Self {
        Self {
            conversation,
            services,
            agent,
            config,
            sender: Default::default(),
            tool_definitions: Default::default(),
            models: Default::default(),
            error_tracker: Default::default(),
            hook: Arc::new(Hook::default()),
        }
    }

    /// Get a reference to the internal conversation
    pub fn get_conversation(&self) -> &Conversation {
        &self.conversation
    }

    // Helper function to get all tool results from a vector of tool calls
    #[async_recursion]
    async fn execute_tool_calls(
        &mut self,
        tool_calls: &[ToolCallFull],
        tool_context: &ToolCallContext,
    ) -> anyhow::Result<Vec<(ToolCallFull, ToolResult)>> {
        let task_tool_name = ToolKind::Task.name();

        // Use a case-insensitive comparison since the model may send "Task" or "task".
        let is_task = |tc: &ToolCallFull| {
            tc.name
                .as_str()
                .eq_ignore_ascii_case(task_tool_name.as_str())
        };

        // Partition into task tool calls (run in parallel) and all others (run
        // sequentially). Use a case-insensitive comparison since the model may
        // send "Task" or "task".
        let is_task_call =
            |tc: &&ToolCallFull| tc.name.as_str().to_lowercase() == task_tool_name.as_str();
        let (task_calls, other_calls): (Vec<_>, Vec<_>) = tool_calls.iter().partition(is_task_call);

        // Execute task tool calls in parallel — mirrors how direct agent-as-tool calls
        // work.
        let task_results: Vec<(ToolCallFull, ToolResult)> = join_all(
            task_calls
                .iter()
                .map(|tc| self.services.call(&self.agent, tool_context, (*tc).clone())),
        )
        .await
        .into_iter()
        .zip(task_calls.iter())
        .map(|(result, tc)| ((*tc).clone(), result))
        .collect();

        let system_tools = self
            .tool_definitions
            .iter()
            .map(|tool| &tool.name)
            .collect::<HashSet<_>>();

        // Process non-task tool calls sequentially (preserving UI notifier handshake
        // and hooks).
        let mut other_results: Vec<(ToolCallFull, ToolResult)> =
            Vec::with_capacity(other_calls.len());
        for tool_call in &other_calls {
            // Send the start notification for system tools and not agent as a tool
            let is_system_tool = system_tools.contains(&tool_call.name);
            if is_system_tool {
                let notifier = Arc::new(Notify::new());
                self.send(ChatResponse::ToolCallStart {
                    tool_call: (*tool_call).clone(),
                    notifier: notifier.clone(),
                })
                .await?;
                // Wait for the UI to acknowledge it has rendered the tool header
                // before we execute the tool. This prevents tool stdout from
                // appearing before the tool name is printed.
                notifier.notified().await;
            }

            // Fire the ToolcallStart lifecycle event
            let toolcall_start_event = LifecycleEvent::ToolcallStart(EventData::new(
                self.agent.clone(),
                self.agent.model.clone(),
                ToolcallStartPayload::new((*tool_call).clone()),
            ));
            self.hook
                .handle(&toolcall_start_event, &mut self.conversation)
                .await?;

            // Execute the tool
            let tool_result = self
                .services
                .call(&self.agent, tool_context, (*tool_call).clone())
                .await;

            // Fire the ToolcallEnd lifecycle event (fires on both success and failure)
            let toolcall_end_event = LifecycleEvent::ToolcallEnd(EventData::new(
                self.agent.clone(),
                self.agent.model.clone(),
                ToolcallEndPayload::new((*tool_call).clone(), tool_result.clone()),
            ));
            self.hook
                .handle(&toolcall_end_event, &mut self.conversation)
                .await?;

            // Send the end notification for system tools and not agent as a tool
            if is_system_tool {
                self.send(ChatResponse::ToolCallEnd(tool_result.clone()))
                    .await?;
            }
            other_results.push(((*tool_call).clone(), tool_result));
        }

        // Reconstruct results in the original order of tool_calls.
        let mut task_iter = task_results.into_iter();
        let mut other_iter = other_results.into_iter();
        let tool_call_records = tool_calls
            .iter()
            .map(|tc| {
                if is_task(tc) {
                    task_iter.next().expect("task result count mismatch")
                } else {
                    other_iter.next().expect("other result count mismatch")
                }
            })
            .collect();

        Ok(tool_call_records)
    }

    async fn send(&self, message: ChatResponse) -> anyhow::Result<()> {
        if let Some(sender) = &self.sender {
            sender.send(Ok(message)).await?
        }
        Ok(())
    }

    fn model_for_agent(&self) -> Option<&Model> {
        self.models
            .iter()
            .find(|model| model.id == self.agent.model && model.provider_id == self.agent.provider)
    }

    /// Returns the configured output token reservation for a request.
    fn output_token_reservation(context: &Context) -> usize {
        const DEFAULT_OUTPUT_TOKEN_RESERVATION: usize = 4_096;

        context
            .max_tokens
            .unwrap_or(DEFAULT_OUTPUT_TOKEN_RESERVATION)
    }

    /// Returns the non-message token estimate that is serialized with the request.
    fn request_overhead_token_count(context: &Context) -> usize {
        let tool_tokens = context
            .tools
            .iter()
            .map(|tool| {
                serde_json::to_string(tool)
                    .unwrap_or_default()
                    .chars()
                    .count()
                    .div_ceil(4)
            })
            .sum::<usize>();

        tool_tokens + context.messages.len().saturating_mul(4)
    }

    /// Returns the estimated token count for the outbound provider request.
    fn estimated_request_tokens(context: &Context) -> usize {
        context
            .token_count_approx()
            .saturating_add(Self::request_overhead_token_count(context))
    }

    /// Returns a conservative provider context-window safety margin.
    fn context_window_safety_margin(context_window: usize) -> usize {
        const MIN_MARGIN: usize = 4_096;
        const MAX_MARGIN: usize = 32_768;
        const PERCENTAGE: usize = 20;

        context_window
            .saturating_mul(PERCENTAGE)
            .saturating_div(100)
            .clamp(MIN_MARGIN, MAX_MARGIN)
    }

    /// Returns the maximum input token budget after reserving output and margin.
    fn effective_input_budget(context_window: usize, output_reservation: usize) -> Option<usize> {
        let safety_margin = Self::context_window_safety_margin(context_window);
        context_window
            .checked_sub(output_reservation)?
            .checked_sub(safety_margin)
    }

    /// Compacts context before provider dispatch when the projected request would exceed the model window.
    ///
    /// # Arguments
    /// * `context` - The fully transformed context that would be sent to the provider.
    ///
    /// # Errors
    /// Returns an actionable local error when the request cannot fit inside the selected model window.
    fn preflight_context_window(&self, context: Context) -> anyhow::Result<Context> {
        let Some(context_window) = self
            .model_for_agent()
            .and_then(|model| model.context_length)
            .and_then(|value| usize::try_from(value).ok())
        else {
            return Ok(context);
        };

        let output_reservation = Self::output_token_reservation(&context);
        let input_budget = Self::effective_input_budget(context_window, output_reservation)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Selected model '{}' has a {} token context window, but the configured output reservation is {} tokens and leaves no safe prompt budget. Lower max_tokens or select a larger-context model.",
                    self.agent.model,
                    context_window,
                    output_reservation
                )
            })?;

        let estimated_tokens = Self::estimated_request_tokens(&context);
        if estimated_tokens <= input_budget {
            return Ok(context);
        }

        let compacted = Compactor::new(self.agent.compact.clone(), self.services.get_environment())
            .compact(context, true)?;
        let compacted_estimated_tokens = Self::estimated_request_tokens(&compacted);
        if compacted_estimated_tokens <= input_budget {
            return Ok(compacted);
        }

        anyhow::bail!(
            "Local context-window guard blocked an oversized request before provider dispatch. Model '{}' has context window {} tokens; reserved output is {} tokens; safety margin is {} tokens; effective input budget is {} tokens; estimated request is {} tokens after compaction. Reduce context, lower max_tokens, or select a larger-context model.",
            self.agent.model,
            context_window,
            output_reservation,
            Self::context_window_safety_margin(context_window),
            input_budget,
            compacted_estimated_tokens
        )
    }

    // Returns if agent supports tool or not.
    fn is_tool_supported(&self) -> anyhow::Result<bool> {
        // Check if at agent level tool support is defined
        let tool_supported = match self.agent.tool_supported {
            Some(tool_supported) => tool_supported,
            None => {
                // If not defined at agent level, check model level

                let model = self.model_for_agent();
                model
                    .and_then(|model| model.tools_supported)
                    .unwrap_or_default()
            }
        };

        Ok(tool_supported)
    }

    async fn execute_chat_turn(
        &self,
        model_id: &ModelId,
        context: Context,
        reasoning_supported: bool,
    ) -> anyhow::Result<ChatCompletionMessageFull> {
        let tool_supported = self.is_tool_supported()?;
        let mut transformers = DefaultTransformation::default()
            .pipe(SortTools::new(self.agent.tool_order()))
            .pipe(NormalizeToolCallArguments::new())
            .pipe(TransformToolCalls::new().when(|_| !tool_supported))
            .pipe(ImageHandling::new())
            // Drop ALL reasoning (including config) when reasoning is not supported by the model
            .pipe(DropReasoningDetails.when(|_| !reasoning_supported))
            // Strip all reasoning from messages when the model has changed (signatures are
            // model-specific and invalid across models). No-op when model is unchanged.
            .pipe(ReasoningNormalizer::new(model_id.clone()))
            // Normalize Anthropic reasoning knobs per model family before provider conversion.
            .pipe(
                ModelSpecificReasoning::new(model_id.as_str())
                    .when(|_| model_id.as_str().to_lowercase().contains("claude")),
            )
            // Drop reasoning-only assistant turns; Anthropic and Bedrock both reject
            // messages whose final content block is `thinking`.
            .pipe(
                DropReasoningOnlyMessages
                    .when(|_| model_id.as_str().to_lowercase().contains("claude")),
            );
        let context = context.initiator(self.conversation.initiator);
        let context = transformers.transform(context);
        let response = self
            .services
            .chat_agent(model_id, context, Some(self.agent.provider.clone()))
            .await?;

        // Always stream content deltas
        response
            .into_full_streaming(!tool_supported, self.sender.clone())
            .await
    }

    async fn execute_chat_turn_vetted(
        &self,
        model_id: &ModelId,
        context: Context,
        reasoning_supported: bool,
    ) -> anyhow::Result<ChatCompletionMessageFull> {
        let msg = self
            .execute_chat_turn(model_id, context, reasoning_supported)
            .await?;

        let trimmed = msg.content.trim();

        if msg.tool_calls.is_empty() {
            // 1. Completely empty response
            let is_empty = trimmed.is_empty();

            // 2. Short generative garbage / parsing artifacts
            let is_short_garbage = matches!(
                trimmed,
                "}" | "{" | "]" | "[" | "```" | "```json" | "```json\n```"
            );

            // 3. Raw JSON/Markdown hallucination (model output tool call syntax as raw
            //    text)
            let has_tool_keywords = trimmed.contains("\"name\"")
                && (trimmed.contains("\"arguments\"")
                    || trimmed.contains("\"tool_calls\"")
                    || trimmed.contains("\"function_call\""));
            let is_json_hallucination =
                (trimmed.starts_with('{') || trimmed.starts_with("```json")) && has_tool_keywords;

            if is_empty || is_short_garbage || is_json_hallucination {
                return Err(anyhow::anyhow!(forge_domain::Error::Retryable(
                    anyhow::anyhow!(
                        "Model hallucination detected (empty, garbage, or unparsed JSON). Triggering retry. Output: {:?}",
                        trimmed
                    )
                )));
            }
        }

        Ok(msg)
    }

    // Create a helper method with the core functionality
    pub async fn run(&mut self) -> anyhow::Result<()> {
        let model_id = self.get_model();

        let mut context = self.conversation.context.clone().unwrap_or_default();

        // Fire the Start lifecycle event
        let start_event = LifecycleEvent::Start(EventData::new(
            self.agent.clone(),
            model_id.clone(),
            StartPayload,
        ));
        self.hook
            .handle(&start_event, &mut self.conversation)
            .await?;

        // Signals that the loop should suspend (task may or may not be completed)
        let mut should_yield = false;

        // Signals that the task is completed
        let mut is_complete = false;

        let mut request_count = 0;

        // Retrieve the number of requests allowed per tick.
        let max_requests_per_turn = self.agent.max_requests_per_turn;
        let tool_context = ToolCallContext::new(self.conversation.metrics.clone())
            .sender(self.sender.clone())
            .conversation_id(Some(self.conversation.id));

        while !should_yield {
            // Set context for the current loop iteration
            self.conversation.context = Some(context.clone());
            self.services.update(self.conversation.clone()).await?;

            let request_event = LifecycleEvent::Request(EventData::new(
                self.agent.clone(),
                model_id.clone(),
                RequestPayload::new(request_count),
            ));
            self.hook
                .handle(&request_event, &mut self.conversation)
                .await?;
            if let Some(updated_context) = &self.conversation.context {
                context = updated_context.clone();
            }

            context = self.preflight_context_window(context)?;
            self.conversation.context = Some(context.clone());
            self.services.update(self.conversation.clone()).await?;

            let message = crate::retry::retry_with_config(
                &self.config.clone().retry.unwrap_or_default(),
                || {
                    self.execute_chat_turn_vetted(
                        &model_id,
                        context.clone(),
                        context.is_reasoning_supported(),
                    )
                },
                self.sender.as_ref().map(|sender| {
                    let sender = sender.clone();
                    let agent_id = self.agent.id.clone();
                    let model_id = model_id.clone();
                    move |error: &anyhow::Error, duration: Duration| {
                        let root_cause = error.root_cause();
                        // Log retry attempts - critical for debugging API failures
                        tracing::error!(
                            agent_id = %agent_id,
                            error = ?root_cause,
                            model = %model_id,
                            "Retry attempt due to error"
                        );
                        let retry_event =
                            ChatResponse::RetryAttempt { cause: error.into(), duration };
                        let _ = sender.try_send(Ok(retry_event));
                    }
                }),
            )
            .await?;

            // Fire the Response lifecycle event
            let response_event = LifecycleEvent::Response(EventData::new(
                self.agent.clone(),
                model_id.clone(),
                ResponsePayload::new(message.clone()),
            ));
            self.hook
                .handle(&response_event, &mut self.conversation)
                .await?;

            // Turn is completed, if finish_reason is 'stop'. Gemini models return stop as
            // finish reason with tool calls.
            let is_stop_reason =
                message.finish_reason == Some(FinishReason::Stop) && message.tool_calls.is_empty();

            // We must also yield if the response has no tool calls AND no content,
            // otherwise we will loop indefinitely sending empty text back.
            let is_empty_loop = message.tool_calls.is_empty() && message.content.is_empty();

            is_complete = is_stop_reason || is_empty_loop;

            // Should yield if a tool is asking for a follow-up
            should_yield = is_complete
                || message
                    .tool_calls
                    .iter()
                    .any(|call| ToolCatalog::should_yield(&call.name));

            // Process tool calls and update context
            let mut tool_call_records = self
                .execute_tool_calls(&message.tool_calls, &tool_context)
                .await?;

            // Update context from conversation after response / tool-call hooks run
            if let Some(updated_context) = &self.conversation.context {
                context = updated_context.clone();
            }

            self.error_tracker.adjust_record(&tool_call_records);
            let allowed_max_attempts = self.error_tracker.limit();
            for (_, result) in tool_call_records.iter_mut() {
                if result.is_error() {
                    let attempts_left = self.error_tracker.remaining_attempts(&result.name);
                    // Add attempt information to the error message so the agent can reflect on it.
                    let context = serde_json::json!({
                        "attempts_left": attempts_left,
                        "allowed_max_attempts": allowed_max_attempts,
                    });
                    let text = TemplateEngine::default()
                        .render("forge-tool-retry-message.md", &context)?;
                    let message = Element::new("retry").text(text);

                    result.output.combine_mut(ToolOutput::text(message));
                }
            }

            context = context.append_message(
                message.content.clone(),
                message.thought_signature.clone(),
                message.reasoning.clone(),
                message.reasoning_details.clone(),
                message.usage,
                tool_call_records,
                message.phase,
            );

            if self
                .services
                .is_primary_conversation(&self.conversation.id)
                .await?
            {
                let steer_messages = self
                    .services
                    .drain_steer_messages(&self.conversation.id)
                    .await?;
                for steer_message in steer_messages {
                    let content = Element::new("steer").text(steer_message.content()).render();
                    context =
                        context.add_message(ContextMessage::user(content, Some(model_id.clone())));
                }
            }

            if self.error_tracker.limit_reached() {
                self.send(ChatResponse::Interrupt {
                    reason: InterruptionReason::MaxToolFailurePerTurnLimitReached {
                        limit: *self.error_tracker.limit() as u64,
                        errors: self.error_tracker.errors().clone(),
                    },
                })
                .await?;
                // Should yield if too many errors are produced
                should_yield = true;
            }

            // Update context in the conversation
            context = SetModel::new(model_id.clone()).transform(context);
            self.conversation.context = Some(context.clone());
            self.services.update(self.conversation.clone()).await?;
            request_count += 1;

            if !should_yield && let Some(max_request_allowed) = max_requests_per_turn {
                // Check if agent has reached the maximum request per turn limit
                if request_count >= max_request_allowed {
                    // Log warning - important for understanding conversation interruptions
                    warn!(
                        agent_id = %self.agent.id,
                        model_id = %model_id,
                        request_count,
                        max_request_allowed,
                        "Agent has reached the maximum request per turn limit"
                    );
                    // raise an interrupt event to notify the UI
                    self.send(ChatResponse::Interrupt {
                        reason: InterruptionReason::MaxRequestPerTurnLimitReached {
                            limit: max_request_allowed as u64,
                        },
                    })
                    .await?;
                    // force completion
                    should_yield = true;
                }
            }

            // Update metrics in conversation
            tool_context.with_metrics(|metrics| {
                self.conversation.metrics = metrics.clone();
            })?;

            // If completing (should_yield is due), fire End hook and check if
            // it adds messages
            if should_yield {
                let end_count_before = self.conversation.len();
                self.hook
                    .handle(
                        &LifecycleEvent::End(EventData::new(
                            self.agent.clone(),
                            model_id.clone(),
                            EndPayload,
                        )),
                        &mut self.conversation,
                    )
                    .await?;
                self.services.update(self.conversation.clone()).await?;
                // Check if End hook added messages - if so, continue the loop
                if self.conversation.len() > end_count_before {
                    // End hook added messages, sync context and continue
                    if let Some(updated_context) = &self.conversation.context {
                        context = updated_context.clone();
                    }
                    should_yield = false;
                }
            }
        }

        self.services.update(self.conversation.clone()).await?;

        // Signal Task Completion
        if is_complete {
            self.send(ChatResponse::TaskComplete).await?;
        }

        Ok(())
    }

    fn get_model(&self) -> ModelId {
        self.agent.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use forge_domain::{
        Agent, AgentId, ChatCompletionMessage, Compact, Context, ContextMessage, Environment,
        InputModality, Model, ModelId, ProviderId, ResultStream, Role, TextMessage,
        ToolCallContext, ToolCallFull, ToolDefinition, ToolResult,
    };
    use pretty_assertions::assert_eq;

    use super::Orchestrator;
    use crate::{AgentService, EnvironmentInfra};

    struct FixtureServices;

    #[async_trait::async_trait]
    impl AgentService for FixtureServices {
        async fn chat_agent(
            &self,
            _id: &ModelId,
            _context: Context,
            _provider_id: Option<ProviderId>,
        ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
            unimplemented!()
        }

        async fn call(
            &self,
            _agent: &Agent,
            _context: &ToolCallContext,
            _call: ToolCallFull,
        ) -> ToolResult {
            unimplemented!()
        }

        async fn update(&self, _conversation: forge_domain::Conversation) -> anyhow::Result<()> {
            unimplemented!()
        }
    }

    impl EnvironmentInfra for FixtureServices {
        type Config = forge_config::ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            unimplemented!()
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            unimplemented!()
        }

        fn get_environment(&self) -> Environment {
            environment_fixture()
        }

        fn get_config(&self) -> anyhow::Result<Self::Config> {
            unimplemented!()
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            unimplemented!()
        }
    }

    fn large_text(tokens: usize) -> String {
        "x".repeat(tokens.saturating_mul(4))
    }

    fn environment_fixture() -> Environment {
        Environment {
            os: "linux".to_string(),
            cwd: "/workspace".into(),
            home: Some("/home/test".into()),
            shell: "/bin/sh".to_string(),
            base_path: "/tmp/forge".into(),
        }
    }

    fn model_fixture(context_length: u64) -> Model {
        Model {
            id: ModelId::new("context-guard-model"),
            provider_id: ProviderId::OPENAI,
            name: None,
            description: None,
            context_length: Some(context_length),
            tools_supported: Some(true),
            supports_parallel_tool_calls: Some(true),
            supports_reasoning: Some(false),
            input_modalities: vec![InputModality::Text],
        }
    }

    fn orchestrator_fixture(
        compact: Compact,
        context_length: u64,
    ) -> Orchestrator<FixtureServices> {
        let agent = Agent::new(
            AgentId::new("context_guard_agent"),
            ProviderId::OPENAI,
            ModelId::new("context-guard-model"),
        )
        .compact(compact);

        Orchestrator::new(
            Arc::new(FixtureServices),
            forge_domain::Conversation::generate(),
            agent,
            forge_config::ForgeConfig::default(),
        )
        .models(vec![model_fixture(context_length)])
    }

    fn droppable_user_message(content: String) -> ContextMessage {
        TextMessage::new(Role::User, content).droppable(true).into()
    }

    #[test]
    fn test_preflight_blocks_oversized_request_after_max_compaction() {
        let fixture = orchestrator_fixture(Compact::new().retention_window(1_usize), 8_000);
        let context = Context::default()
            .add_message(ContextMessage::user(large_text(7_000), None))
            .max_tokens(512_usize);

        let actual = fixture.preflight_context_window(context);

        assert!(actual.is_err());
    }

    #[test]
    fn test_preflight_returns_compacted_context_before_provider_dispatch() {
        let fixture = orchestrator_fixture(Compact::new().retention_window(1_usize), 40_000);
        let context = Context::default()
            .add_message(ContextMessage::system("system prompt"))
            .add_message(droppable_user_message(large_text(35_000)))
            .add_message(ContextMessage::user("fresh user request", None))
            .max_tokens(2_000_usize);

        let original_estimated_tokens =
            Orchestrator::<FixtureServices>::estimated_request_tokens(&context);
        let actual = fixture.preflight_context_window(context).unwrap();
        let input_budget =
            Orchestrator::<FixtureServices>::effective_input_budget(40_000, 2_000).unwrap();
        let expected = true;

        assert_eq!(
            Orchestrator::<FixtureServices>::estimated_request_tokens(&actual)
                < original_estimated_tokens,
            expected
        );
        assert_eq!(
            Orchestrator::<FixtureServices>::estimated_request_tokens(&actual) <= input_budget,
            expected
        );
    }

    #[test]
    fn test_preflight_skips_unknown_context_window_without_data_loss() {
        let fixture =
            orchestrator_fixture(Compact::new().retention_window(1_usize), 40_000).models(vec![
                Model { context_length: None, ..model_fixture(40_000) },
            ]);
        let context = Context::default()
            .add_message(ContextMessage::user(large_text(50_000), None))
            .max_tokens(2_000_usize);

        let actual = fixture.preflight_context_window(context.clone()).unwrap();
        let expected = context;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_preflight_requires_matching_provider_model() {
        let fixture =
            orchestrator_fixture(Compact::new().retention_window(1_usize), 40_000).models(vec![
                Model { provider_id: ProviderId::ANTHROPIC, ..model_fixture(40_000) },
            ]);
        let context = Context::default()
            .add_message(ContextMessage::user(large_text(50_000), None))
            .max_tokens(2_000_usize);

        let actual = fixture.preflight_context_window(context.clone()).unwrap();
        let expected = context;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_preflight_blocks_266k_window_230k_prompt_60k_output_request() {
        let fixture = orchestrator_fixture(Compact::new().retention_window(1_usize), 266_300);
        let context = Context::default()
            .add_message(ContextMessage::user(large_text(230_000), None))
            .max_tokens(60_000_usize);

        let actual = fixture.preflight_context_window(context);

        assert!(actual.is_err());
    }

    #[test]
    fn test_default_output_reservation_is_conservative_when_max_tokens_is_missing() {
        let fixture = Context::default();

        let actual = Orchestrator::<FixtureServices>::output_token_reservation(&fixture);
        let expected = 4_096;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_effective_input_budget_subtracts_output_reservation_and_margin() {
        let fixture_context_window = 266_300;
        let fixture_output_reservation = 60_000;

        let actual = Orchestrator::<FixtureServices>::effective_input_budget(
            fixture_context_window,
            fixture_output_reservation,
        );
        let expected = Some(173_532);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_context_window_safety_margin_is_capped_for_large_windows() {
        let fixture = 266_300;

        let actual = Orchestrator::<FixtureServices>::context_window_safety_margin(fixture);
        let expected = 32_768;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_regression_230k_threshold_with_60k_output_exceeds_266k_window_budget() {
        let fixture = Context::default()
            .add_message(ContextMessage::user(large_text(230_000), None))
            .max_tokens(60_000_usize);
        let input_budget =
            Orchestrator::<FixtureServices>::effective_input_budget(266_300, 60_000).unwrap();

        let actual =
            Orchestrator::<FixtureServices>::estimated_request_tokens(&fixture) > input_budget;
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_request_estimate_includes_tool_definitions() {
        let fixture = Context::default()
            .add_message(ContextMessage::user("short", None))
            .add_tool(
                ToolDefinition::new("large_tool")
                    .description(large_text(1_000))
                    .input_schema(schemars::schema_for!(())),
            );

        let actual = Orchestrator::<FixtureServices>::estimated_request_tokens(&fixture)
            > fixture.token_count_approx();
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_effective_input_budget_returns_none_when_output_reservation_consumes_window() {
        let fixture_context_window = 8_000;
        let fixture_output_reservation = 7_000;

        let actual = Orchestrator::<FixtureServices>::effective_input_budget(
            fixture_context_window,
            fixture_output_reservation,
        );
        let expected = None;

        assert_eq!(actual, expected);
    }
}
