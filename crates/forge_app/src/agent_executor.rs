use std::sync::Arc;

use anyhow::Context;
use convert_case::{Case, Casing};
use forge_domain::{
    AgentId, ChatRequest, ChatResponse, ChatResponseContent, Conversation, ConversationId, Event,
    SubagentTaskSession, TitleFormat, ToolCallContext, ToolDefinition, ToolName, ToolOutput,
};
use forge_template::Element;
use futures::StreamExt;
use tokio::sync::RwLock;

use crate::error::Error;
use crate::{AgentRegistry, ConversationService, EnvironmentInfra, Services, SteerService};
#[derive(Clone)]
pub struct AgentExecutor<S> {
    services: Arc<S>,
    pub tool_agents: Arc<RwLock<Option<Vec<ToolDefinition>>>>,
}

impl<S: Services + EnvironmentInfra<Config = forge_config::ForgeConfig>> AgentExecutor<S> {
    pub fn new(services: Arc<S>) -> Self {
        Self { services, tool_agents: Arc::new(RwLock::new(None)) }
    }

    /// Returns a list of tool definitions for all available agents.
    pub async fn agent_definitions(&self) -> anyhow::Result<Vec<ToolDefinition>> {
        if let Some(tool_agents) = self.tool_agents.read().await.clone() {
            return Ok(tool_agents);
        }
        let agents = self.services.get_agents().await?;
        let tools: Vec<ToolDefinition> = agents.into_iter().map(Into::into).collect();
        *self.tool_agents.write().await = Some(tools.clone());
        Ok(tools)
    }

    /// Executes an agent tool call by creating a new chat request for the
    /// specified agent. If conversation_id is provided, the agent will reuse
    /// that conversation only when it is already owned by the same parent or is
    /// parentless compatibility state. Otherwise, a new conversation is created.
    pub async fn execute(
        &self,
        agent_id: AgentId,
        task: String,
        ctx: &ToolCallContext,
        conversation_id: Option<ConversationId>,
    ) -> anyhow::Result<ToolOutput> {
        ctx.send_tool_input(
            TitleFormat::debug(format!(
                "{} [Agent]",
                agent_id.as_str().to_case(Case::UpperSnake)
            ))
            .sub_title(task.as_str()),
        )
        .await?;

        let parent_id = ctx.conversation_id;
        let conversation = if let Some(conversation_id) = conversation_id {
            self.services
                .ensure_delegated_conversation(&conversation_id, parent_id)
                .await?
        } else {
            let mut conversation = Conversation::generate()
                .title(task.clone())
                .initiator(forge_domain::Initiator::Agent)
                .context(forge_domain::Context::default());
            conversation.ensure_delegated(parent_id);
            self.services
                .conversation_service()
                .upsert_conversation(conversation.clone())
                .await?;
            conversation
        };
        self.services.clear_steer(&conversation.id).await?;
        let root_id = match parent_id {
            Some(parent_id) => self
                .services
                .find_conversation(&parent_id)
                .await?
                .and_then(|parent| parent.parent_id)
                .or(Some(parent_id)),
            None => None,
        };
        let mut task_session = if let Some(existing) = self
            .services
            .get_subagent_task_session_by_conversation(&conversation.id)
            .await?
        {
            if existing.parent_conversation_id.is_some()
                && existing.parent_conversation_id != parent_id
            {
                anyhow::bail!(
                    "Subagent session {} belongs to parent {:?}; refusing silent reparent to {:?}",
                    conversation.id,
                    existing.parent_conversation_id,
                    parent_id
                );
            }
            existing.task(task.clone()).agent_id(agent_id.clone())
        } else {
            SubagentTaskSession::new(
                agent_id.clone(),
                conversation.id,
                parent_id,
                root_id,
                task.clone(),
            )
        };
        task_session.mark_running();
        self.services
            .upsert_subagent_task_session(task_session.clone())
            .await?;
        // Execute the request through the ForgeApp
        let app = crate::ForgeApp::new(self.services.clone());
        let mut response_stream = match app
            .chat(
                agent_id.clone(),
                ChatRequest::new(Event::new(task.clone()), conversation.id),
            )
            .await
        {
            Ok(response_stream) => response_stream,
            Err(error) => {
                task_session.mark_failed(error.to_string());
                self.services
                    .upsert_subagent_task_session(task_session)
                    .await?;
                return Err(error);
            }
        };

        // Collect responses from the agent
        let mut output = String::new();
        while let Some(message) = response_stream.next().await {
            let message = match message {
                Ok(message) => message,
                Err(error) => {
                    task_session.mark_failed(error.to_string());
                    self.services
                        .upsert_subagent_task_session(task_session)
                        .await?;
                    return Err(error);
                }
            };
            if matches!(&message, ChatResponse::ToolCallEnd(_)) {
                task_session.heartbeat();
                self.services
                    .upsert_subagent_task_session(task_session.clone())
                    .await?;
            }
            match message {
                ChatResponse::TaskMessage { ref content } => match content {
                    ChatResponseContent::ToolInput(_) => ctx.send(message).await?,
                    ChatResponseContent::ToolOutput(_) => {}
                    ChatResponseContent::Markdown { text, partial } => {
                        if *partial {
                            output.push_str(text);
                        } else {
                            output = text.to_string();
                        }
                    }
                },
                ChatResponse::TaskReasoning { .. } => {}
                ChatResponse::TaskComplete => {}
                ChatResponse::ToolCallStart { .. } => ctx.send(message).await?,
                ChatResponse::ToolCallEnd(_) => ctx.send(message).await?,
                ChatResponse::RetryAttempt { .. } => ctx.send(message).await?,
                ChatResponse::Interrupt { reason } => {
                    task_session.mark_interrupted(reason.to_string());
                    self.services
                        .upsert_subagent_task_session(task_session.clone())
                        .await?;
                    return Err(Error::AgentToolInterrupted(reason))
                        .context(format!(
                            "Tool call to '{}' failed.\n\
                             Note: This is an AGENTIC tool (powered by an LLM), not a traditional function.\n\
                             The failure occurred because the underlying LLM did not behave as expected.\n\
                             This is typically caused by model limitations, prompt issues, or reaching safety limits.",
                            agent_id.as_str()
                        ));
                }
            }
        }
        if !output.trim().is_empty() {
            task_session.mark_completed(output.clone());
            self.services
                .upsert_subagent_task_session(task_session.clone())
                .await?;
            let tool_output = ToolOutput::ai_task(
                conversation.id,
                task_session.task_id,
                Element::new("task_completed")
                    .attr("task_id", task_session.task_id.into_string())
                    .attr("conversation_id", conversation.id.into_string())
                    .attr("session_id", conversation.id.into_string())
                    .attr("task", &task)
                    .append(Element::new("output").text(output)),
            );
            task_session.mark_delivered();
            self.services
                .upsert_subagent_task_session(task_session)
                .await?;
            Ok(tool_output)
        } else {
            task_session.mark_failed("Empty tool response");
            self.services
                .upsert_subagent_task_session(task_session)
                .await?;
            Err(Error::EmptyToolResponse.into())
        }
    }

    pub async fn contains_tool(&self, tool_name: &ToolName) -> anyhow::Result<bool> {
        let agent_tools = self.agent_definitions().await?;
        Ok(agent_tools.iter().any(|tool| tool.name == *tool_name))
    }
}
