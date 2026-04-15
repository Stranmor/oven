use std::sync::Arc;

use anyhow::Result;
use forge_domain::{AgentId, ModelId, Provider};

use crate::{AgentRegistry, AppConfigService, ProviderService};

/// Resolver for agent providers and models.
/// Handles provider resolution, credential refresh, and model lookup.
pub struct AgentProviderResolver<S>(Arc<S>);

impl<S> AgentProviderResolver<S> {
    /// Creates a new AgentProviderResolver instance
    pub fn new(services: Arc<S>) -> Self {
        Self(services)
    }
}

impl<S> AgentProviderResolver<S>
where
    S: AgentRegistry + ProviderService + AppConfigService,
{
    async fn resolve_agent_or_session_config<T, F1, F2>(
        &self,
        agent_id: Option<AgentId>,
        extract_from_agent: F1,
        extract_from_config: F2,
    ) -> Result<T>
    where
        F1: FnOnce(forge_domain::Agent) -> T,
        F2: FnOnce(forge_domain::ModelConfig) -> T,
    {
        if let Some(id) = agent_id {
            if let Some(agent) = self.0.get_agent(&id).await? {
                Ok(extract_from_agent(agent))
            } else {
                Err(crate::Error::AgentNotFound(id).into())
            }
        } else {
            self.0
                .get_session_config()
                .await
                .map(extract_from_config)
                .ok_or_else(|| forge_domain::Error::NoDefaultSession.into())
        }
    }

    /// Gets the provider for the specified agent, or the default provider if no
    /// agent is provided. Automatically refreshes OAuth credentials if they're
    /// about to expire.
    pub async fn get_provider(&self, agent_id: Option<AgentId>) -> Result<Provider<url::Url>> {
        let provider_id = self
            .resolve_agent_or_session_config(
                agent_id,
                |agent| agent.provider,
                |config| config.provider,
            )
            .await?;

        let provider = self.0.get_provider(provider_id).await?;
        Ok(provider)
    }

    /// Gets the model for the specified agent, or the default model if no agent
    /// is provided
    pub async fn get_model(&self, agent_id: Option<AgentId>) -> Result<ModelId> {
        self.resolve_agent_or_session_config(agent_id, |agent| agent.model, |config| config.model)
            .await
    }
}
