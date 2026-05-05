use std::sync::Arc;

use anyhow::{Context, Result};
use forge_app::{AgentRepository, DirectoryReaderInfra, EnvironmentInfra, FileInfoInfra};
use forge_config::ForgeConfig;
use forge_domain::{ModelId, ProviderId, Template, ToolName};
use gray_matter::Matter;
use gray_matter::engine::YAML;

use crate::agent_definition::AgentDefinition;

/// Infrastructure implementation for loading agent definitions from multiple
/// sources:
/// 1. Built-in agents (embedded in the application)
/// 2. Global custom agents (from ~/.forge/agents/ directory)
/// 3. Project-local agents (from .forge/agents/ directory in current working
///    directory)
///
/// ## Agent Precedence
/// When agents have duplicate IDs across different sources, the precedence
/// order is: **CWD (project-local) > Global custom > Built-in**
///
/// This means project-local agents can override global agents, and both can
/// override built-in agents.
///
/// ## Directory Resolution
/// - **Built-in agents**: Embedded in application binary
/// - **Global agents**: `~/forge/agents/*.md`
/// - **CWD agents**: `./.forge/agents/*.md` (relative to current working
///   directory)
///
/// Missing directories are handled gracefully and don't prevent loading from
/// other sources.
pub struct ForgeAgentRepository<I> {
    infra: Arc<I>,
}

impl<I> ForgeAgentRepository<I> {
    pub fn new(infra: Arc<I>) -> Self {
        Self { infra }
    }
}

impl<I: FileInfoInfra + EnvironmentInfra<Config = ForgeConfig> + DirectoryReaderInfra>
    ForgeAgentRepository<I>
{
    /// Load all agent definitions from all available sources with conflict
    /// resolution.
    async fn load_agents(&self) -> anyhow::Result<Vec<AgentDefinition>> {
        self.load_all_agents().await
    }

    /// Load all agent definitions from all available sources
    async fn load_all_agents(&self) -> anyhow::Result<Vec<AgentDefinition>> {
        // Load built-in agents (no path - will display as "BUILT IN")
        let mut agents = self.init_default().await?;

        // Load custom agents from global directory
        let dir = self.infra.get_environment().agent_path();
        let custom_agents = self.init_agent_dir(&dir).await?;
        agents.extend(custom_agents);

        // Load custom agents from CWD
        let dir = self.infra.get_environment().agent_cwd_path();
        let cwd_agents = self.init_agent_dir(&dir).await?;
        agents.extend(cwd_agents);

        // Handle agent ID conflicts by keeping the last occurrence
        // This gives precedence order: CWD > Global Custom > Built-in
        Ok(resolve_agent_conflicts(agents))
    }

    async fn init_default(&self) -> anyhow::Result<Vec<AgentDefinition>> {
        let config = self.infra.get_config()?;
        parse_agent_iter(
            [
                ("forge", include_str!("agents/forge.md")),
                ("muse", include_str!("agents/muse.md")),
                ("sage", include_str!("agents/sage.md")),
            ]
            .into_iter()
            .map(|(name, content)| (name.to_string(), content.to_string())),
            &config,
        )
    }

    async fn init_agent_dir(&self, dir: &std::path::Path) -> anyhow::Result<Vec<AgentDefinition>> {
        let config = self.infra.get_config()?;
        if !self.infra.exists(dir).await? {
            return Ok(vec![]);
        }

        // Use DirectoryReaderInfra to read all .md files in parallel
        let files = self
            .infra
            .read_directory_files(dir, Some("*.md"))
            .await
            .with_context(|| format!("Failed to read agents from: {}", dir.display()))?;

        let mut agents = Vec::new();
        for (path, content) in files {
            let mut agent = parse_configured_agent_file(&path, &content, &config)?;

            // Store the file path
            agent.path = Some(path.display().to_string());
            agents.push(agent);
        }

        Ok(agents)
    }
}

/// Implementation function for resolving agent ID conflicts by keeping the last
/// occurrence. This implements the precedence order: CWD Custom > Global Custom
/// > Built-in
fn resolve_agent_conflicts(agents: Vec<AgentDefinition>) -> Vec<AgentDefinition> {
    use std::collections::HashMap;

    // Use HashMap to deduplicate by agent ID, keeping the last occurrence
    let mut agent_map: HashMap<String, AgentDefinition> = HashMap::new();

    for agent in agents {
        agent_map.insert(agent.id.to_string(), agent);
    }

    // Convert back to vector (order is not guaranteed but doesn't matter for the
    // service)
    agent_map.into_values().collect()
}

fn parse_agent_iter<I, Path: AsRef<str>, Content: AsRef<str>>(
    contents: I,
    config: &ForgeConfig,
) -> anyhow::Result<Vec<AgentDefinition>>
where
    I: Iterator<Item = (Path, Content)>,
{
    let mut agents = vec![];

    for (name, content) in contents {
        let agent = apply_subagent_tool_config(parse_agent_file(content.as_ref())?, config)
            .with_context(|| format!("Failed to parse agent: {}", name.as_ref()))?;

        agents.push(agent);
    }

    Ok(agents)
}

fn parse_configured_agent_file(
    path: &std::path::Path,
    content: &str,
    config: &ForgeConfig,
) -> Result<AgentDefinition> {
    let agent = parse_agent_file(content)
        .with_context(|| format!("Failed to parse agent: {}", path.display()))?;
    apply_subagent_tool_config(agent, config)
        .with_context(|| format!("Failed to parse agent: {}", path.display()))
}

fn apply_subagent_tool_config(
    mut agent: AgentDefinition,
    config: &ForgeConfig,
) -> Result<AgentDefinition> {
    if agent.id.as_str() != "forge" {
        return Ok(agent);
    }

    let Some(tools) = agent.tools.as_mut() else {
        return Ok(agent);
    };

    tools.retain(|tool| !matches!(tool.as_str(), "task" | "sage"));

    if config.subagents {
        let insert_index = tools
            .iter()
            .position(|tool| tool.as_str() == "mcp_*")
            .unwrap_or(tools.len());
        tools.insert(insert_index, ToolName::new("task"));
    }

    Ok(agent)
}

/// Parse raw content into an AgentDefinition with YAML frontmatter
fn parse_agent_file(content: &str) -> Result<AgentDefinition> {
    // Parse the frontmatter using gray_matter with type-safe deserialization
    let gray_matter = Matter::<YAML>::new();
    let result = gray_matter.parse::<AgentDefinition>(content)?;

    // Extract the frontmatter
    let agent = result
        .data
        .context("Empty system prompt content")?
        .system_prompt(non_empty_system_prompt(result.content)?);

    Ok(agent)
}

fn non_empty_system_prompt(content: String) -> Result<Template<forge_domain::SystemContext>> {
    if content.trim().is_empty() {
        anyhow::bail!("Empty system prompt content");
    }

    Ok(Template::new(content))
}

#[async_trait::async_trait]
impl<F: FileInfoInfra + EnvironmentInfra<Config = ForgeConfig> + DirectoryReaderInfra>
    AgentRepository for ForgeAgentRepository<F>
{
    async fn get_agents(&self) -> anyhow::Result<Vec<forge_domain::Agent>> {
        let config = self.infra.get_config()?;
        let agent_defs = self.load_agents().await?;

        let session = config
            .session
            .clone()
            .ok_or(forge_domain::Error::NoDefaultSession)?;

        Ok(agent_defs
            .into_iter()
            .map(|def| {
                def.into_agent(
                    ProviderId::from(session.provider_id.clone()),
                    ModelId::from(session.model_id.clone()),
                )
            })
            .collect())
    }

    async fn get_agent_infos(&self) -> anyhow::Result<Vec<forge_domain::AgentInfo>> {
        let agent_defs = self.load_agents().await?;
        Ok(agent_defs
            .into_iter()
            .map(|def| forge_domain::AgentInfo {
                id: def.id,
                title: def.title,
                description: def.description,
            })
            .collect())
    }
}

#[cfg(test)]
#[path = "agent/tests.rs"]
mod tests;
