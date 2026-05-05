use forge_config::ForgeConfig;
use forge_domain::AgentId;
use insta::{assert_snapshot, assert_yaml_snapshot, with_settings};
use pretty_assertions::assert_eq;

use super::*;

macro_rules! assert_agent_snapshot {
    ($assertion:expr) => {
        with_settings!({snapshot_path => "../snapshots"}, {
            $assertion;
        });
    };
}

#[tokio::test]
async fn test_parse_basic_agent() {
    let content = forge_test_kit::fixture!("/src/fixtures/agents/basic.md").await;

    let actual = parse_agent_file(&content).unwrap();

    assert_eq!(actual.id.as_str(), "test-basic");
    assert_eq!(actual.title.as_ref().unwrap(), "Basic Test Agent");
    assert_eq!(
        actual.description.as_ref().unwrap(),
        "A simple test agent for basic functionality"
    );
    assert_eq!(
        actual.system_prompt.as_ref().unwrap().template,
        "This is a basic test agent used for testing fundamental functionality."
    );
}

#[tokio::test]
async fn test_parse_advanced_agent() {
    let content = forge_test_kit::fixture!("/src/fixtures/agents/advanced.md").await;

    let actual = parse_agent_file(&content).unwrap();

    assert_eq!(actual.id.as_str(), "test-advanced");
    assert_eq!(actual.title.as_ref().unwrap(), "Advanced Test Agent");
    assert_eq!(
        actual.description.as_ref().unwrap(),
        "An advanced test agent with full configuration"
    );
}

#[test]
fn test_parse_configured_agent_file_reports_source_path_for_empty_file() {
    let setup = std::path::PathBuf::from("/tmp/empty-agent.md");

    let actual = parse_configured_agent_file(&setup, "", &ForgeConfig::default()).unwrap_err();
    let expected = "Failed to parse agent: /tmp/empty-agent.md";

    assert!(actual.to_string().contains(expected));
}

#[test]
fn test_parse_configured_agent_file_reports_source_path_for_invalid_frontmatter() {
    let setup = std::path::PathBuf::from("/tmp/invalid-agent.md");
    let fixture = r#"---
id: [
---
Invalid body.
"#;

    let actual = parse_configured_agent_file(&setup, fixture, &ForgeConfig::default()).unwrap_err();
    let expected = "Failed to parse agent: /tmp/invalid-agent.md";

    assert!(actual.to_string().contains(expected));
}

#[test]
fn test_parse_configured_agent_file_rejects_missing_body_with_source_path() {
    let setup = std::path::PathBuf::from("/tmp/missing-body-agent.md");
    let fixture = r#"---
id: "missing-body"
---
"#;

    let actual = parse_configured_agent_file(&setup, fixture, &ForgeConfig::default()).unwrap_err();
    let expected = "Failed to parse agent: /tmp/missing-body-agent.md";

    assert!(actual.to_string().contains(expected));
}

#[test]
fn test_parse_configured_agent_file_error_chain_keeps_path_and_root_cause() {
    let setup = std::path::PathBuf::from("/tmp/empty-agent.md");

    let actual = parse_configured_agent_file(&setup, "", &ForgeConfig::default()).unwrap_err();
    let actual_chain = actual
        .chain()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>();
    let expected = vec![
        "Failed to parse agent: /tmp/empty-agent.md".to_string(),
        "Empty system prompt content".to_string(),
    ];

    assert_eq!(actual_chain, expected);
}

#[test]
fn test_parse_builtin_forge_agent_file_is_valid() {
    let fixture = include_str!("../agents/forge.md");

    let actual = parse_agent_file(fixture).unwrap();

    assert_eq!(actual.id, AgentId::new("forge"));
    assert!(
        actual
            .system_prompt
            .unwrap()
            .template
            .contains("You are Forge")
    );
}

#[test]
fn test_parse_agent_file_renders_conditional_frontmatter_when_subagents_enabled() {
    let fixture = r#"---
id: "forge"
tools:
  - read
  - task
  - sage
  - mcp_*
---
Body keeps {{tool_names.read}} untouched.
"#;
    let config = ForgeConfig { subagents: true, ..Default::default() };

    let actual = apply_subagent_tool_config(parse_agent_file(fixture).unwrap(), &config).unwrap();

    assert_eq!(actual.id, AgentId::new("forge"));
    assert_eq!(
        actual.system_prompt.unwrap().template,
        "Body keeps {{tool_names.read}} untouched."
    );
    assert_agent_snapshot!(assert_yaml_snapshot!(
        "parse_agent_file_subagents_enabled_tools",
        actual.tools
    ));
}

#[test]
fn test_parse_agent_file_renders_conditional_frontmatter_when_subagents_disabled() {
    let fixture = r#"---
id: "forge"
tools:
  - read
  - task
  - sage
  - mcp_*
---
Body keeps {{tool_names.read}} untouched.
"#;
    let config = ForgeConfig { subagents: false, ..Default::default() };

    let actual = apply_subagent_tool_config(parse_agent_file(fixture).unwrap(), &config).unwrap();

    assert_eq!(actual.id, AgentId::new("forge"));
    assert_agent_snapshot!(assert_snapshot!(
        "parse_agent_file_subagents_disabled_prompt",
        actual.system_prompt.unwrap().template
    ));
    assert_agent_snapshot!(assert_yaml_snapshot!(
        "parse_agent_file_subagents_disabled_tools",
        actual.tools
    ));
}

#[test]
fn test_parse_agent_file_preserves_runtime_user_prompt_variables() {
    let fixture = r#"---
id: "forge"
tools:
  - read
  - task
  - sage
  - mcp_*
user_prompt: |-
  <{{event.name}}>{{event.value}}</{{event.name}}>
  <system_date>{{current_date}}</system_date>
---
Body keeps {{tool_names.read}} untouched.
"#;

    let actual = parse_agent_file(fixture).unwrap();
    let actual_user_prompt = actual.user_prompt.clone().unwrap().template;

    assert_eq!(actual.id, AgentId::new("forge"));
    assert_agent_snapshot!(assert_snapshot!(
        "parse_agent_file_preserves_runtime_user_prompt_variables",
        actual_user_prompt
    ));
    assert_agent_snapshot!(assert_yaml_snapshot!(
        "parse_agent_file_preserves_runtime_user_prompt_variables_tools",
        apply_subagent_tool_config(
            actual,
            &ForgeConfig { subagents: true, ..Default::default() }
        )
        .unwrap()
        .tools
    ));
}
