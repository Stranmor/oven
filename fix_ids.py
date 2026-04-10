with open("crates/forge_domain/src/conversation.rs", "r") as f:
    content = f.read()

content = content.replace(
    "#[derive(Debug, Default, Display, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]",
    "#[derive(Debug, Default, Display, Serialize, Deserialize, Clone, PartialEq, Eq, Hash, schemars::JsonSchema)]"
)

with open("crates/forge_domain/src/conversation.rs", "w") as f:
    f.write(content)

with open("crates/forge_domain/src/agent.rs", "r") as f:
    ag = f.read()

ag = ag.replace(
    "#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]\n#[serde(transparent)]\npub struct AgentId",
    "#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, schemars::JsonSchema)]\n#[serde(transparent)]\npub struct AgentId"
)

with open("crates/forge_domain/src/agent.rs", "w") as f:
    f.write(ag)
