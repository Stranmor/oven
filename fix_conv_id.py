with open("crates/forge_domain/src/conversation.rs", "r") as f:
    content = f.read()

content = content.replace(
    "#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]\npub struct ConversationId(Uuid);",
    "#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, schemars::JsonSchema)]\npub struct ConversationId(Uuid);"
)

with open("crates/forge_domain/src/conversation.rs", "w") as f:
    f.write(content)
