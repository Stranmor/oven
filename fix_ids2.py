with open("crates/forge_domain/src/conversation.rs", "r") as f:
    content = f.read()

content = content.replace(
    "pub struct ConversationId(Uuid);",
    "#[schemars(with = \"String\")]\npub struct ConversationId(Uuid);"
)

with open("crates/forge_domain/src/conversation.rs", "w") as f:
    f.write(content)
