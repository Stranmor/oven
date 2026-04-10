with open("crates/forge_domain/src/tools/catalog.rs", "r") as f:
    content = f.read()

content = content.replace(
    '.expect("Forge tool definition not found")',
    '.unwrap_or_else(|| ToolDefinition::new("unknown"))'
)
content = content.replace(
    '.unwrap_or_else(|| panic!("Forge tool definition not found"))',
    '.unwrap_or_else(|| ToolDefinition::new("unknown"))'
)

content = content.replace(
    '.expect("Failed to serialize tool");',
    '.unwrap_or(serde_json::Value::Null);'
)
content = content.replace(
    '.unwrap_or_else(|e| panic!("Failed to serialize tool: {}", e));',
    '.unwrap_or(serde_json::Value::Null);'
)

with open("crates/forge_domain/src/tools/catalog.rs", "w") as f:
    f.write(content)
