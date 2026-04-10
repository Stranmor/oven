import re

with open("crates/forge_app/src/error.rs", "r") as f:
    content = f.read()

content = re.sub(
    r'#\[error\(\s*"Tool \'\{name\}\' is not available. Please try again with one of these tools: \[\{supported_tools\}\]"\s*\)\]',
    r'#[error("Tool \'{name}\' is not available. Please try again with one of these tools: {:?}", supported_tools)]',
    content
)

with open("crates/forge_app/src/error.rs", "w") as f:
    f.write(content)

with open("crates/forge_app/src/tool_registry.rs", "r") as f:
    tr_content = f.read()

tr_content = tr_content.replace(
    'let supported_tools = state.allowed_tools.iter().map(|n| n.as_str()).collect::<Vec<_>>().join(", ");',
    'let supported_tools = state.allowed_tools.clone();'
)

with open("crates/forge_app/src/tool_registry.rs", "w") as f:
    f.write(tr_content)
