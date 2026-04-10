import re

with open("crates/forge_app/src/error.rs", "r") as f:
    content = f.read()

content = content.replace("supported_tools: String", "supported_tools: Vec<forge_domain::ToolName>")

content = re.sub(
    r'#\[error\(\s*"Tool \'\{name\}\' is not allowed in this mode. Supported tools: \{supported_tools\}"\s*\)\]',
    r'#[error("Tool \'{name}\' is not allowed in this mode. Supported tools: {:?}", supported_tools)]',
    content
)

content = re.sub(
    r'#\[error\(\s*"Tool \'\{tool_name\}\' requires \{required_modality\} modality, but model only supports: \{supported_modalities\}"\s*\)\]',
    r'#[error("Tool \'{tool_name}\' requires {required_modality:?} modality, but model only supports: {supported_modalities:?}")]',
    content
)

with open("crates/forge_app/src/error.rs", "w") as f:
    f.write(content)
