import re

with open("crates/forge_app/src/error.rs", "r") as f:
    content = f.read()

content = content.replace("#[derive(Debug, thiserror::Error)]\n#[derive(thiserror::Error, Debug)]", "#[derive(Debug, thiserror::Error)]")
content = content.replace("#[derive(thiserror::Error, Debug)]\n#[derive(Debug, thiserror::Error)]", "#[derive(Debug, thiserror::Error)]")

content = content.replace(
    '#[error("Tool \'{tool_name}\' requires {required_modality} modality, but model only supports: {supported_modalities}")]',
    '#[error("Tool \'{tool_name}\' requires {required_modality:?} modality, but model only supports: {supported_modalities:?}")]'
)

with open("crates/forge_app/src/error.rs", "w") as f:
    f.write(content)
