import re

with open("crates/forge_app/src/error.rs", "r") as f:
    content = f.read()

content = content.replace("forge_domain::Modality", "forge_domain::InputModality")
content = content.replace("pub enum PreconditionReason", "#[derive(Debug, thiserror::Error)]\npub enum PreconditionReason")
content = content.replace("pub enum OperationPermitReason", "#[derive(Debug, thiserror::Error)]\npub enum OperationPermitReason")

with open("crates/forge_app/src/error.rs", "w") as f:
    f.write(content)


with open("crates/forge_app/src/tool_registry.rs", "r") as f:
    tr_content = f.read()

old_modality = """                let required_modality = "image".to_string();
                let supported_modalities = model
                    .map(|m| {
                        m.input_modalities
                            .iter()
                            .map(|im| match im {
                                InputModality::Text => "text".to_string(),
                                InputModality::Image => "image".to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_else(|| "unknown".to_string());"""

new_modality = """                let required_modality = InputModality::Image;
                let supported_modalities = model
                    .map(|m| m.input_modalities.clone())
                    .unwrap_or_default();"""

tr_content = tr_content.replace(old_modality, new_modality)

with open("crates/forge_app/src/tool_registry.rs", "w") as f:
    f.write(tr_content)
