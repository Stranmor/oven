with open("crates/forge_app/src/tool_registry.rs", "r") as f:
    content = f.read()

old_modality = """                    let required_modality = "image".to_string();
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

new_modality = """                    let required_modality = InputModality::Image;
                    let supported_modalities = model
                        .map(|m| m.input_modalities.clone())
                        .unwrap_or_default();"""

content = content.replace(old_modality, new_modality)

old_supported = """            let supported_tools = agent
                .tools
                .iter()
                .flatten()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join(", ");"""

new_supported = """            let supported_tools = agent.tools.clone().unwrap_or_default();"""

content = content.replace(old_supported, new_supported)

# One more thing: has_image_extension takes &str. In `catalog.rs`, I changed `file_path` to `PathBuf`. 
# So `input.file_path` is a PathBuf. We need to do `input.file_path.to_string_lossy()`.
content = content.replace(
    "if Self::has_image_extension(&input.file_path) {",
    "if Self::has_image_extension(&input.file_path.to_string_lossy()) {"
)

with open("crates/forge_app/src/tool_registry.rs", "w") as f:
    f.write(content)
