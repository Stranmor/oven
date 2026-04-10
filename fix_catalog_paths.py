import re
with open("crates/forge_domain/src/tools/catalog.rs", "r") as f:
    content = f.read()

# FSRead, FSWrite, FSRemove, FSPatch
content = content.replace("pub file_path: String,", "pub file_path: std::path::PathBuf,")
content = content.replace("pub start_line: Option<i32>,", "pub start_line: Option<u32>,")
content = content.replace("pub end_line: Option<i32>,", "pub end_line: Option<u32>,")
content = content.replace("pub path: String,", "pub path: std::path::PathBuf,")

# TaskInput
content = content.replace("pub agent_id: String,", "pub agent_id: crate::AgentId,")
content = content.replace("pub session_id: Option<String>,", "pub session_id: Option<crate::ConversationId>,")

with open("crates/forge_domain/src/tools/catalog.rs", "w") as f:
    f.write(content)

with open("crates/forge_app/src/tool_registry.rs", "r") as f:
    tr = f.read()

tr = tr.replace("has_image_extension(&input.file_path.to_string_lossy())", "has_image_extension(&input.file_path.to_string_lossy())")

with open("crates/forge_app/src/tool_registry.rs", "w") as f:
    f.write(tr)
