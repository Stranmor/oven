import re

# 1. Fix eserde(compat) on AgentId and session_id in catalog.rs
with open("crates/forge_domain/src/tools/catalog.rs", "r") as f:
    cat = f.read()

cat = cat.replace("pub agent_id: crate::AgentId,", "#[eserde(compat)]\n    pub agent_id: crate::AgentId,")
cat = cat.replace("pub session_id: Option<crate::ConversationId>,", "#[eserde(compat)]\n    pub session_id: Option<crate::ConversationId>,")

# Fix tests/builders in catalog.rs
cat = cat.replace("file_path: path.to_string(),", "file_path: path.into(),")
cat = cat.replace("FSRemove { path: path.to_string() }", "FSRemove { path: path.into() }")
cat = cat.replace("FSUndo { path: path.to_string() }", "FSUndo { path: path.into() }")

# Fix display_path_for in catalog.rs
cat = cat.replace("display_path_for(&input.file_path)", "display_path_for(&input.file_path.to_string_lossy())")
cat = cat.replace("display_path_for(&input.path)", "display_path_for(&input.path.to_string_lossy())")

with open("crates/forge_domain/src/tools/catalog.rs", "w") as f:
    f.write(cat)


# 2. Fix compact/summary.rs paths
with open("crates/forge_domain/src/compact/summary.rs", "r") as f:
    sum = f.read()

sum = sum.replace("path: input.file_path }", "path: input.file_path.to_string_lossy().to_string() }")
sum = sum.replace("path: input.path }", "path: input.path.to_string_lossy().to_string() }")
sum = sum.replace("agent_id: input.agent_id }", "agent_id: input.agent_id.to_string() }")

with open("crates/forge_domain/src/compact/summary.rs", "w") as f:
    f.write(sum)

