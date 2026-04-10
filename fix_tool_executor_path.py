import re
with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    content = f.read()

# Replace any `.into()` used for path arguments to safely stringify them
content = content.replace(
    "normalized_path.into(),",
    "normalized_path.into_os_string().into_string().unwrap_or_default(),"
)
content = content.replace(
    "path.clone(),",
    "path.clone().into_os_string().into_string().unwrap_or_default(),"
)

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(content)
