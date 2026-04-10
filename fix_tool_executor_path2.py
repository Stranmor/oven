with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    content = f.read()

content = content.replace(
    "unwrap_or_default(),",
    'map_err(|_| crate::Error::OperationNotPermitted("Invalid UTF-8 in path".into()))?,'
)
# I also need to fix `create_temp_file`.
# self.services.write(path.clone().into_os_string().into_string().map_err(|_| crate::Error::OperationNotPermitted("Invalid UTF-8 in path".into()))?, content.to_string(), true).await?;
with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(content)
