import re

with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    te = f.read()

# Replace the signature and function body
old_func = """    fn require_prior_read(
        &self,
        context: &ToolCallContext,
        raw_path: &str,
        action: &str,
    ) -> anyhow::Result<()> {
        let target_path = self.normalize_path(raw_path.to_path_buf());
        let has_read = context.with_metrics(|metrics| {
            metrics.files_accessed.contains(&target_path.to_string_lossy().to_string())
                || metrics.files_accessed.contains(raw_path)
        })?;

        if has_read {
            Ok(())
        } else {
            Err(crate::Error::PreconditionFailed(crate::error::PreconditionReason::UnreadTarget(format!("You must read the file with the read tool before attempting to {action}.", action=action))).into())
        }
    }"""

new_func = """    fn require_prior_read(
        &self,
        context: &ToolCallContext,
        raw_path: &std::path::Path,
        action: &str,
    ) -> anyhow::Result<()> {
        let target_path = self.normalize_path(raw_path.to_path_buf());
        let target_path_str = target_path.to_string_lossy().to_string();
        let raw_path_str = raw_path.to_string_lossy().to_string();
        
        let has_read = context.with_metrics(|metrics| {
            metrics.files_accessed.contains(&target_path_str)
        })?;

        if has_read {
            Ok(())
        } else {
            Err(crate::Error::PreconditionFailed(crate::error::PreconditionReason::UnreadTarget(format!("You must read the file with the read tool before attempting to {action}.", action=action))).into())
        }
    }"""

te = te.replace(old_func, new_func)
te = te.replace("self.require_prior_read(context, path, path, \"edit it\")?", "self.require_prior_read(context, &path, \"edit it\")?")
te = te.replace("self.require_prior_read(context, &input.file_path, &input.file_path, \"overwrite it\")?", "self.require_prior_read(context, &input.file_path, \"overwrite it\")?")

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(te)

with open("crates/forge_app/src/operation.rs", "r") as f:
    op = f.read()

# Fix FSRemove { path: file_path } where file_path doesn't exist
op = op.replace("FSRemove { path: file_path }", "FSRemove { path: input.path.clone() }")

with open("crates/forge_app/src/operation.rs", "w") as f:
    f.write(op)
