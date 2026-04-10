import re
import os

with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    te = f.read()

# 1. normalize_path uses String instead of PathBuf
te = te.replace("fn normalize_path(&self, path: String) -> String {", "fn normalize_path(&self, path: std::path::PathBuf) -> std::path::PathBuf {")
te = te.replace("        let path_buf = PathBuf::from(&path);", "")
te = te.replace("        if path_buf.is_absolute() {", "        if path.is_absolute() {")
te = te.replace("            path.clone()", "            path.clone()")
te = te.replace("        } else {", "        } else {")
te = te.replace(
    "            env.cwd.join(path).to_string_lossy().to_string()",
    "            env.cwd.join(path)"
)

# Fix require_prior_read
te = te.replace(
    "fn require_prior_read(\n        &self,\n        context: &Conversation,\n        target_path: &str,\n        action: &str,\n    ) -> Result<(), Error> {",
    "fn require_prior_read(\n        &self,\n        context: &Conversation,\n        target_path: &std::path::Path,\n        action: &str,\n    ) -> Result<(), Error> {"
)

# In require_prior_read:
te = te.replace(
    "        let raw_path = target_path;\n        let target_path = self.normalize_path(target_path.to_string());",
    "        let target_path = self.normalize_path(target_path.to_path_buf());"
)

# In require_prior_read:
te = te.replace(
    "            if metrics.files_accessed.contains_key(&target_path)\n                || metrics.files_accessed.contains_key(raw_path)\n            {",
    "            if metrics.files_accessed.contains_key(&target_path.to_string_lossy().to_string()) {"
)

# Now fix call sites of require_prior_read
te = te.replace("self.require_prior_read(context, &input.file_path.to_string_lossy().into_owned(),", "self.require_prior_read(context, &input.file_path,")
te = te.replace("self.require_prior_read(context, &input.path.to_string_lossy().into_owned(),", "self.require_prior_read(context, &input.path,")
te = te.replace("self.require_prior_read(context, path, \"edit it\")?", "self.require_prior_read(context, &path, \"edit it\")?")

# Fix normalized_path assignment
te = re.sub(
    r"let normalized_path = self\.normalize_path\(input\.(file_path|path)\.clone\(\)\);",
    r"let normalized_path = self.normalize_path(input.\1.clone());",
    te
)

# And fix `PathBuf::from(&normalized_path)` -> `normalized_path` if already PathBuf
te = te.replace("PathBuf::from(&normalized_path)", "normalized_path.clone()")
te = te.replace("PathBuf::from(normalized_path)", "normalized_path.clone()")
te = te.replace("PathBuf::from(&path)", "path.clone()")

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(te)

with open("crates/forge_app/src/operation.rs", "r") as f:
    op = f.read()

# Fix .insert(input.file_path.clone(), ...) -> .insert(input.file_path.to_string_lossy().to_string(), ...)
op = re.sub(r"\.insert\(\n\s+input\.(file_path|path)\.clone\(\),", r".insert(\n                    input.\1.to_string_lossy().to_string(),", op)

# Fix .attr("path", input.path) -> .attr("path", input.path.to_string_lossy())
op = re.sub(r"\.attr\(\"path\",\s*input\.(file_path|path)\)", r'.attr("path", input.\1.to_string_lossy())', op)

# Fix tracing::info!(path = %input.file_path
op = re.sub(r"path = %input\.(file_path|path),", r"path = %input.\1.display(),", op)

# Fix FSRemove path issue
op = op.replace("FSRemove { path }", "FSRemove { path: file_path }")

with open("crates/forge_app/src/operation.rs", "w") as f:
    f.write(op)
