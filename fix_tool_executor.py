with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    content = f.read()

# 1. normalize_path takes PathBuf and returns PathBuf
content = content.replace("fn normalize_path(&self, path: String) -> String {", "fn normalize_path(&self, path: std::path::PathBuf) -> std::path::PathBuf {")
content = content.replace("let path_buf = PathBuf::from(&path);", "")
content = content.replace("if path_buf.is_absolute() {", "if path.is_absolute() {")
content = content.replace(
    "env.cwd.join(path).to_string_lossy().to_string()",
    "env.cwd.join(path)"
)
# We need to change `let normalized_path = self.normalize_path(input.file_path.clone());`
# The return is PathBuf, so `services.read(normalized_path)` etc. will pass PathBuf.

# 2. create_temp_file uses PathBuf for services.write
content = content.replace("path.to_string_lossy().to_string()", "path.clone()")

# 3. ToolCatalog::Shell
# cwd: cwd.map(|p| p.display().to_string()).unwrap_or_else(|| self.services.get_environment().cwd.display().to_string());
# let normalized_cwd = self.normalize_path(cwd);
content = content.replace(
"""                let cwd = input
                    .cwd
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| self.services.get_environment().cwd.display().to_string());
                let normalized_cwd = self.normalize_path(cwd);
                let output = self
                    .services
                    .execute(
                        input.command.clone(),
                        PathBuf::from(normalized_cwd),""",
"""                let cwd = input.cwd.unwrap_or_else(|| self.services.get_environment().cwd.clone());
                let normalized_cwd = self.normalize_path(cwd);
                let output = self
                    .services
                    .execute(
                        input.command.clone(),
                        normalized_cwd,"""
)

# 4. unreachable!("...") in ToolCatalog::Task(_)
# "It must return a structured operational error instead of deliberately terminating the thread."
content = content.replace('unreachable!("Tasks should be intercepted before execution")', 'return Err(anyhow::anyhow!("Tasks should be intercepted before execution"));')
# Wait, the critic said:
# "Pervasive use of anyhow::Result and anyhow!(...). Errors must be classified at the point of origin (Transient, Recoverable, Permanent) into strict enums. anyhow acts as a catch-all that erases the error type, violating the mandate that caller layers must handle specific error instances via explicit match arms."
# So I should use `crate::Error` or `forge_domain::Error`. Let's see what Error exists.
