with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    te = f.read()

# fix require_prior_read to take std::path::PathBuf
te = te.replace(
"""    fn require_prior_read(
        &self,
        context: &Conversation,
        target_path: &std::path::Path,
        action: &str,
    ) -> Result<(), Error> {""",
"""    fn require_prior_read(
        &self,
        context: &Conversation,
        target_path: &std::path::Path,
        raw_path: &std::path::Path,
        action: &str,
    ) -> Result<(), Error> {""")

te = te.replace("let target_path = self.normalize_path(std::path::PathBuf::from(raw_path));", "let target_path = self.normalize_path(raw_path.to_path_buf());")

te = te.replace("self.require_prior_read(context, &path, \"edit it\")?", "self.require_prior_read(context, path, path, \"edit it\")?")
te = te.replace("self.require_prior_read(context, &input.file_path, \"overwrite it\")?", "self.require_prior_read(context, &input.file_path, &input.file_path, \"overwrite it\")?")

te = te.replace("let normalized_cwd = self.normalize_path(cwd);", "let normalized_cwd = self.normalize_path(cwd.into());")

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(te)
