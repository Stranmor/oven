import re

with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    te = f.read()

te = te.replace("PathBuf::from(&env.cwd).join(path_buf).display().to_string()", "std::path::PathBuf::from(&env.cwd).join(path)")
te = te.replace("let target_path = self.normalize_path(raw_path.to_string());", "let target_path = self.normalize_path(std::path::PathBuf::from(raw_path));")

# files_accessed expects String (it seems, since contains_key failed)
# wait, metrics.files_accessed.contains(&target_path) failed because target_path is PathBuf and contains expects String.
# Let's fix tool_executor.rs:54
te = te.replace(
    "metrics.files_accessed.contains(&target_path)",
    "metrics.files_accessed.contains(&target_path.to_string_lossy().to_string())"
)

# fix `.write(path.to_string_lossy().to_string().into(),`
te = re.sub(r"\.write\(\s*path\.to_string_lossy\(\)\.to_string\(\)(?:\.into\(\))?,", ".write(path.clone(),", te)
# fix params.path = Some(self.normalize_path(path.clone())); in SemanticSearch
te = re.sub(r"params\.path\s*=\s*Some\(self\.normalize_path\(path\.clone\(\)(?:\.into\(\))?\)\);", "params.path = Some(self.normalize_path(path.clone().into()).to_string_lossy().to_string());", te)

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(te)

with open("crates/forge_app/src/data_gen.rs", "r") as f:
    dg = f.read()
dg = dg.replace("resolved_path.display().to_string()", "resolved_path")
with open("crates/forge_app/src/data_gen.rs", "w") as f:
    f.write(dg)

with open("crates/forge_app/src/file_tracking.rs", "r") as f:
    ft = f.read()
ft = ft.replace("file_path.to_string_lossy().to_string()", "file_path.clone()")
with open("crates/forge_app/src/file_tracking.rs", "w") as f:
    f.write(ft)

with open("crates/forge_app/src/operation.rs", "r") as f:
    op = f.read()
op = op.replace("create_validation_warning(&input.file_path,", "create_validation_warning(&input.file_path.to_string_lossy(),")
op = op.replace(".attr(\"path\", &input.file_path)", ".attr(\"path\", input.file_path.to_string_lossy())")
with open("crates/forge_app/src/operation.rs", "w") as f:
    f.write(op)

with open("crates/forge_app/src/utils.rs", "r") as f:
    ut = f.read()
ut = ut.replace("matched.path, err", "matched.path.display(), err")
with open("crates/forge_app/src/utils.rs", "w") as f:
    f.write(ut)

