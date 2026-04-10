import os
services_files = [
    "crates/forge_app/src/services.rs",
    "crates/forge_services/src/tool_services/fs_write.rs",
    "crates/forge_services/src/tool_services/fs_read.rs",
    "crates/forge_services/src/tool_services/fs_search.rs",
    "crates/forge_services/src/tool_services/fs_patch.rs",
    "crates/forge_services/src/tool_services/fs_remove.rs",
    "crates/forge_services/src/tool_services/fs_undo.rs",
]

for file in services_files:
    if not os.path.exists(file): continue
    with open(file, "r") as f:
        c = f.read()
    
    c = c.replace("path: String,", "path: std::path::PathBuf,")
    c = c.replace("async fn undo(&self, path: String)", "async fn undo(&self, path: std::path::PathBuf)")
    c = c.replace("async fn remove(&self, path: String)", "async fn remove(&self, path: std::path::PathBuf)")
    c = c.replace("async fn patch(\n        &self,\n        path: String", "async fn patch(\n        &self,\n        path: std::path::PathBuf")
    c = c.replace("async fn multi_patch(\n        &self,\n        path: String", "async fn multi_patch(\n        &self,\n        path: std::path::PathBuf")
    c = c.replace("async fn read(\n        &self,\n        path: String", "async fn read(\n        &self,\n        path: std::path::PathBuf")
    c = c.replace("async fn read_utf8(&self, path: String)", "async fn read_utf8(&self, path: std::path::PathBuf)")
    c = c.replace("async fn range_read_utf8(&self, path: String", "async fn range_read_utf8(&self, path: std::path::PathBuf")

    c = c.replace("let path = std::path::PathBuf::from(path);", "")
    c = c.replace("let path = std::path::PathBuf::from(&path);", "")
    c = c.replace("let file_path = std::path::PathBuf::from(path);", "let file_path = path;")

    with open(file, "w") as f:
        f.write(c)

