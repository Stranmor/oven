import re
with open("crates/forge_app/src/services.rs", "r") as f:
    content = f.read()

# I already replaced "path: String," with "path: std::path::PathBuf," in services.rs but it might be missing imports or something?
# No, std::path::PathBuf is fully qualified.
# Wait, "path: String" was used in:
# fn write(..., path: String, ...) -> ...
# I will change it globally for tool services.

services_files = [
    "crates/forge_services/src/tool_services/fs_write.rs",
    "crates/forge_services/src/tool_services/fs_read.rs",
    "crates/forge_services/src/tool_services/fs_search.rs",
    "crates/forge_services/src/tool_services/fs_patch.rs",
    "crates/forge_services/src/tool_services/fs_remove.rs",
    "crates/forge_services/src/tool_services/fs_undo.rs",
]

for file in services_files:
    try:
        with open(file, "r") as f:
            c = f.read()
        
        # Replace async fn write(&self, path: String
        c = c.replace("path: String,", "path: std::path::PathBuf,")
        c = c.replace("async fn undo(&self, path: String)", "async fn undo(&self, path: std::path::PathBuf)")
        c = c.replace("async fn remove(&self, path: String)", "async fn remove(&self, path: std::path::PathBuf)")
        c = c.replace("async fn patch(\n        &self,\n        path: String", "async fn patch(\n        &self,\n        path: std::path::PathBuf")
        c = c.replace("async fn multi_patch(\n        &self,\n        path: String", "async fn multi_patch(\n        &self,\n        path: std::path::PathBuf")
        c = c.replace("async fn read(\n        &self,\n        path: String", "async fn read(\n        &self,\n        path: std::path::PathBuf")
        c = c.replace("async fn read_utf8(&self, path: String)", "async fn read_utf8(&self, path: std::path::PathBuf)")
        c = c.replace("async fn range_read_utf8(&self, path: String", "async fn range_read_utf8(&self, path: std::path::PathBuf")
        
        # For fs_write: let path = PathBuf::from(path); -> let path = path;
        c = c.replace("let path = PathBuf::from(path);", "")
        # For fs_read: let path = PathBuf::from(&path);
        c = c.replace("let path = PathBuf::from(&path);", "")
        c = c.replace("let file_path = PathBuf::from(path);", "let file_path = path;")
        c = c.replace("let path = PathBuf::from(path);", "")
        
        with open(file, "w") as f:
            f.write(c)
    except FileNotFoundError:
        pass
