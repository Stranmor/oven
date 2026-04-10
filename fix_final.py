import re
import os

files_to_fix = []
for root, dirs, files in os.walk("crates"):
    for file in files:
        if file.endswith(".rs"):
            files_to_fix.append(os.path.join(root, file))

for file in files_to_fix:
    with open(file, "r") as f:
        content = f.read()

    # Revert `std::path::PathBuf::from("...")` -> `"...".to_string()`
    content = re.sub(
        r'std::path::PathBuf::from\((["\'].*?["\'])\)',
        r'\1.to_string()',
        content
    )
    
    # Revert file_path: "..." -> file_path: "...".to_string() (Wait, in catalog.rs it MUST be PathBuf)
    
    # Revert paths in changed_files.rs
    content = content.replace("pub path: std::path::PathBuf,", "pub path: String,")
    
    # Revert fs_remove.rs
    content = content.replace("async fn remove(&self, path: std::path::PathBuf) -> anyhow::Result<FsRemoveOutput>", "async fn remove(&self, input_path: String) -> anyhow::Result<FsRemoveOutput>")
    content = content.replace("Path::new(&path)", "Path::new(&input_path)")
    
    # Revert fs_search.rs
    content = content.replace("path: path.to_string_lossy().to_string().into()", "path: path.to_string_lossy().to_string()")
    content = content.replace("path: path.clone().into()", "path: path.to_string_lossy().to_string()")
    
    with open(file, "w") as f:
        f.write(content)

