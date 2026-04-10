import os
import re

files_to_fix = []
for root, dirs, files in os.walk("crates"):
    for file in files:
        if file.endswith(".rs"):
            files_to_fix.append(os.path.join(root, file))

for file in files_to_fix:
    with open(file, "r") as f:
        content = f.read()

    # Fix file_path: "".to_string()
    content = re.sub(
        r'file_path:\s*(".+?")\.to_string\(\)',
        r'file_path: std::path::PathBuf::from(\1)',
        content
    )
    
    # Fix path: "".to_string()
    content = re.sub(
        r'path:\s*(".+?")\.to_string\(\)',
        r'path: std::path::PathBuf::from(\1)',
        content
    )
    
    # Fix file_path: path.to_string()
    content = re.sub(
        r'file_path:\s*([a-zA-Z0-9_]+)\.to_string\(\)',
        r'file_path: std::path::PathBuf::from(\1)',
        content
    )

    # Fix path: path.to_string()
    content = re.sub(
        r'path:\s*([a-zA-Z0-9_]+)\.to_string\(\)',
        r'path: std::path::PathBuf::from(\1)',
        content
    )
    
    # Fix assert_eq!
    content = re.sub(
        r'assert_eq!\((fs_[a-z]+)\.file_path,\s*(".+?")\);',
        r'assert_eq!(\1.file_path, std::path::PathBuf::from(\2));',
        content
    )
    
    content = re.sub(
        r'assert_eq!\((fs_[a-z]+)\.path,\s*(".+?")\);',
        r'assert_eq!(\1.path, std::path::PathBuf::from(\2));',
        content
    )
    
    content = content.replace("input_path: String", "input_path: std::path::PathBuf")
    content = content.replace("async fn read_utf8(&self, path: String)", "async fn read_utf8(&self, path: std::path::PathBuf)")
    content = content.replace("async fn range_read_utf8(&self, path: String", "async fn range_read_utf8(&self, path: std::path::PathBuf")
    content = content.replace("async fn remove(&self, input_path: std::path::PathBuf)", "async fn remove(&self, path: std::path::PathBuf)")
    
    with open(file, "w") as f:
        f.write(content)

