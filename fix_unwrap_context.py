import re

with open("crates/forge_services/src/context_engine.rs", "r") as f:
    content = f.read()

# Replace async fn test...() { with async fn test...() -> anyhow::Result<()> {
content = re.sub(
    r"async fn test_([a-zA-Z0-9_]+)\(\) {",
    r"async fn test_\1() -> anyhow::Result<()> {",
    content
)

# Fix unwraps inside the test
content = content.replace("temp_dir.path().canonicalize().unwrap()", "temp_dir.path().canonicalize()?")
content = content.replace(".unwrap()", "?")
content = content.replace("assert_eq!(*infra.search_limit.lock()?, Some(15));", "assert_eq!(*infra.search_limit.lock().unwrap(), Some(15));")
content = content.replace("assert_eq!(*infra.search_limit.lock()?, Some(10));", "assert_eq!(*infra.search_limit.lock().unwrap(), Some(10));")
content = content.replace("panic!(\"Expected SyncProgress::Starting\");", "anyhow::bail!(\"Expected SyncProgress::Starting\");")

# Ensure every test ends with Ok(())
# We can do this by finding the closing brace of each test function and inserting Ok(()) before it.
def append_ok(content):
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("    async fn test_") and "-> anyhow::Result<()> {" in line:
            # find end of function
            brackets = 1
            for j in range(i+1, len(lines)):
                if "{" in lines[j]: brackets += lines[j].count("{")
                if "}" in lines[j]: brackets -= lines[j].count("}")
                if brackets == 0:
                    # found end
                    lines.insert(j, "        Ok(())")
                    break
    return '\n'.join(lines)

content = append_ok(content)

with open("crates/forge_services/src/context_engine.rs", "w") as f:
    f.write(content)
