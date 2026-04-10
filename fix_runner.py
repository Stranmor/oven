import re
with open("crates/forge_app/src/orch_spec/orch_runner.rs", "r") as f:
    content = f.read()

# Remove unwrap on register_template_string
content = content.replace("fn new(setup: &TestContext) -> Self {", "fn new(setup: &TestContext) -> anyhow::Result<Self> {")
content = content.replace(".unwrap();", "?;")

# Replace return `Self { ... }` with `Ok(Self { ... })` at the end of new()
# We can do this with regex matching the specific block
match = re.search(r"        Self \{\n.*?\n        \}", content, re.DOTALL)
if match:
    content = content.replace(match.group(0), "        Ok(" + match.group(0).replace("        Self {", "Self {") + ")")

# In list_skills: unimplemented!() -> anyhow::bail!("unimplemented")
content = content.replace("unimplemented!()", "anyhow::bail!(\"unimplemented\")")

# In ShellService::execute:
content = re.sub(
    r"if let Some\(output\) = outputs\.pop_front\(\) {\s*Ok\(output\)\s*} else \{\s*Ok\(ShellOutput.*?\)\s*\}",
    r"if let Some(output) = outputs.pop_front() { Ok(output) } else { anyhow::bail!(\"exhausted shell output queue\") }",
    content,
    flags=re.DOTALL
)

# In FileDiscoveryService::collect_files:
content = re.sub(
    r"if let Some\(output\) = outputs\.pop_front\(\) \{.*?else \{\s*Ok\(vec\!\[\]\)\s*\}",
    r"""if let Some(output) = outputs.pop_front() {
            let files = output.output.stdout
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| forge_domain::File {
                    path: line.to_string(),
                    is_dir: false,
                })
                .collect();
            Ok(files)
        } else {
            anyhow::bail!("exhausted shell output queue")
        }""",
    content,
    flags=re.DOTALL
)

with open("crates/forge_app/src/orch_spec/orch_runner.rs", "w") as f:
    f.write(content)
