import re
with open("crates/forge_app/src/orch_spec/orch_runner.rs", "r") as f:
    content = f.read()

# Remove unwrap on register_template_string
content = content.replace("pub fn new() -> Self", "pub fn new() -> anyhow::Result<Self>")
content = content.replace(".unwrap();", "?;")
# Add Ok(Self { ... }) at end of new()
content = content.replace("        Self {", "        Ok(Self {")
content = content.replace("        }", "        })")

# In list_skills: unimplemented!() -> anyhow::bail!("unimplemented")
content = content.replace("unimplemented!()", "anyhow::bail!(\"unimplemented\")")

# In ShellService::execute:
# if let Some(output) = outputs.pop_front() { Ok(output) } else { Ok(fallback) } -> anyhow::bail!("exhausted queue")
content = re.sub(
    r"if let Some\(output\) = outputs\.pop_front\(\) {\s*Ok\(output\)\s*} else \{\s*Ok\(ShellOutput.*?\)\s*\}",
    r"if let Some(output) = outputs.pop_front() { Ok(output) } else { anyhow::bail!(\"exhausted shell output queue\") }",
    content,
    flags=re.DOTALL
)

# In FileDiscoveryService::collect_files:
content = re.sub(
    r"if let Some\(output\) = outputs\.pop_front\(\) \{.*?} else \{\s*Ok\(vec\!\[\]\)\s*\}",
    r"if let Some(output) = outputs.pop_front() { \n            let files = output.output.stdout.lines().map(str::trim).filter(|line| !line.is_empty()).map(|line| forge_domain::File { path: line.to_string(), is_dir: false }).collect();\n            Ok(files) } else { anyhow::bail!(\"exhausted discovery queue\") }",
    content,
    flags=re.DOTALL
)

with open("crates/forge_app/src/orch_spec/orch_runner.rs", "w") as f:
    f.write(content)
