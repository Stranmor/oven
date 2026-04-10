import re
with open("crates/forge_app/src/orch_spec/orch_runner.rs", "r") as f:
    content = f.read()

content = content.replace(
    r"""if let Some(output) = outputs.pop_front() { Ok(output) } else { anyhow::bail!(\"exhausted shell output queue\") }""",
    r"""if let Some(output) = outputs.pop_front() { Ok(output) } else { anyhow::bail!("exhausted shell output queue") }"""
)

with open("crates/forge_app/src/orch_spec/orch_runner.rs", "w") as f:
    f.write(content)
