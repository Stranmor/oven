with open("crates/forge_domain/src/compact/summary.rs", "r") as f:
    sum = f.read()

sum = sum.replace("path: input.file_path }", "path: input.file_path.to_string_lossy().to_string() }")
sum = sum.replace("path: input.path }", "path: input.path.to_string_lossy().to_string() }")
sum = sum.replace("agent_id: input.agent_id }", "agent_id: input.agent_id.to_string() }")

with open("crates/forge_domain/src/compact/summary.rs", "w") as f:
    f.write(sum)
