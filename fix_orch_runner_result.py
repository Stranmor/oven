import re

with open("crates/forge_app/src/orch_spec/orch_runner.rs", "r") as f:
    content = f.read()

content = content.replace(
    "let services = Arc::new(Runner::new(setup));",
    "let services = Arc::new(Runner::new(setup)?);"
)

with open("crates/forge_app/src/orch_spec/orch_runner.rs", "w") as f:
    f.write(content)
