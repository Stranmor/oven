import re
import os

files_to_fix = [
    "crates/forge_app/src/orch_spec/orch_spec.rs",
    "crates/forge_app/src/orch_spec/orch_system_spec.rs"
]

for file in files_to_fix:
    with open(file, "r") as f:
        content = f.read()

    # Replace `Runner::new(&setup)` with `Runner::new(&setup).unwrap()`? No, unwrap is banned!
    # Since these are test functions, we can just use `?` because they return Result<()>.
    # Wait, do they return `Result<()>`? Yes, they have `#[tokio::test] async fn test...() -> anyhow::Result<()> {`
    
    # In some places it might be `Runner::new(setup)`
    content = content.replace("Runner::new(&setup)", "Runner::new(&setup)?")
    content = content.replace("Runner::new(setup)", "Runner::new(setup)?")
    content = content.replace("Runner::new(&mut setup)", "Runner::new(&mut setup)?")
    
    with open(file, "w") as f:
        f.write(content)

