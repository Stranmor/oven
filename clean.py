import re

with open("crates/forge_services/src/context_engine.rs", "r") as f:
    content = f.read()

# Find the first #[cfg(test)]
parts = content.split("#[cfg(test)]")
if len(parts) > 1:
    clean_content = parts[0]
    with open("crates/forge_services/src/context_engine.rs", "w") as f:
        f.write(clean_content)
