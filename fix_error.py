import re

with open("crates/forge_app/src/error.rs", "r") as f:
    content = f.read()

content = content.replace("#[derive(Debug, thiserror::Error)]\npub enum PreconditionReason", "pub enum PreconditionReason")
content = content.replace("#[derive(Debug, thiserror::Error)]\npub enum OperationPermitReason", "pub enum OperationPermitReason")

with open("crates/forge_app/src/error.rs", "w") as f:
    f.write(content)
