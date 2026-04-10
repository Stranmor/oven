with open("crates/forge_services/src/context_engine.rs", "r") as f:
    content = f.read()

init_result_def = """pub enum InitResult {
    Created(forge_domain::WorkspaceId),
    Existing(forge_domain::WorkspaceId),
}

"""

if "pub enum InitResult" not in content:
    content = content.replace("pub struct ForgeWorkspaceService<F, D> {", init_result_def + "pub struct ForgeWorkspaceService<F, D> {")

with open("crates/forge_services/src/context_engine.rs", "w") as f:
    f.write(content)
