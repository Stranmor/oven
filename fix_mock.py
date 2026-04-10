with open("crates/forge_app/src/system_prompt.rs", "r") as f:
    content = f.read()

content = content.replace(
    'crate::ShellOutput { output: forge_domain::CommandOutput { command: "".into(), exit_code: Some(0), stdout: "".into(), stderr: "".into() }, command: "".into() }',
    'crate::ShellOutput { output: forge_domain::CommandOutput { command: "".into(), exit_code: Some(0), stdout: "".into(), stderr: "".into() }, shell: "sh".into(), description: None }'
)

with open("crates/forge_app/src/system_prompt.rs", "w") as f:
    f.write(content)
