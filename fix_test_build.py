with open("crates/forge_app/src/system_prompt.rs", "r") as f:
    content = f.read()

old_test = """agent.system_prompt = Some(forge_domain::Template::new(r#"System prompt <workspace_extensions extensions="{{extensions}}" total="{{total_files}}" limit="{{limit}}" />"#));"""

new_test = """agent.system_prompt = Some(forge_domain::Template::new(r#"System prompt <workspace_extensions extensions="{{extensions.total_extensions}}" total="{{extensions.git_tracked_files}}" />"#));"""

content = content.replace(old_test, new_test)

with open("crates/forge_app/src/system_prompt.rs", "w") as f:
    f.write(content)
