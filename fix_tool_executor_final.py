with open("crates/forge_app/src/tool_executor.rs", "r") as f:
    content = f.read()

content = content.replace('i as u64', 'u64::try_from(i).unwrap_or(0)')

# Err(anyhow!(...)) -> Err(crate::Error::PreconditionFailed(...).into())
# Wait, let me replace it using regex to match any formatting
import re
content = re.sub(
    r'Err\(anyhow!\(\s*"You must read the file with the read tool before attempting to \{action\}\.",\s*action = action\s*\)\)',
    r'Err(crate::Error::PreconditionFailed(format!("You must read the file with the read tool before attempting to {action}.", action=action)).into())',
    content
)

# unreachable! -> Err
content = content.replace('unreachable!("Tasks should be intercepted before execution")', 'return Err(crate::Error::OperationNotPermitted("Tasks should be intercepted before execution".into()).into())')

# stdout_truncated / stderr_truncated limit fix
old_trunc = """                let stdout_truncated =
                    stdout_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;
                let stderr_truncated =
                    stderr_lines > config.max_stdout_prefix_lines + config.max_stdout_suffix_lines;"""

new_trunc = """                let max_total_lines = config.max_stdout_prefix_lines.saturating_add(config.max_stdout_suffix_lines);
                let stdout_truncated = stdout_lines > max_total_lines;
                let stderr_truncated = stderr_lines > max_total_lines;"""

content = content.replace(old_trunc, new_trunc)

with open("crates/forge_app/src/tool_executor.rs", "w") as f:
    f.write(content)
