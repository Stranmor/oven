with open("crates/forge_app/src/system_prompt.rs", "r") as f:
    lines = f.read().split('\n')

# The appended stuff starts at index 312 (line 313)
appended_start = 312
# We want to put lines 312 to end INSIDE the mod tests, which ends at 310
appended_content = lines[appended_start:]
main_content = lines[:311] # everything before the closing brace of mod tests

new_lines = main_content + appended_content + ["}"]

with open("crates/forge_app/src/system_prompt.rs", "w") as f:
    f.write('\n'.join(new_lines))
