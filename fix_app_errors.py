import re

with open("crates/forge_app/src/tool_registry.rs", "r") as f:
    tr = f.read()

tr = tr.replace("has_image_extension(&input.file_path)", "has_image_extension(&input.file_path.to_string_lossy())")

with open("crates/forge_app/src/tool_registry.rs", "w") as f:
    f.write(tr)

with open("crates/forge_app/src/fmt/fmt_input.rs", "r") as f:
    fi = f.read()

fi = fi.replace("display_path_for(&input.file_path)", "display_path_for(&input.file_path.to_string_lossy())")
fi = fi.replace("display_path_for(&input.path)", "display_path_for(&input.path.to_string_lossy())")
fi = fi.replace("sub_title(&input.agent_id)", "sub_title(input.agent_id.to_string())")

with open("crates/forge_app/src/fmt/fmt_input.rs", "w") as f:
    f.write(fi)

