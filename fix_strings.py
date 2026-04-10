import re
import os

with open("crates/forge_domain/src/compact/summary.rs", "r") as f:
    sum = f.read()

sum = sum.replace("input.path.to_string_lossy().to_string()", "input.path.clone()")
sum = sum.replace("input.file_path.to_string_lossy().to_string()", "input.file_path.clone()")
sum = sum.replace("input.agent_id.to_string()", "input.agent_id.clone()")
with open("crates/forge_domain/src/compact/summary.rs", "w") as f:
    f.write(sum)

with open("crates/forge_domain/src/tools/catalog.rs", "r") as f:
    cat = f.read()

cat = cat.replace("display_path_for(&input.path.to_string_lossy())", "display_path_for(&input.path)")
cat = cat.replace("display_path_for(&input.file_path.to_string_lossy())", "display_path_for(&input.file_path)")

with open("crates/forge_domain/src/tools/catalog.rs", "w") as f:
    f.write(cat)

