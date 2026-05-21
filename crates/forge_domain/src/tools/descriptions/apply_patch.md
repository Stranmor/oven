Use this tool for coordinated multi-file text patches when one logical change must touch more than one file with a shared all-or-none preflight.

Input:
- `patch`: a V4A-style patch string. The first non-empty line must be `*** Begin Patch` and the last non-empty line must be `*** End Patch`.

Supported first-slice operations:
- `*** Update File: <path>` with strict hunks using lines prefixed by `-` for removed text and `+` for replacement text. Blank hunk lines must still be prefixed. Matching is exact and deterministic; there is no fuzzy fallback.
- `*** Add File: <path>` with new file lines prefixed by `+`. The file must not already exist and the parent directory must already exist.
- `*** Delete File: <path>` is parsed but currently rejected fail-closed. Rename/move operations are not supported.

Use `multi_patch` instead when you need multiple exact replacements in one existing file. Use `apply_patch` only when the patch coordinates multiple file-level operations or when the V4A patch format is the natural source format.

Safety model:
- Every target path is normalized relative to the current working directory and must remain inside the workspace.
- Duplicate/conflicting targets are rejected before mutation.
- Every `Update File` target must have been read earlier in the conversation.
- Permissions are checked for every touched path before execution.
- All update hunks and add-file checks are validated before any write.
- Existing touched files are snapshotted before writing.
- This is not a full filesystem transaction. It provides strict all-or-none preflight plus snapshot/checkpoint semantics; it does not claim rollback if the operating system fails mid-write.

Example:

```text
*** Begin Patch
*** Update File: src/lib.rs
- old line
+ new line
*** Add File: src/new_module.rs
+ pub fn value() -> u8 {
+     1
+ }
*** End Patch
```
