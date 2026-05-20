Actuator tool that continues exact-fact reference production only for the current workspace root.

Use this when the workspace exact-fact status is inactive and production is safe. The executor first validates that `workspace_path` canonicalizes to the current workspace root, then performs a read-only status preflight. It never accepts alternate modes and never exposes raw producer/status errors, source text, JSON-RPC, stdout, or stderr.

Input:
- `workspace_path`: workspace root path. Relative paths are resolved against the current working directory. The canonical requested path must equal the current workspace root.

Behavior:
- Already-active exact facts return a typed no-op status.
- Missing, stale, corrupt, or unreadable status states return typed non-produced statuses.
- Production is attempted at most once and only through the workspace service boundary.
- Postflight status is always attempted after a producer attempt, and active success is reported only when postflight proves exact facts are active.