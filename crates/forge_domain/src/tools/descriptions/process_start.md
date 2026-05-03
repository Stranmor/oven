Starts a shell command as a managed background process and returns immediately with a `process_id` handle.

Use this for long-running servers, watchers, REPL-adjacent daemons, and commands expected to run longer than an interactive foreground turn. Do not use it for short commands; use `shell` instead.

The process captures stdout/stderr asynchronously. Use `process_status` to inspect lifecycle state, `process_read` to read captured output with a cursor, `process_list` to see active handles, and `process_kill` to stop a process.

The `cwd` parameter sets the working directory. Do not put `cd` in the command string.
