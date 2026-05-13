Returns structured lifecycle status for a managed background process. The `process_id` handle comes from a shell timeout handoff when a shell command exceeds the synchronous execution window.

The result includes the typed `process_id`, command, working directory, and status: running, exited with exit code, or killed. Recent completed process ids can still be queried after they disappear from `process_list`.

Use `wait_seconds` when a process is expected to finish soon. The bounded wait returns early when the process leaves `running`, and otherwise returns the current status after the requested delay. Prefer this over repeated immediate status polling.
