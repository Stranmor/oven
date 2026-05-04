Returns structured lifecycle status for a managed background process. The `process_id` handle may come from `process_start` or from a shell timeout handoff when a shell command exceeds the synchronous execution window.

The result includes the typed `process_id`, command, working directory, and status: running, exited with exit code, or killed.
