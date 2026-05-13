Lists currently running managed background process handles known to the current Forge runtime.

Completed or killed managed processes are intentionally omitted from this inventory. Recent completed process ids may still be queried directly with `process_status` and `process_read` for status and captured-output readback.

Use this to discover active handles for status, log reading, and cleanup before starting duplicate servers or watchers.
