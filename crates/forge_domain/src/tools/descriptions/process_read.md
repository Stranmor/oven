Reads captured stdout/stderr from a managed background process. The `process_id` handle may come from `process_start` or from a shell timeout handoff when a shell command exceeds the synchronous execution window.

Pass the `cursor` returned by the previous `process_read` call to receive only new output. Omit it or pass `0` to read from the beginning. The result returns `next_cursor` for idempotent continuation.
