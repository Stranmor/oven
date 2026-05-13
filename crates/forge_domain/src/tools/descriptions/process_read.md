Reads captured stdout/stderr from a managed background process. The `process_id` handle comes from a shell timeout handoff when a shell command exceeds the synchronous execution window.

Pass the `cursor` returned by the previous `process_read` call to receive only new output. Omit it or pass `0` to read from the beginning. The result returns `next_cursor` for idempotent continuation. Recent completed process ids can still return captured output after they disappear from `process_list`.

Use `wait_seconds` when the same cursor previously returned no entries and more output is expected. The bounded wait returns early as soon as new output is captured or the process exits, and otherwise returns after the requested delay without advancing `next_cursor`. Prefer this over repeated immediate polling of the same cursor.
