Reads captured stdout/stderr from a managed background process previously started with `process_start`.

Pass the `cursor` returned by the previous `process_read` call to receive only new output. Omit it or pass `0` to read from the beginning. The result returns `next_cursor` for idempotent continuation.
