CREATE TABLE IF NOT EXISTS learning_ledger_events (
    event_seq INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    record_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    workspace_id BIGINT NOT NULL,
    event_kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    redaction_status TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    conversation_id TEXT,
    task_id TEXT,
    tool_name TEXT,
    eval_id TEXT,
    created_at TIMESTAMP NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_ledger_workspace_record_seq
ON learning_ledger_events(workspace_id, record_id, event_seq DESC);

CREATE INDEX IF NOT EXISTS idx_learning_ledger_workspace_kind_seq
ON learning_ledger_events(workspace_id, event_kind, event_seq DESC);

CREATE INDEX IF NOT EXISTS idx_learning_ledger_workspace_source
ON learning_ledger_events(workspace_id, source_kind, source_id, source_event_id);
