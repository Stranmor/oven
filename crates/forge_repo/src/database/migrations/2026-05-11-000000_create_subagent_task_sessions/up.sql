CREATE TABLE IF NOT EXISTS subagent_task_sessions (
    task_id TEXT PRIMARY KEY NOT NULL,
    agent_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    parent_conversation_id TEXT,
    root_conversation_id TEXT,
    workspace_id BIGINT NOT NULL,
    status TEXT NOT NULL,
    task TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    heartbeat_at TIMESTAMP NOT NULL,
    final_result TEXT,
    final_error TEXT,
    delivered_at TIMESTAMP,
    UNIQUE(workspace_id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_subagent_task_sessions_workspace_status_updated
ON subagent_task_sessions(workspace_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_subagent_task_sessions_workspace_parent_updated
ON subagent_task_sessions(workspace_id, parent_conversation_id, updated_at DESC);
