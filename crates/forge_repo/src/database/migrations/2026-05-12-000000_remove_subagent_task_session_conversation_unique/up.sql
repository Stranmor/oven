DROP INDEX IF EXISTS idx_subagent_task_sessions_workspace_parent_updated;
DROP INDEX IF EXISTS idx_subagent_task_sessions_workspace_conversation_updated;
DROP INDEX IF EXISTS idx_subagent_task_sessions_workspace_status_updated;

CREATE TABLE subagent_task_sessions_next (
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
    delivered_at TIMESTAMP
);

INSERT INTO subagent_task_sessions_next (
    task_id, agent_id, conversation_id, parent_conversation_id,
    root_conversation_id, workspace_id, status, task, created_at,
    updated_at, heartbeat_at, final_result, final_error, delivered_at
)
SELECT
    task_id, agent_id, conversation_id, parent_conversation_id,
    root_conversation_id, workspace_id, status, task, created_at,
    updated_at, heartbeat_at, final_result, final_error, delivered_at
FROM subagent_task_sessions;

DROP TABLE subagent_task_sessions;
ALTER TABLE subagent_task_sessions_next RENAME TO subagent_task_sessions;

CREATE INDEX IF NOT EXISTS idx_subagent_task_sessions_workspace_status_updated
ON subagent_task_sessions(workspace_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_subagent_task_sessions_workspace_conversation_updated
ON subagent_task_sessions(workspace_id, conversation_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_subagent_task_sessions_workspace_parent_updated
ON subagent_task_sessions(workspace_id, parent_conversation_id, updated_at DESC);
