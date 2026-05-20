// @generated automatically by Diesel CLI.

diesel::table! {
    conversations (conversation_id) {
        conversation_id -> Text,
        title -> Nullable<Text>,
        workspace_id -> BigInt,
        context -> Nullable<Text>,
        created_at -> Timestamp,
        updated_at -> Nullable<Timestamp>,
        metrics -> Nullable<Text>,
        parent_id -> Nullable<Text>,
        initiator -> Nullable<Text>,
    }
}

diesel::table! {
    learning_ledger_events (event_seq) {
        event_seq -> BigInt,
        event_id -> Text,
        record_id -> Text,
        idempotency_key -> Text,
        workspace_id -> BigInt,
        event_kind -> Text,
        summary -> Text,
        content_fingerprint -> Text,
        redaction_status -> Text,
        source_kind -> Text,
        source_id -> Text,
        source_event_id -> Text,
        source_fingerprint -> Text,
        conversation_id -> Nullable<Text>,
        task_id -> Nullable<Text>,
        tool_name -> Nullable<Text>,
        eval_id -> Nullable<Text>,
        capture_metadata -> Nullable<Text>,
        created_at -> Timestamp,
        schema_version -> Integer,
    }
}

diesel::table! {
    subagent_task_sessions (task_id) {
        task_id -> Text,
        agent_id -> Text,
        conversation_id -> Text,
        parent_conversation_id -> Nullable<Text>,
        root_conversation_id -> Nullable<Text>,
        workspace_id -> BigInt,
        status -> Text,
        task -> Text,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        heartbeat_at -> Timestamp,
        final_result -> Nullable<Text>,
        final_error -> Nullable<Text>,
        delivered_at -> Nullable<Timestamp>,
    }
}
