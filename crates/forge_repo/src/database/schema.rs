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
