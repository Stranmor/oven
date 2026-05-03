use forge_domain::{ContextMessage, Role};
use pretty_assertions::assert_eq;

use super::fixtures::{
    assistant_message, event, repeated_read_conversation, text_message, tool_call,
};
use super::*;

fn system_reminder_count(conversation: &Conversation) -> usize {
    conversation
        .context
        .as_ref()
        .unwrap()
        .messages
        .iter()
        .filter(|entry| {
            entry.message.has_role(Role::System)
                && entry
                    .message
                    .content()
                    .is_some_and(|content| content.contains("system_reminder"))
        })
        .count()
}

#[tokio::test]
async fn test_doom_loop_handler_injects_system_level_reminder() {
    let fixture = event();
    let mut conversation = repeated_read_conversation();

    DoomLoopDetector::new()
        .handle(&fixture, &mut conversation)
        .await
        .unwrap();

    let actual = system_reminder_count(&conversation);
    let expected = 1;
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_doom_loop_handler_suppresses_duplicate_reminder_for_same_loop() {
    let fixture = event();
    let mut conversation = repeated_read_conversation();
    let detector = DoomLoopDetector::new();

    detector.handle(&fixture, &mut conversation).await.unwrap();
    detector.handle(&fixture, &mut conversation).await.unwrap();

    let actual = system_reminder_count(&conversation);
    let expected = 1;
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_doom_loop_handler_allows_new_reminder_after_new_loop_evidence() {
    let fixture = event();
    let mut conversation = repeated_read_conversation();
    let detector = DoomLoopDetector::new();

    detector.handle(&fixture, &mut conversation).await.unwrap();
    let tool_call = tool_call("read", r#"{"path": "file.txt"}"#);
    conversation
        .context
        .as_mut()
        .unwrap()
        .messages
        .push(ContextMessage::Text(assistant_message(&tool_call)).into());
    detector.handle(&fixture, &mut conversation).await.unwrap();

    let actual = system_reminder_count(&conversation);
    let expected = 2;
    assert_eq!(actual, expected);
}

#[test]
fn test_extract_assistant_messages() {
    let assistant_first = text_message(Role::Assistant, "Response 1");
    let user = text_message(Role::User, "Question");
    let assistant_second = text_message(Role::Assistant, "Response 2");
    let fixture = [
        ContextMessage::Text(assistant_first.clone()),
        ContextMessage::Text(user),
        ContextMessage::Text(assistant_second.clone()),
    ];

    let actual = DoomLoopDetector::extract_assistant_messages(fixture.iter())
        .into_iter()
        .map(|message| message.content.clone())
        .collect::<Vec<_>>();
    let expected = vec!["Response 1".to_string(), "Response 2".to_string()];
    assert_eq!(actual, expected);
}
