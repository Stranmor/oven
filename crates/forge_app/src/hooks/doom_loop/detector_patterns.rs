use pretty_assertions::assert_eq;

use super::fixtures::{
    assistant_message, assistant_message_with_tool_calls, conversation_from_tool_calls,
    conversation_with_context_messages, conversation_with_messages, detector_with_threshold,
    tool_call,
};
use super::*;
use forge_domain::{ContextMessage, ToolOutput, ToolResult};

#[test]
fn test_doom_loop_detector_detects_repeating_pattern_123_123_123() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let third = tool_call("patch", r#"{"path": "file3.txt"}"#);
    let conversation = conversation_from_tool_calls(&[
        first.clone(),
        second.clone(),
        third.clone(),
        first.clone(),
        second.clone(),
        third.clone(),
        first,
        second,
        third,
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_detects_repeating_pattern_12_12_12() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let conversation = conversation_from_tool_calls(&[
        first.clone(),
        second.clone(),
        first.clone(),
        second.clone(),
        first,
        second,
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_no_pattern_with_partial_repetition() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let third = tool_call("patch", r#"{"path": "file3.txt"}"#);
    let conversation =
        conversation_from_tool_calls(&[first.clone(), second.clone(), third, first, second]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_pattern_with_custom_threshold() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let conversation =
        conversation_from_tool_calls(&[first.clone(), second.clone(), first, second]);

    let actual = detector_with_threshold(2).detect_from_conversation(&conversation);
    let expected = Some(2);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_complex_pattern_1234_1234_1234() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let third = tool_call("patch", r#"{"path": "file3.txt"}"#);
    let fourth = tool_call("shell", r#"{"command": "ls"}"#);
    let conversation = conversation_from_tool_calls(&[
        first.clone(),
        second.clone(),
        third.clone(),
        fourth.clone(),
        first.clone(),
        second.clone(),
        third.clone(),
        fourth.clone(),
        first,
        second,
        third,
        fourth,
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_collapses_equivalent_read_intents() {
    let first = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let second = tool_call("read", r#"{"file_path":"./src/main.rs"}"#);
    let third = tool_call("Read", r#"{"file_path":"src/main.rs"}"#);
    let conversation = conversation_from_tool_calls(&[first, second, third]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_treats_progress_results_as_boundary() {
    let first = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let second = tool_call("read", r#"{"file_path":"./src/main.rs"}"#);
    let third = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let conversation = conversation_with_context_messages(vec![
        ContextMessage::Text(assistant_message(&first)),
        ContextMessage::Tool(ToolResult::new("read").output(Ok(ToolOutput::text(
            r#"<file path="src/main.rs" total_lines="100">"#,
        )))),
        ContextMessage::Text(assistant_message(&second)),
        ContextMessage::Text(assistant_message(&third)),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_does_not_reset_on_error_shaped_output() {
    let first = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let second = tool_call("read", r#"{"file_path":"./src/main.rs"}"#);
    let third = tool_call("read", r#"{"path":"src/main.rs"}"#);
    let error = ToolOutput::text(r#"<file path="src/main.rs">"#).is_error(true);
    let conversation = conversation_with_context_messages(vec![
        ContextMessage::Text(assistant_message(&first)),
        ContextMessage::Tool(ToolResult::new("read").output(Ok(error))),
        ContextMessage::Text(assistant_message(&second)),
        ContextMessage::Text(assistant_message(&third)),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_progress_boundary_is_intent_local() {
    let first = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let second = tool_call("read", r#"{"file_path":"./src/main.rs"}"#);
    let third = tool_call("read", r#"{"path":"src/main.rs"}"#);
    let conversation = conversation_with_context_messages(vec![
        ContextMessage::Text(assistant_message(&first)),
        ContextMessage::Tool(ToolResult::new("read").output(Ok(ToolOutput::text(
            r#"<file path="src/lib.rs" total_lines="100">"#,
        )))),
        ContextMessage::Text(assistant_message(&second)),
        ContextMessage::Text(assistant_message(&third)),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_normalizes_search_field_order() {
    let first = tool_call(
        "fs_search",
        r#"{"path":"src","pattern":"doom","glob":"*.rs"}"#,
    );
    let second = tool_call(
        "fs_search",
        r#"{"glob":"*.rs","pattern":"doom","path":"src"}"#,
    );
    let third = tool_call(
        "fs_search",
        r#"{"pattern":"doom","path":"src","glob":"*.rs"}"#,
    );
    let conversation = conversation_from_tool_calls(&[first, second, third]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_does_not_treat_parallel_calls_as_retries() {
    let first = tool_call("read", r#"{"path":"src/main.rs"}"#);
    let second = tool_call("read", r#"{"path":"./src/main.rs"}"#);
    let third = tool_call("read", r#"{"file_path":"src/main.rs"}"#);
    let conversation = conversation_with_messages(vec![assistant_message_with_tool_calls(vec![
        first, second, third,
    ])]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_treats_process_read_output_as_progress_boundary() {
    let first = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let second = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let third = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let conversation = conversation_with_context_messages(vec![
        ContextMessage::Text(assistant_message(&first)),
        ContextMessage::Tool(ToolResult::new("process_read").output(Ok(ToolOutput::text(
            r#"<process_output process_id="process-1" next_cursor="1"><![CDATA[{"cursor":1,"stream":"stdout","content":"ready"}]]></process_output>"#,
        )))),
        ContextMessage::Text(assistant_message(&second)),
        ContextMessage::Text(assistant_message(&third)),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_keeps_process_read_without_progress_in_loop_scope() {
    let first = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let second = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let third = tool_call("process_read", r#"{"process_id":"process-1","cursor":0}"#);
    let conversation = conversation_with_context_messages(vec![
        ContextMessage::Text(assistant_message(&first)),
        ContextMessage::Tool(ToolResult::new("process_read").output(Ok(ToolOutput::text(
            r#"<process_output process_id="process-1" next_cursor="0"><![CDATA[[]]]></process_output>"#,
        )))),
        ContextMessage::Text(assistant_message(&second)),
        ContextMessage::Text(assistant_message(&third)),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_real_world_scenario() {
    let read = tool_call("read", r#"{"path": "src/main.rs"}"#);
    let diagnostics = tool_call(
        "mcp_forge_extension_tool_get_diagnostics",
        r#"{"severity": "error"}"#,
    );
    let patch = tool_call(
        "patch",
        r#"{"path": "src/main.rs", "old": "foo", "new": "bar"}"#,
    );
    let conversation = conversation_from_tool_calls(&[
        read.clone(),
        diagnostics.clone(),
        patch.clone(),
        read.clone(),
        diagnostics.clone(),
        patch.clone(),
        read,
        diagnostics,
        patch,
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}
