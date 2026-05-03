use pretty_assertions::assert_eq;

use super::fixtures::{
    assistant_message, conversation_from_tool_calls, conversation_with_messages,
    detector_with_threshold, tool_call,
};
use super::*;

#[test]
fn test_doom_loop_detector_detects_identical_calls() {
    let fixture = tool_call("read", r#"{"path": "file.txt"}"#);
    let conversation = conversation_with_messages(vec![
        assistant_message(&fixture),
        assistant_message(&fixture),
        assistant_message(&fixture),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_no_loop_with_two_calls() {
    let fixture = tool_call("read", r#"{"path": "file.txt"}"#);
    let conversation = conversation_with_messages(vec![assistant_message(&fixture)]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_resets_on_different_arguments() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("read", r#"{"path": "file2.txt"}"#);
    let conversation = conversation_with_messages(vec![
        assistant_message(&first),
        assistant_message(&first),
        assistant_message(&second),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_resets_on_different_tool() {
    let first = tool_call("read", r#"{"path": "file.txt"}"#);
    let second = tool_call("write", r#"{"path": "file.txt"}"#);
    let conversation = conversation_with_messages(vec![
        assistant_message(&first),
        assistant_message(&first),
        assistant_message(&second),
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_custom_threshold() {
    let fixture = tool_call("read", r#"{"path": "file.txt"}"#);
    let conversation = conversation_with_messages(vec![
        assistant_message(&fixture),
        assistant_message(&fixture),
    ]);

    let actual = detector_with_threshold(2).detect_from_conversation(&conversation);
    let expected = Some(2);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_empty_history() {
    let conversation = conversation_with_messages(vec![]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_threshold_clamps_invalid_values_to_minimum() {
    let fixture = tool_call("read", r#"{"path": "file.txt"}"#);
    let conversation = conversation_with_messages(vec![
        assistant_message(&fixture),
        assistant_message(&fixture),
    ]);

    let actual = detector_with_threshold(1).detect_from_conversation(&conversation);
    let expected = Some(2);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_consecutive_identical_takes_precedence() {
    let fixture = tool_call("read", r#"{"path": "file1.txt"}"#);
    let conversation = conversation_from_tool_calls(&[fixture.clone(), fixture.clone(), fixture]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}
