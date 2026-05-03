use pretty_assertions::assert_eq;

use super::fixtures::{conversation_from_tool_calls, detector_with_threshold, tool_call};
use super::*;

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
