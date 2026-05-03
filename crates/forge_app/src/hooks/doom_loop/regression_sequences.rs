use pretty_assertions::assert_eq;

use super::fixtures::{conversation_from_tool_calls, tool_call};
use super::*;

#[test]
fn test_doom_loop_detector_pattern_changes_midway_123123454545() {
    let first = tool_call("read", r#"{"path": "file1.txt"}"#);
    let second = tool_call("write", r#"{"path": "file2.txt"}"#);
    let third = tool_call("patch", r#"{"path": "file3.txt"}"#);
    let fourth = tool_call("shell", r#"{"command": "ls"}"#);
    let fifth = tool_call("fs_search", r#"{"pattern": "test"}"#);
    let conversation = conversation_from_tool_calls(&[
        first.clone(),
        second.clone(),
        third.clone(),
        first,
        second,
        third,
        fourth.clone(),
        fifth.clone(),
        fourth.clone(),
        fifth.clone(),
        fourth,
        fifth,
    ]);

    let actual = DoomLoopDetector::new().detect_from_conversation(&conversation);
    let expected = Some(3);
    assert_eq!(actual, expected);
}

#[test]
fn test_doom_loop_detector_sequence_1234546454545_step_by_step() {
    let detector = DoomLoopDetector::new();
    let tool_1 = tool_call("read", r#"{"path": "file1.txt"}"#);
    let tool_2 = tool_call("write", r#"{"path": "file2.txt"}"#);
    let tool_3 = tool_call("patch", r#"{"path": "file3.txt"}"#);
    let tool_4 = tool_call("shell", r#"{"command": "ls"}"#);
    let tool_5 = tool_call("fs_search", r#"{"pattern": "test"}"#);
    let tool_6 = tool_call("sem_search", r#"{"queries": []}"#);
    let mut fixture = Vec::new();
    let steps = [
        (&tool_1, None),
        (&tool_2, None),
        (&tool_3, None),
        (&tool_4, None),
        (&tool_5, None),
        (&tool_4, None),
        (&tool_6, None),
        (&tool_4, None),
        (&tool_5, None),
        (&tool_4, None),
        (&tool_5, None),
        (&tool_4, None),
        (&tool_5, Some(3)),
    ];

    for (tool, expected) in steps {
        fixture.push(tool.clone());
        let conversation = conversation_from_tool_calls(&fixture);

        let actual = detector.detect_from_conversation(&conversation);
        assert_eq!(actual, expected);
    }
}
