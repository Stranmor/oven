use pretty_assertions::assert_eq;

use super::*;

#[test]
fn test_detect_pattern_start_with_integers_for_123_123_123() {
    let fixture = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];

    let actual = DoomLoopDetector::new().check_repeating_pattern(&fixture);
    let expected = Some((0, 3));
    assert_eq!(actual, expected);
}

#[test]
fn test_detect_pattern_start_with_integers_detects_recent_suffix_pattern() {
    let fixture = vec![1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5];

    let actual = DoomLoopDetector::new().check_repeating_pattern(&fixture);
    let expected = Some((6, 3));
    assert_eq!(actual, expected);
}

#[test]
fn test_detect_pattern_start_with_integers_detects_consecutive_identical() {
    let fixture = vec![1, 2, 3, 3, 3];

    let actual = DoomLoopDetector::new().check_repeating_pattern(&fixture);
    let expected = Some((2, 3));
    assert_eq!(actual, expected);
}

#[test]
fn test_detect_pattern_start_requires_complete_repetitions() {
    let fixture = vec![1, 2, 1, 2, 1];

    let actual = DoomLoopDetector::new().check_repeating_pattern(&fixture);
    let expected = None;
    assert_eq!(actual, expected);
}

#[test]
fn test_detect_pattern_start_uses_stable_threshold_minimum() {
    let fixture = vec![1, 2, 1];

    let actual = DoomLoopDetector::new()
        .threshold(0)
        .check_repeating_pattern(&fixture);
    let expected = None;
    assert_eq!(actual, expected);
}
