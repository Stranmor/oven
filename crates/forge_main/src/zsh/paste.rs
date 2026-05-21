//! Pasted-text formatting utilities.
//!
//! Bracketed paste content is inserted literally except for terminal newline
//! normalization. Explicit `@[...]` markers are preserved as typed text and are
//! interpreted later by the attachment pipeline.

/// Normalizes pasted text without converting plain paths into attachments.
///
/// Called when a bracketed-paste event is received. The pasted content is
/// normalized from CRLF/CR to LF and otherwise returned unchanged. This keeps a
/// plain pasted file path as literal text while still preserving explicit
/// `@[...]` marker syntax for the downstream attachment pipeline.
pub fn wrap_pasted_text(pasted: &str) -> String {
    pasted.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_wrap_pasted_text_cjk_no_paths() {
        let fixture = "公";
        let actual = wrap_pasted_text(fixture);
        let expected = "公";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_no_paths() {
        let fixture = "hello world";
        let actual = wrap_pasted_text(fixture);
        let expected = "hello world";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_preserves_explicit_attachment_marker() {
        let fixture = "check @[/usr/bin/env]";
        let actual = wrap_pasted_text(fixture);
        let expected = "check @[/usr/bin/env]";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_existing_file_path_remains_literal() {
        let fixture = "look at /usr/bin/env please";
        let actual = wrap_pasted_text(fixture);
        let expected = "look at /usr/bin/env please";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_bare_existing_file_path_remains_literal() {
        let fixture = "/usr/bin/env";
        let actual = wrap_pasted_text(fixture);
        let expected = "/usr/bin/env";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_existing_directory_path_remains_literal() {
        let fixture = "/tmp";
        let actual = wrap_pasted_text(fixture);
        let expected = "/tmp";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_quoted_existing_path_remains_literal() {
        let fixture = "'/usr/bin/env'";
        let actual = wrap_pasted_text(fixture);
        let expected = "'/usr/bin/env'";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_backslash_escaped_path_remains_literal() {
        let fixture = "/tmp/my\\ file.txt";
        let actual = wrap_pasted_text(fixture);
        let expected = "/tmp/my\\ file.txt";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_preserves_whitespace() {
        let fixture = "hello  world";
        let actual = wrap_pasted_text(fixture);
        let expected = "hello  world";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_crlf_normalised_without_attachment_rewrite() {
        let fixture = "/usr/bin/env\r\n";
        let actual = wrap_pasted_text(fixture);
        let expected = "/usr/bin/env\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_cr_normalised_without_attachment_rewrite() {
        let fixture = "/usr/bin/env\r";
        let actual = wrap_pasted_text(fixture);
        let expected = "/usr/bin/env\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_does_not_wrap_plain_text_in_xml() {
        let fixture = "Implement /goal without changing paste semantics";
        let actual = wrap_pasted_text(fixture);
        let expected = "Implement /goal without changing paste semantics";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_cyrillic_no_crash() {
        let fixture = "Проверь ПОЛНОСТЬЮ этот проект на соответствие КАЖДОГО пункта функционала исходному тексту задачи";
        let actual = wrap_pasted_text(fixture);
        let expected = fixture;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_wrap_pasted_text_cyrillic_with_mixed_paths_remains_literal() {
        let fixture = "Проверь /usr/bin/env и /tmp пожалуйста";
        let actual = wrap_pasted_text(fixture);
        let expected = "Проверь /usr/bin/env и /tmp пожалуйста";
        assert_eq!(actual, expected);
    }
}
