//! Rust symbol source line range resolution.

use std::collections::BTreeMap;

use crate::types::SymbolKind;
use crate::util::line_number_from_index;

#[derive(Default)]
pub(super) struct SymbolRangeResolver {
    cursors: BTreeMap<(SymbolKind, String), usize>,
}

impl SymbolRangeResolver {
    pub(super) fn line_range(
        &mut self,
        content: &str,
        name: &str,
        kind: &SymbolKind,
    ) -> (u32, u32) {
        let sanitized_lines = sanitized_rust_lines(content);
        let key = (kind.clone(), name.to_string());
        let start_at = self.cursors.get(&key).copied().unwrap_or_default();
        let range = symbol_line_range(&sanitized_lines, name, kind, start_at);
        self.cursors
            .insert(key, usize::try_from(range.1).unwrap_or(usize::MAX));
        range
    }
}

fn sanitized_rust_lines(content: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = content.chars().peekable();
    let mut block_comment_depth = 0usize;
    let mut string_literal = false;
    let mut char_literal = false;
    let mut raw_string_hashes: Option<usize> = None;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if ch == '\n' {
            current.push('\n');
            lines.push(std::mem::take(&mut current));
            escaped = false;
            continue;
        }
        if let Some(hash_count) = raw_string_hashes {
            if ch == '"' {
                let mut consumed_hashes = 0usize;
                while consumed_hashes < hash_count && chars.peek() == Some(&'#') {
                    chars.next();
                    consumed_hashes = consumed_hashes
                        .checked_add(1)
                        .expect("raw string hash count should not overflow");
                }
                if consumed_hashes == hash_count {
                    raw_string_hashes = None;
                }
            }
            current.push(' ');
            continue;
        }
        if block_comment_depth > 0 {
            if ch == '/' && chars.peek() == Some(&'*') {
                chars.next();
                block_comment_depth = block_comment_depth
                    .checked_add(1)
                    .expect("block comment depth should not overflow");
                current.push(' ');
                current.push(' ');
            } else if ch == '*' && chars.peek() == Some(&'/') {
                chars.next();
                block_comment_depth = block_comment_depth.saturating_sub(1);
                current.push(' ');
                current.push(' ');
            } else {
                current.push(' ');
            }
            continue;
        }
        if string_literal {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                string_literal = false;
            }
            current.push(' ');
            continue;
        }
        if char_literal {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '\'' {
                char_literal = false;
            }
            current.push(' ');
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            current.push(' ');
            current.push(' ');
            while let Some(next) = chars.peek() {
                if *next == '\n' {
                    break;
                }
                chars.next();
                current.push(' ');
            }
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            block_comment_depth = 1;
            current.push(' ');
            current.push(' ');
            continue;
        }
        if ch == 'r' {
            let mut probe = chars.clone();
            let mut hash_count = 0usize;
            while probe.peek() == Some(&'#') {
                probe.next();
                hash_count = hash_count
                    .checked_add(1)
                    .expect("raw string hash count should not overflow");
            }
            if probe.peek() == Some(&'"') {
                for _ in 0..hash_count {
                    chars.next();
                }
                chars.next();
                raw_string_hashes = Some(hash_count);
                current.push(' ');
                for _ in 0..hash_count {
                    current.push(' ');
                }
                current.push(' ');
                continue;
            }
        }
        if ch == '"' {
            string_literal = true;
            current.push(' ');
            continue;
        }
        if ch == '\'' {
            let mut probe = chars.clone();
            if matches!(probe.next(), Some(next) if next.is_alphanumeric() || next == '_')
                && probe.peek() != Some(&'\'')
            {
                current.push(ch);
            } else {
                char_literal = true;
                current.push(' ');
            }
            continue;
        }
        current.push(ch);
    }
    lines.push(current);
    lines
}

fn extract_name_after_keyword(line: &str, keyword: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let needle = format!("{keyword} ");
    let start = trimmed.find(&needle)?;
    let prefix = trimmed.get(..start)?;
    if prefix
        .chars()
        .next_back()
        .is_some_and(|previous| previous.is_alphanumeric() || previous == '_')
    {
        return None;
    }
    let rest_start = start
        .checked_add(needle.len())
        .expect("keyword match offset should be within the line");
    let rest = trimmed.get(rest_start..)?;
    Some(
        rest.trim_start()
            .chars()
            .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
            .collect::<String>(),
    )
    .filter(|name| !name.is_empty())
}

fn symbol_line_range(
    lines: &[String],
    name: &str,
    kind: &SymbolKind,
    start_at: usize,
) -> (u32, u32) {
    let keywords = match kind {
        SymbolKind::Struct => vec!["struct"],
        SymbolKind::Enum => vec!["enum"],
        SymbolKind::Trait => vec!["trait"],
        SymbolKind::Function | SymbolKind::Test | SymbolKind::Method => vec!["fn"],
        SymbolKind::Module => vec!["mod"],
        SymbolKind::Impl => vec!["impl"],
        SymbolKind::Unknown => vec![],
    };
    let mut start = 1u32;
    for (index, line) in lines.iter().enumerate().skip(start_at) {
        let matched = if *kind == SymbolKind::Impl {
            let trimmed = line.trim_start();
            let signature = name.strip_prefix("impl ").unwrap_or(name);
            (starts_with_impl_signature(trimmed) || starts_with_unsafe_impl_signature(trimmed))
                && trimmed.contains(signature)
        } else {
            keywords
                .iter()
                .filter_map(|keyword| extract_name_after_keyword(line, keyword))
                .any(|candidate| candidate == name)
        };
        if matched {
            start = line_number_from_index(index).unwrap_or(u32::MAX);
            break;
        }
    }
    let end = balanced_block_end(lines, start).unwrap_or(start);
    (start, end)
}

fn starts_with_impl_signature(line: &str) -> bool {
    line.strip_prefix("impl")
        .and_then(|rest| rest.chars().next())
        .is_some_and(|ch| ch.is_whitespace() || ch == '<')
}

fn starts_with_unsafe_impl_signature(line: &str) -> bool {
    line.strip_prefix("unsafe")
        .map(str::trim_start)
        .is_some_and(starts_with_impl_signature)
}

fn balanced_block_end(lines: &[String], start: u32) -> Option<u32> {
    let mut depth = 0i32;
    let mut seen_open = false;
    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(usize::try_from(start.saturating_sub(1)).unwrap_or(usize::MAX))
    {
        for ch in line.chars() {
            if ch == '{' {
                depth = depth.saturating_add(1);
                seen_open = true;
            } else if ch == '}' {
                depth = depth.saturating_sub(1);
            }
        }
        if seen_open && depth <= 0 {
            return line_number_from_index(index);
        }
        if !seen_open && line.trim_end().ends_with(';') {
            return line_number_from_index(index);
        }
    }
    Some(start)
}
