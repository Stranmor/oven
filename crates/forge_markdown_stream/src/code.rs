//! Code block rendering with syntax highlighting and line wrapping.

use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;
use unicode_width::UnicodeWidthChar;

use crate::utils::{ThemeMode, detect_theme_mode};

const RESET: &str = "\x1b[0m";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnsiState {
    Text,
    Esc,
    Csi,
    Osc,
    OscEsc,
}

impl AnsiState {
    fn advance(&mut self, c: char) {
        match self {
            AnsiState::Text => {
                if c == '\x1b' {
                    *self = AnsiState::Esc;
                }
            }
            AnsiState::Esc => {
                if c == '[' {
                    *self = AnsiState::Csi;
                } else if c == ']' {
                    *self = AnsiState::Osc;
                } else {
                    *self = AnsiState::Text;
                }
            }
            AnsiState::Csi => {
                if (0x40..=0x7E).contains(&(c as u32)) {
                    *self = AnsiState::Text;
                }
            }
            AnsiState::Osc => {
                if c == '\x07' {
                    *self = AnsiState::Text;
                } else if c == '\x1b' {
                    *self = AnsiState::OscEsc;
                }
            }
            AnsiState::OscEsc => {
                if c == '\\' {
                    *self = AnsiState::Text;
                } else {
                    *self = AnsiState::Osc;
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AnsiToken {
    Text(char),
    Sequence(String),
}

#[allow(clippy::if_same_then_else, reason = "Pre-existing lint")]
fn tokenize_ansi(line: &str) -> Vec<AnsiToken> {
    let mut tokens = Vec::new();
    let mut state = AnsiState::Text;
    let mut current_seq = String::new();
    
    for c in line.chars() {
        let prev_state = state;
        state.advance(c);
        
        if prev_state == AnsiState::Text && state == AnsiState::Esc {
            current_seq.push(c);
        } else if state != AnsiState::Text {
            current_seq.push(c);
        } else if prev_state != AnsiState::Text && state == AnsiState::Text {
            current_seq.push(c);
            tokens.push(AnsiToken::Sequence(current_seq.clone()));
            current_seq.clear();
        } else {
            tokens.push(AnsiToken::Text(c));
        }
    }
    
    if !current_seq.is_empty() {
        tokens.push(AnsiToken::Sequence(current_seq));
    }
    
    tokens
}

#[allow(clippy::arithmetic_side_effects, reason = "Pre-existing lint")]
/// Wraps a line of code, respecting ANSI escape sequences and expanding tabs.
/// Returns the leading indentation level and the wrapped lines.
pub fn code_wrap(line: &str, width: usize, _pretty_broken: bool) -> (usize, Vec<String>) {
    let tokens = tokenize_ansi(line);
    
    // 1. Expand tabs properly
    let mut expanded_tokens = Vec::new();
    let mut visible_col = 0;
    
    for token in tokens {
        match token {
            AnsiToken::Text('\t') => {
                let spaces = 8_usize.saturating_sub(visible_col % 8);
                for _ in 0..spaces {
                    expanded_tokens.push(AnsiToken::Text(' '));
                }
                visible_col = visible_col.saturating_add(spaces);
            }
            AnsiToken::Text(c) => {
                expanded_tokens.push(AnsiToken::Text(c));
                visible_col = visible_col.saturating_add(c.width().unwrap_or(0));
            }
            AnsiToken::Sequence(s) => {
                expanded_tokens.push(AnsiToken::Sequence(s));
            }
        }
    }
    
    // 2. Find indentation
    let mut indent = 0;
    for token in &expanded_tokens {
        match token {
            AnsiToken::Text(' ') => indent += 1,
            AnsiToken::Sequence(_) => continue,
            _ => break,
        }
    }
    
    if width == 0 {
        let mut full_line = String::new();
        for t in expanded_tokens {
            match t {
                AnsiToken::Text(c) => full_line.push(c),
                AnsiToken::Sequence(s) => full_line.push_str(&s),
            }
        }
        return (indent, vec![full_line]);
    }
    
    // 3. Wrap line using visible width
    let continuation_indent_width = (indent.min(4) / 2 + 1) * 2;
    
    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0;
    let mut active_ansi = String::new();
    
    let mut current_max_width = width;
    
    for token in expanded_tokens {
        match token {
            AnsiToken::Text(c) => {
                let char_width = c.width().unwrap_or(0);
                
                if current_width + char_width > current_max_width && current_width > 0 {
                    lines.push(current_line.clone());
                    current_line = String::new();
                    current_width = 0;
                    current_max_width = width.saturating_sub(continuation_indent_width);
                    
                    if !active_ansi.is_empty() {
                        current_line.push_str(&active_ansi);
                    }
                }
                
                current_line.push(c);
                current_width += char_width;
            }
            AnsiToken::Sequence(s) => {
                current_line.push_str(&s);
                
                if s == "\x1b[0m" || s == "\x1b[m" {
                    active_ansi.clear();
                } else if s.starts_with("\x1b[") && s.ends_with('m') {
                    active_ansi.push_str(&s);
                }
            }
        }
    }
    
    if !current_line.is_empty() || lines.is_empty() {
        lines.push(current_line);
    }
    
    (indent, lines)
}

/// Code block highlighter using syntect.
pub struct CodeHighlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
    theme_mode: ThemeMode,
}

impl Default for CodeHighlighter {
    fn default() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
            theme_mode: detect_theme_mode(),
        }
    }
}

impl CodeHighlighter {
    #[allow(clippy::indexing_slicing, reason = "Pre-existing lint")]
    /// Highlight a single line of code.
    fn highlight_line(&self, line: &str, language: Option<&str>) -> String {
        let syntax = language
            .and_then(|lang| self.syntax_set.find_syntax_by_token(lang))
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let theme_name = match self.theme_mode {
            ThemeMode::Dark => "base16-ocean.dark",
            ThemeMode::Light => "InspiredGitHub",
        };
        let theme = self.theme_set.themes.get(theme_name).unwrap_or_else(|| {
            self.theme_set
                .themes
                .values()
                .next()
                .expect("ThemeSet is empty")
        });
        let mut highlighter = HighlightLines::new(syntax, theme);

        match highlighter.highlight_line(line, &self.syntax_set) {
            Ok(ranges) => as_24_bit_terminal_escaped(&ranges[..], false),
            Err(_) => line.to_string(),
        }
    }

    #[allow(clippy::arithmetic_side_effects, reason = "Pre-existing lint")]
    /// Render a code line with margin, wrapping if needed.
    ///
    /// Returns multiple lines if the code exceeds the available width.
    pub fn render_code_line(
        &self,
        line: &str,
        language: Option<&str>,
        margin: &str,
        width: usize,
    ) -> Vec<String> {
        let highlighted = self.highlight_line(line, language);
        let (indent, wrapped_lines) = code_wrap(&highlighted, width, true);

        let mut result = Vec::new();

        for (i, code_line) in wrapped_lines.iter().enumerate() {
            let line_indent = if i == 0 {
                String::new()
            } else {
                "  ".repeat(indent.min(4) / 2 + 1)
            };

            result.push(format!("{}{}{}{}", margin, line_indent, code_line, RESET));
        }

        if result.is_empty() {
            result.push(format!("{}{}", margin, RESET));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::code_wrap;

    #[test]
    fn test_code_wrap_short_line() {
        let (indent, lines) = code_wrap("let x = 1;", 80, true);
        assert_eq!(indent, 0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "let x = 1;");
    }

    #[test]
    fn test_code_wrap_with_indent() {
        let (indent, lines) = code_wrap("    let x = 1;", 80, true);
        assert_eq!(indent, 4);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_code_wrap_long_line() {
        let long_line = "x".repeat(100);
        let (_, lines) = code_wrap(&long_line, 40, true);
        assert!(lines.len() > 1);
    }

    #[test]
    fn test_code_wrap_empty() {
        let (indent, lines) = code_wrap("", 80, true);
        assert_eq!(indent, 0);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_code_wrap_ansi() {
        let line = "\x1b[31mThis is a long text with \x1b[1many\x1b[0m formatting that should wrap safely.\x1b[0m";
        let (_, lines) = code_wrap(line, 20, true);
        assert_eq!(lines[0], "\x1b[31mThis is a long text ");
        assert_eq!(lines[1], "\x1b[31mwith \x1b[1many\x1b[0m formattin");
        assert_eq!(lines[2], "g that should wrap");
        assert_eq!(lines[3], " safely.\x1b[0m");
    }

    #[test]
    fn test_code_wrap_tabs() {
        let (indent, lines) = code_wrap("\tlet x = 1;", 80, true);
        assert_eq!(indent, 8);
        assert_eq!(lines[0], "        let x = 1;");
    }
}
