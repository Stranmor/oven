use unicode_width::UnicodeWidthChar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnsiState { Text, Esc, Csi, Osc, OscEsc }

impl AnsiState {
    fn advance(&mut self, c: char) {
        match self {
            AnsiState::Text => if c == '\x1b' { *self = AnsiState::Esc; },
            AnsiState::Esc => {
                if c == '[' { *self = AnsiState::Csi; }
                else if c == ']' { *self = AnsiState::Osc; }
                else { *self = AnsiState::Text; }
            }
            AnsiState::Csi => {
                if (0x40..=0x7E).contains(&(c as u32)) { *self = AnsiState::Text; }
            }
            AnsiState::Osc => {
                if c == '\x07' { *self = AnsiState::Text; }
                else if c == '\x1b' { *self = AnsiState::OscEsc; }
            }
            AnsiState::OscEsc => {
                if c == '\\' { *self = AnsiState::Text; }
                else { *self = AnsiState::Osc; }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AnsiToken { Text(char), Sequence(String) }

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
    if !current_seq.is_empty() { tokens.push(AnsiToken::Sequence(current_seq)); }
    tokens
}

pub fn code_wrap(line: &str, width: usize, _pretty_broken: bool) -> (usize, Vec<String>) {
    let tokens = tokenize_ansi(line);
    let mut expanded_tokens = Vec::new();
    let mut visible_col = 0;
    
    for token in tokens {
        match token {
            AnsiToken::Text('\t') => {
                let spaces = 8 - (visible_col % 8);
                for _ in 0..spaces { expanded_tokens.push(AnsiToken::Text(' ')); }
                visible_col += spaces;
            }
            AnsiToken::Text(c) => {
                expanded_tokens.push(AnsiToken::Text(c));
                visible_col += c.width().unwrap_or(0);
            }
            AnsiToken::Sequence(s) => {
                expanded_tokens.push(AnsiToken::Sequence(s));
            }
        }
    }
    
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
    
    let continuation_indent_width = ("  ".repeat(indent.min(4) / 2 + 1)).len();
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
                    if !active_ansi.is_empty() { current_line.push_str(&active_ansi); }
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
    if !current_line.is_empty() || lines.is_empty() { lines.push(current_line); }
    (indent, lines)
}

fn main() {
    let line = "\x1b[31mThis is a long text with \x1b[1many\x1b[0m formatting that should wrap safely.\x1b[0m";
    let (i, lines) = code_wrap(line, 20, true);
    for l in &lines { println!("{:?}", l); }
}
