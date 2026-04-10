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

fn main() {
    let res = tokenize_ansi("\x1b[31mHello\x1b[0mWorld\x1b[20~!");
    println!("{:?}", res);
}
