//! Ratatui renderer and lightweight terminal session for Forge typed UI models.
//!
//! This crate is presentation-only. It depends on `ratatui`, `crossterm`, and
//! the typed `forge_ui_model` boundary. Runtime orchestration remains in
//! `forge_main`.

use std::convert::Infallible;
use std::io::{self, Stdout, Write};

use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use forge_ui_model::{UiBlock, UiModel, UiToolDetail, UiToolPhase, UiTurnPhase};
use ratatui::Terminal;
use ratatui::backend::{CrosstermBackend, TestBackend};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

/// Owns a live TUI render session and its append-only typed model.
pub struct TuiSession<W: Write> {
    model: UiModel,
    output: W,
    width: u16,
    height: u16,
}

impl<W: Write> TuiSession<W> {
    /// Creates a session with an explicit render area.
    ///
    /// # Arguments
    /// * `output` - Writable terminal or test buffer receiving rendered frames.
    /// * `width` - Render width in terminal cells.
    /// * `height` - Render height in terminal cells.
    pub fn new(output: W, width: u16, height: u16) -> Self {
        Self { model: UiModel::default(), output, width, height }
    }

    /// Appends a typed response model and renders the next frame.
    ///
    /// # Arguments
    /// * `event_model` - Non-empty typed model produced from one chat response.
    ///
    /// # Errors
    /// Returns an error if writing the rendered frame to the terminal fails.
    pub fn queue_and_render(&mut self, event_model: UiModel) -> io::Result<()> {
        for block in event_model.blocks {
            self.model.push(block);
        }
        self.render()
    }

    /// Renders the current typed model to the owned output.
    ///
    /// # Errors
    /// Returns an error if terminal clearing or writing fails.
    pub fn render(&mut self) -> io::Result<()> {
        execute!(self.output, Clear(ClearType::All), MoveTo(0, 0))?;
        write!(
            self.output,
            "{}",
            render_dashboard_to_string(&self.model, self.width, self.height)
        )?;
        self.output.flush()
    }

    /// Consumes the session and returns its output sink.
    pub fn into_output(self) -> W {
        self.output
    }
}

impl<W: Write> TuiRenderer for TuiSession<W> {
    fn queue_and_render(&mut self, event_model: UiModel) -> io::Result<()> {
        TuiSession::queue_and_render(self, event_model)
    }

    fn render_current(&mut self) -> io::Result<()> {
        self.render()
    }
}

/// Common renderer boundary for stdout and alternate-screen TUI sessions.
pub trait TuiRenderer {
    /// Appends a typed response model and renders the next visible frame.
    ///
    /// # Arguments
    /// * `event_model` - Non-empty typed model produced from one chat response.
    ///
    /// # Errors
    /// Returns an error if rendering or terminal I/O fails.
    fn queue_and_render(&mut self, event_model: UiModel) -> io::Result<()>;

    /// Temporarily leaves any raw alternate-screen terminal mode before external
    /// stdout/stderr is released to a tool.
    ///
    /// # Errors
    /// Returns an error if restoring the terminal to ordinary stdout mode fails.
    fn suspend_for_stdout(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Re-enters the renderer's terminal mode after a stdout/stderr tool has
    /// completed.
    ///
    /// # Errors
    /// Returns an error if re-entering the renderer mode or redrawing fails.
    fn resume_after_stdout(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Renders the current append-only typed model without appending a response.
    ///
    /// # Errors
    /// Returns an error if rendering or terminal I/O fails.
    fn render_current(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Result of one interactive TUI input read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiInput {
    /// User submitted a line of text.
    Submitted(String),
    /// User requested session exit with Ctrl+C, Ctrl+D, or terminal EOF.
    Exit,
}

/// Owns an alternate-screen interactive terminal session.
pub struct InteractiveTuiSession {
    model: UiModel,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    input: String,
    suspended_for_stdout: bool,
}

impl InteractiveTuiSession {
    /// Creates an alternate-screen TUI session on stdout.
    ///
    /// # Errors
    /// Returns an error if terminal raw mode, alternate-screen setup, or
    /// initial terminal construction fails.
    pub fn new() -> io::Result<Self> {
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        enable_raw_mode()?;

        let alternate_screen_result = execute!(terminal.backend_mut(), EnterAlternateScreen);
        if let Err(error) = alternate_screen_result {
            let _ = disable_raw_mode();
            return Err(error);
        }

        Ok(Self {
            model: UiModel::default(),
            terminal,
            input: String::new(),
            suspended_for_stdout: false,
        })
    }

    /// Renders the current interactive shell frame.
    ///
    /// # Errors
    /// Returns an error if drawing to the terminal fails.
    pub fn render(&mut self) -> io::Result<()> {
        self.terminal.draw(|frame| {
            draw_dashboard(frame, &self.model, Some(self.input.as_str()));
        })?;
        self.terminal.backend_mut().flush()?;
        Ok(())
    }

    /// Reads one submitted input line from the TUI input area.
    ///
    /// # Errors
    /// Returns an error if terminal event reading or redraw fails.
    pub fn read_input(&mut self) -> io::Result<TuiInput> {
        loop {
            match event::read()? {
                Event::Key(KeyEvent { code: KeyCode::Char('c'), modifiers, .. })
                    if modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    return Ok(TuiInput::Exit);
                }
                Event::Key(KeyEvent { code: KeyCode::Char('d'), modifiers, .. })
                    if modifiers.contains(KeyModifiers::CONTROL) && self.input.is_empty() =>
                {
                    return Ok(TuiInput::Exit);
                }
                Event::Key(KeyEvent { code: KeyCode::Enter, .. }) => {
                    let submitted = self.input.trim().to_string();
                    if submitted.is_empty() {
                        self.input.clear();
                        self.render()?;
                        continue;
                    }
                    self.input.clear();
                    return Ok(TuiInput::Submitted(submitted));
                }
                Event::Key(KeyEvent { code: KeyCode::Backspace, .. }) => {
                    self.input.pop();
                    self.render()?;
                }
                Event::Key(KeyEvent { code: KeyCode::Char(value), modifiers, .. }) => {
                    if !modifiers.contains(KeyModifiers::CONTROL)
                        && !modifiers.contains(KeyModifiers::ALT)
                    {
                        self.input.push(value);
                        self.render()?;
                    }
                }
                Event::Resize(_, _) => self.render()?,
                _ => {}
            }
        }
    }
}

impl TuiRenderer for InteractiveTuiSession {
    fn queue_and_render(&mut self, event_model: UiModel) -> io::Result<()> {
        for block in event_model.blocks {
            self.model.push(block);
        }
        self.render()
    }

    fn suspend_for_stdout(&mut self) -> io::Result<()> {
        if self.suspended_for_stdout {
            return Ok(());
        }

        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        self.suspended_for_stdout = true;
        Ok(())
    }

    fn resume_after_stdout(&mut self) -> io::Result<()> {
        if !self.suspended_for_stdout {
            return Ok(());
        }

        enable_raw_mode()?;
        let alternate_screen_result = execute!(self.terminal.backend_mut(), EnterAlternateScreen);
        if let Err(error) = alternate_screen_result {
            let _ = disable_raw_mode();
            return Err(error);
        }
        self.suspended_for_stdout = false;
        self.render()
    }

    fn render_current(&mut self) -> io::Result<()> {
        self.render()
    }
}

impl Drop for InteractiveTuiSession {
    fn drop(&mut self) {
        if !self.suspended_for_stdout {
            let _ = disable_raw_mode();
            let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        }
        let _ = self.terminal.show_cursor();
    }
}

/// Creates an interactive alternate-screen session targeting stdout.
///
/// # Errors
/// Returns an error if terminal setup or initial rendering fails.
pub fn interactive_session() -> io::Result<InteractiveTuiSession> {
    InteractiveTuiSession::new()
}

/// Creates a TUI session targeting stdout with the current terminal size.
pub fn stdout_session() -> TuiSession<io::Stdout> {
    let (width, height) = crossterm::terminal::size().unwrap_or((100, 30));
    TuiSession::new(io::stdout(), width, height)
}

/// Renders a typed UI model into a deterministic `ratatui` test backend string.
///
/// # Arguments
/// * `model` - The typed UI model to render.
/// * `width` - Test backend width in terminal cells.
/// * `height` - Test backend height in terminal cells.
pub fn render_dashboard_to_string(model: &UiModel, width: u16, height: u16) -> String {
    render_model_to_string(model, width, height, None)
}

/// Renders the initial interactive TUI shell into a deterministic string.
///
/// # Arguments
/// * `width` - Test backend width in terminal cells.
/// * `height` - Test backend height in terminal cells.
pub fn render_interactive_shell_to_string(width: u16, height: u16) -> String {
    render_model_to_string(&UiModel::default(), width, height, Some(""))
}

fn render_model_to_string(model: &UiModel, width: u16, height: u16, input: Option<&str>) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = infallible(ratatui::Terminal::new(backend));
    infallible(terminal.draw(|frame| {
        draw_dashboard(frame, model, input);
    }));

    terminal.backend().to_string()
}

fn draw_dashboard(frame: &mut ratatui::Frame<'_>, model: &UiModel, input: Option<&str>) {
    let area = frame.area();
    let footer_height = if input.is_some() { 5 } else { 3 };
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(footer_height),
        ])
        .split(area);

    let Some(header_area) = vertical.first().copied() else {
        return;
    };
    let Some(body_area) = vertical.get(1).copied() else {
        return;
    };
    let Some(footer_area) = vertical.get(2).copied() else {
        return;
    };

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(body_area);

    let Some(events_area) = horizontal.first().copied() else {
        return;
    };
    let Some(detail_area) = horizontal.get(1).copied() else {
        return;
    };

    let status = status_text(model);
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "Forge",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" TUI "),
        Span::styled(status, Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL).title("status"));
    frame.render_widget(header, header_area);

    let events = Paragraph::new(render_event_lines(model))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("events"));
    frame.render_widget(events, events_area);

    let detail = Paragraph::new(render_detail_lines(model))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("detail/output"),
        );
    frame.render_widget(detail, detail_area);

    let footer = if let Some(input) = input {
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Input ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(input.to_string()),
            ]),
            Line::from(vec![
                Span::styled(
                    "Enter",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" send  "),
                Span::styled(
                    "Ctrl+C",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" exit  "),
                Span::styled(
                    "--tui",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" interactive"),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("input"))
    } else {
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Ctrl+C",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" interrupt  "),
            Span::styled(
                "--tui",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" opt-in preview"),
        ]))
        .block(Block::default().borders(Borders::ALL).title("help"))
    };
    frame.render_widget(footer, footer_area);
}

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

fn status_text(model: &UiModel) -> String {
    let total = model.blocks.len();
    let running = model
        .blocks
        .iter()
        .fold(0usize, |active, block| match block {
            UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Started => {
                active.saturating_add(1)
            }
            UiBlock::ToolStatus(status)
                if matches!(status.phase, UiToolPhase::Finished | UiToolPhase::Failed) =>
            {
                active.saturating_sub(1)
            }
            _ => active,
        });
    let failed = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Failed))
        .count();
    let turn_phase = model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::TurnStatus(status) => Some(status.phase.clone()),
        UiBlock::Markdown { .. }
        | UiBlock::Reasoning(_)
        | UiBlock::ToolInput(_)
        | UiBlock::ToolOutput(_)
        | UiBlock::ToolStatus(_)
        | UiBlock::ToolDetail(_)
        | UiBlock::Retry { .. }
        | UiBlock::Completion
        | UiBlock::Interrupt(_) => Some(UiTurnPhase::Running),
        UiBlock::UserMessage(_) => None,
    });
    let completed = model
        .blocks
        .iter()
        .any(|block| matches!(block, UiBlock::Completion));
    let state = if completed {
        "complete"
    } else if running > 0 || turn_phase == Some(UiTurnPhase::Running) {
        "running"
    } else if turn_phase == Some(UiTurnPhase::Pending) {
        "pending"
    } else {
        "streaming"
    };
    format!("events={total} tools_running={running} failed={failed} state={state}")
}

fn render_event_lines(model: &UiModel) -> Vec<Line<'static>> {
    if model.is_empty() {
        return vec![Line::from(Span::styled(
            "no events",
            Style::default().fg(Color::DarkGray),
        ))];
    }

    model
        .blocks
        .iter()
        .filter(|block| !matches!(block, UiBlock::ToolDetail(_)))
        .map(render_block)
        .collect()
}

fn render_detail_lines(model: &UiModel) -> Vec<Line<'static>> {
    let mut details: Vec<Line<'static>> = model
        .blocks
        .iter()
        .filter_map(|block| match block {
            UiBlock::ToolDetail(detail) => Some(render_tool_detail(detail)),
            UiBlock::ToolOutput(text) => Some(tagged_line("output", text, Color::Green)),
            _ => None,
        })
        .collect();

    if details.is_empty() {
        details.push(Line::from(Span::styled(
            "select a tool event for details",
            Style::default().fg(Color::DarkGray),
        )));
    }
    details
}

fn render_block(block: &UiBlock) -> Line<'static> {
    match block {
        UiBlock::UserMessage(text) => tagged_line("user", text, Color::Green),
        UiBlock::TurnStatus(status) => {
            let color = match status.phase {
                UiTurnPhase::Pending => Color::Yellow,
                UiTurnPhase::Running => Color::Cyan,
            };
            tagged_line("turn", &status.display_text(), color)
        }
        UiBlock::Markdown { text, partial } => {
            let marker = if *partial { "markdown~" } else { "markdown" };
            tagged_line(marker, text, Color::White)
        }
        UiBlock::Reasoning(text) => tagged_line("reason", text, Color::Blue),
        UiBlock::ToolInput(title) => tagged_line("tool", &title.display_text(), Color::Yellow),
        UiBlock::ToolOutput(text) => tagged_line("output", text, Color::Green),
        UiBlock::ToolStatus(status) => {
            let color = match status.phase {
                UiToolPhase::Started => Color::Yellow,
                UiToolPhase::Finished => Color::Green,
                UiToolPhase::Failed => Color::Red,
            };
            tagged_line("tool", &status.display_text(), color)
        }
        UiBlock::ToolDetail(detail) => render_tool_detail(detail),
        UiBlock::Retry { cause, delay } => tagged_line(
            "retry",
            &format!("{} {cause}", delay.display_text()),
            Color::Magenta,
        ),
        UiBlock::Completion => tagged_line("done", "complete", Color::Green),
        UiBlock::Interrupt(reason) => tagged_line("interrupt", reason, Color::Red),
    }
}

fn render_tool_detail(detail: &UiToolDetail) -> Line<'static> {
    tagged_line(
        "detail",
        &detail.display_text(),
        if detail.is_error {
            Color::Red
        } else {
            Color::Cyan
        },
    )
}

fn tagged_line(tag: &'static str, text: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("[{tag}] "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(text.to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use forge_ui_model::{
        UiBlock, UiModel, UiRetryDelay, UiToolDetail, UiToolPhase, UiToolStatus, UiTurnStatus,
        submitted_user_turn,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_ratatui_dashboard_renders_submitted_user_and_pending_turn() {
        let fixture = submitted_user_turn("Hello from TUI");

        let actual = render_dashboard_to_string(&fixture, 80, 10);

        assert!(actual.contains("events=2"));
        assert!(actual.contains("state=pending"));
        assert!(actual.contains("[user] Hello from TUI"));
        assert!(actual.contains("[turn] turn pending: waiting"));
    }

    #[test]
    fn test_ratatui_dashboard_renders_running_turn_state() {
        let fixture = UiModel::new(vec![UiBlock::TurnStatus(UiTurnStatus::running())]);

        let actual = render_dashboard_to_string(&fixture, 80, 9);

        assert!(actual.contains("state=running"));
        assert!(actual.contains("[turn] turn running: provider"));
    }

    #[test]
    fn test_ratatui_dashboard_renders_deterministic_model_output() {
        let fixture = UiModel::new(vec![
            UiBlock::Markdown { text: "Hello markdown".to_string(), partial: false },
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("exit 0".to_string()),
            }),
            UiBlock::ToolDetail(UiToolDetail {
                call_id: Some("call-1".to_string()),
                name: "shell".to_string(),
                arguments: Some("{\"command\":\"true\"}".to_string()),
                output: Some("exit 0".to_string()),
                is_error: false,
            }),
        ]);

        let actual = render_dashboard_to_string(&fixture, 80, 10);
        let expected = render_dashboard_to_string(&fixture, 80, 10);

        assert_eq!(actual, expected);
        assert!(actual.contains("Forge TUI"));
        assert!(actual.contains("events=3"));
        assert!(actual.contains("[markdown] Hello markdown"));
        assert!(actual.contains("[tool] shell finished: exit 0"));
        assert!(actual.contains("call_id=call-1"));
        assert!(actual.contains("args={\"command\":\"true\"}"));
    }

    #[test]
    fn test_ratatui_dashboard_renders_retry_delay_through_ui_retry_delay() {
        let fixture = UiModel::new(vec![UiBlock::Retry {
            cause: "network".to_string(),
            delay: UiRetryDelay::from_duration(Duration::from_millis(250)),
        }]);

        let actual = render_dashboard_to_string(&fixture, 64, 9);
        let expected = render_dashboard_to_string(&fixture, 64, 9);

        assert_eq!(actual, expected);
        assert!(actual.contains("Forge TUI"));
        assert!(actual.contains("events=1"));
        assert!(actual.contains("[retry] 250ms network"));
        assert!(actual.contains("select a tool event for"));
    }

    #[test]
    fn test_ratatui_dashboard_clears_running_tool_after_finish() {
        let fixture = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Started,
                summary: None,
            }),
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("exit 0".to_string()),
            }),
        ]);

        let actual = render_dashboard_to_string(&fixture, 80, 9);

        assert!(actual.contains("tools_running=0"));
    }

    #[test]
    fn test_interactive_shell_initial_frame_is_visibly_tui() {
        let actual = render_interactive_shell_to_string(80, 12);

        assert!(actual.contains("Forge TUI"));
        assert!(actual.contains("events"));
        assert!(actual.contains("detail/output"));
        assert!(actual.contains("Input"));
        assert!(actual.contains("Enter send"));
        assert!(actual.contains("--tui interactive"));
    }

    #[test]
    fn test_tui_session_queues_and_renders_frames() {
        let output = Vec::new();
        let mut fixture = TuiSession::new(output, 80, 9);
        let setup = UiModel::new(vec![UiBlock::Completion]);

        fixture
            .queue_and_render(setup)
            .expect("expected TUI session render to write to the in-memory buffer");
        let actual = String::from_utf8(fixture.into_output())
            .expect("expected rendered TUI frame to be valid UTF-8");

        assert!(actual.contains("Forge TUI"));
        assert!(actual.contains("state=complete"));
        assert!(actual.contains("[done] complete"));
    }
}
