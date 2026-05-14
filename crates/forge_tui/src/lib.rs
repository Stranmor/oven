//! Ratatui renderer and lightweight terminal session for Forge typed UI models.
//!
//! This crate is presentation-only. It depends on `ratatui`, `crossterm`, and
//! the typed `forge_ui_model` boundary. Runtime orchestration remains in
//! `forge_main`.

use std::convert::Infallible;
use std::io::{self, Write};

use crossterm::cursor::MoveTo;
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType};
use forge_ui_model::{UiBlock, UiModel, UiToolDetail, UiToolPhase};
use ratatui::backend::TestBackend;
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
    let backend = TestBackend::new(width, height);
    let mut terminal = infallible(ratatui::Terminal::new(backend));
    infallible(terminal.draw(|frame| {
        let area = frame.area();
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
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

        let footer = Paragraph::new(Line::from(vec![
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
        .block(Block::default().borders(Borders::ALL).title("help"));
        frame.render_widget(footer, footer_area);
    }));

    terminal.backend().to_string()
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
        .filter(|block| matches!(block, UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Started))
        .count();
    let failed = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Failed))
        .count();
    let completed = model
        .blocks
        .iter()
        .any(|block| matches!(block, UiBlock::Completion));
    let state = if completed {
        "complete"
    } else if running > 0 {
        "running"
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

    use forge_ui_model::{UiBlock, UiModel, UiRetryDelay, UiToolDetail, UiToolPhase, UiToolStatus};
    use pretty_assertions::assert_eq;

    use super::*;

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
