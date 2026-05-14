//! Minimal non-interactive Ratatui renderer for Forge typed UI models.
//!
//! This crate is presentation-only. It depends on `ratatui` and the typed
//! `forge_ui_model` boundary, but it does not own runtime terminal activation.

use std::convert::Infallible;

use forge_ui_model::{UiBlock, UiModel, UiToolPhase};
use ratatui::{
    backend::TestBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

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
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        let Some(header_area) = chunks.first().copied() else {
            return;
        };
        let Some(body_area) = chunks.get(1).copied() else {
            return;
        };

        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                "Forge",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" typed TUI preview"),
        ]))
        .block(Block::default().borders(Borders::ALL).title("status"));
        frame.render_widget(header, header_area);

        let body = Paragraph::new(render_lines(model))
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title("events"));
        frame.render_widget(body, body_area);
    }));

    terminal.backend().to_string()
}

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

fn render_lines(model: &UiModel) -> Vec<Line<'static>> {
    if model.is_empty() {
        return vec![Line::from(Span::styled(
            "no events",
            Style::default().fg(Color::DarkGray),
        ))];
    }

    model.blocks.iter().map(render_block).collect()
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
        UiBlock::Retry { cause, delay } => tagged_line(
            "retry",
            &format!("{} {cause}", delay.display_text()),
            Color::Magenta,
        ),
        UiBlock::Completion => tagged_line("done", "complete", Color::Green),
        UiBlock::Interrupt(reason) => tagged_line("interrupt", reason, Color::Red),
    }
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

    use forge_ui_model::{UiBlock, UiModel, UiRetryDelay, UiToolPhase, UiToolStatus};
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
        ]);

        let actual = render_dashboard_to_string(&fixture, 48, 8);

        let expected = concat!(
            "\"┌status────────────────────────────────────────┐\"\n",
            "\"│Forge typed TUI preview                       │\"\n",
            "\"└──────────────────────────────────────────────┘\"\n",
            "\"┌events────────────────────────────────────────┐\"\n",
            "\"│[markdown] Hello markdown                     │\"\n",
            "\"│[tool] shell finished: exit 0                 │\"\n",
            "\"│                                              │\"\n",
            "\"└──────────────────────────────────────────────┘\"\n",
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_ratatui_dashboard_renders_retry_delay_through_ui_retry_delay() {
        let fixture = UiModel::new(vec![UiBlock::Retry {
            cause: "network".to_string(),
            delay: UiRetryDelay::from_duration(Duration::from_millis(250)),
        }]);

        let actual = render_dashboard_to_string(&fixture, 48, 6);

        let expected = concat!(
            "\"┌status────────────────────────────────────────┐\"\n",
            "\"│Forge typed TUI preview                       │\"\n",
            "\"└──────────────────────────────────────────────┘\"\n",
            "\"┌events────────────────────────────────────────┐\"\n",
            "\"│[retry] 250ms network                         │\"\n",
            "\"└──────────────────────────────────────────────┘\"\n",
        );
        assert_eq!(actual, expected);
    }
}
