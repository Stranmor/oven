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
            render_conversation_to_string(&self.model, self.width, self.height)
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
            draw_conversation(frame, &self.model, Some(self.input.as_str()));
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
pub fn render_conversation_to_string(model: &UiModel, width: u16, height: u16) -> String {
    render_model_to_string(model, width, height, None)
}

/// Renders a typed UI model into a deterministic `ratatui` test backend string.
///
/// # Arguments
/// * `model` - The typed UI model to render.
/// * `width` - Test backend width in terminal cells.
/// * `height` - Test backend height in terminal cells.
pub fn render_dashboard_to_string(model: &UiModel, width: u16, height: u16) -> String {
    render_conversation_to_string(model, width, height)
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
        draw_conversation(frame, model, input);
    }));

    terminal.backend().to_string()
}

fn draw_conversation(frame: &mut ratatui::Frame<'_>, model: &UiModel, input: Option<&str>) {
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
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(body_area);

    let Some(transcript_area) = horizontal.first().copied() else {
        return;
    };
    let Some(detail_area) = horizontal.get(1).copied() else {
        return;
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "Forge",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" conversation  "),
        Span::styled(status_text(model), status_style(model)),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Session"));
    frame.render_widget(header, header_area);

    let transcript_lines = viewport_lines_latest(
        render_transcript_lines(model),
        transcript_area.height.saturating_sub(2),
    );
    let transcript = Paragraph::new(transcript_lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("Conversation"));
    frame.render_widget(transcript, transcript_area);

    let detail_lines = viewport_lines_latest(
        render_tool_detail_lines(model),
        detail_area.height.saturating_sub(2),
    );
    let detail = Paragraph::new(detail_lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title("Tool details"));
    frame.render_widget(detail, detail_area);

    let footer = if let Some(input) = input {
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Message ",
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
                Span::raw(" opt-in conversation UI"),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("Compose"))
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
            Span::raw(" opt-in conversation view"),
        ]))
        .block(Block::default().borders(Borders::ALL).title("Shortcuts"))
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
    let running = running_tool_count(model);
    let failed = model.blocks.iter().any(
        |block| matches!(block, UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Failed),
    );
    let completed = model
        .blocks
        .iter()
        .any(|block| matches!(block, UiBlock::Completion));
    let turn_phase = latest_turn_phase(model);

    if failed {
        "Attention - tool reported an error".to_string()
    } else if completed {
        "Complete - response finished".to_string()
    } else if running > 0 {
        "Working - tool activity in progress".to_string()
    } else if turn_phase == Some(UiTurnPhase::Running) {
        "Working - assistant is responding".to_string()
    } else if turn_phase == Some(UiTurnPhase::Pending) {
        "Waiting - preparing response".to_string()
    } else if latest_tool_phase(model).is_some() {
        "Ready - tool activity complete".to_string()
    } else {
        "Ready - start a conversation".to_string()
    }
}

fn status_style(model: &UiModel) -> Style {
    let failed = model.blocks.iter().any(
        |block| matches!(block, UiBlock::ToolStatus(status) if status.phase == UiToolPhase::Failed),
    );
    if failed {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if model
        .blocks
        .iter()
        .any(|block| matches!(block, UiBlock::Completion))
    {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else if running_tool_count(model) > 0
        || latest_turn_phase(model) == Some(UiTurnPhase::Running)
    {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else if latest_turn_phase(model) == Some(UiTurnPhase::Pending) {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else if latest_tool_phase(model).is_some() {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn running_tool_count(model: &UiModel) -> usize {
    model
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
        })
}

fn latest_tool_phase(model: &UiModel) -> Option<UiToolPhase> {
    model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::ToolStatus(status) => Some(status.phase.clone()),
        _ => None,
    })
}

fn latest_turn_phase(model: &UiModel) -> Option<UiTurnPhase> {
    model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::TurnStatus(status) => Some(status.phase.clone()),
        UiBlock::Markdown { partial, .. } if *partial => Some(UiTurnPhase::Running),
        UiBlock::Reasoning(_) => Some(UiTurnPhase::Running),
        UiBlock::Markdown { .. }
        | UiBlock::ToolInput(_)
        | UiBlock::ToolOutput(_)
        | UiBlock::ToolStatus(_)
        | UiBlock::ToolDetail(_)
        | UiBlock::Retry { .. }
        | UiBlock::Interrupt(_)
        | UiBlock::Completion
        | UiBlock::UserMessage(_) => None,
    })
}

fn viewport_lines_latest(lines: Vec<Line<'static>>, visible_height: u16) -> Vec<Line<'static>> {
    let visible_height = usize::from(visible_height);
    if visible_height == 0 || lines.len() <= visible_height {
        return lines;
    }

    if visible_height == 1 {
        return vec![viewport_marker(lines.len())];
    }

    let visible_tail = visible_height.saturating_sub(1);
    let hidden_count = lines.len().saturating_sub(visible_tail);
    let mut visible = vec![viewport_marker(hidden_count)];
    visible.extend(lines.into_iter().skip(hidden_count));
    visible
}

fn viewport_marker(hidden_count: usize) -> Line<'static> {
    Line::from(Span::styled(
        format!("↑ {hidden_count} hidden"),
        Style::default().fg(Color::DarkGray),
    ))
}

fn render_transcript_lines(model: &UiModel) -> Vec<Line<'static>> {
    if model.is_empty() {
        return vec![
            Line::from(Span::styled(
                "Start a conversation. Assistant replies, tool activity, and status updates appear here.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "Tool cards stay compact here; the side pane keeps arguments, output, and errors discoverable.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
    }

    let mut lines = Vec::new();
    for block in model
        .blocks
        .iter()
        .filter(|block| !matches!(block, UiBlock::ToolDetail(_)))
    {
        lines.extend(render_transcript_block(block));
    }
    lines
}

fn render_transcript_block(block: &UiBlock) -> Vec<Line<'static>> {
    match block {
        UiBlock::UserMessage(text) => actor_block("You", text, Color::Green),
        UiBlock::TurnStatus(status) => vec![status_line(
            "Assistant",
            &status.display_text().replace("turn ", ""),
            match status.phase {
                UiTurnPhase::Pending => Color::Yellow,
                UiTurnPhase::Running => Color::Cyan,
            },
        )],
        UiBlock::Markdown { text, partial } => markdown_block(text, *partial),
        UiBlock::Reasoning(text) => actor_block("Reasoning", text, Color::Blue),
        UiBlock::ToolInput(title) => vec![status_line(
            "Tool request",
            &title.display_text(),
            Color::Yellow,
        )],
        UiBlock::ToolOutput(text) => vec![status_line(
            "Tool output",
            &preview_text(text),
            Color::Green,
        )],
        UiBlock::ToolStatus(status) => vec![tool_status_line(status)],
        UiBlock::ToolDetail(detail) => vec![status_line("Tool detail", &detail.name, Color::Cyan)],
        UiBlock::Retry { cause, delay } => vec![status_line(
            "Retry",
            &format!("waiting {} - {cause}", delay.display_text()),
            Color::Magenta,
        )],
        UiBlock::Completion => vec![status_line("Assistant", "complete", Color::Green)],
        UiBlock::Interrupt(reason) => vec![status_line("Interrupted", reason, Color::Red)],
    }
}

fn actor_block(label: &'static str, text: &str, color: Color) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        label,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))];
    lines.extend(render_body_lines(text, "  ", Color::White));
    lines
}

fn markdown_block(text: &str, partial: bool) -> Vec<Line<'static>> {
    let label = if partial {
        "Assistant - streaming"
    } else {
        "Assistant"
    };
    let mut lines = vec![Line::from(Span::styled(
        label,
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))];
    lines.extend(render_markdown_lines(text));
    lines
}

fn render_markdown_lines(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code = false;

    for raw_line in text.lines() {
        let trimmed = raw_line.trim_end();
        if trimmed.trim_start().starts_with("```") {
            in_code = !in_code;
            let label = if in_code { "  code" } else { "  end code" };
            lines.push(Line::from(Span::styled(
                label,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        if in_code {
            lines.push(Line::from(vec![
                Span::styled("  | ", Style::default().fg(Color::DarkGray)),
                Span::styled(trimmed.to_string(), Style::default().fg(Color::Yellow)),
            ]));
            continue;
        }

        let cleaned = trimmed.trim_start_matches('#').trim_start();
        if cleaned.is_empty() {
            lines.push(Line::from(""));
        } else if trimmed.starts_with('#') {
            lines.push(Line::from(Span::styled(
                format!("  {cleaned}"),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(item) = cleaned
            .strip_prefix("- ")
            .or_else(|| cleaned.strip_prefix("* "))
        {
            lines.push(Line::from(vec![
                Span::styled("  - ", Style::default().fg(Color::Cyan)),
                Span::raw(item.to_string()),
            ]));
        } else {
            lines.extend(render_body_lines(cleaned, "  ", Color::White));
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Assistant response is streaming...",
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

fn render_body_lines(text: &str, prefix: &'static str, color: Color) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![Line::from(Span::styled(
            prefix,
            Style::default().fg(Color::DarkGray),
        ))];
    }

    text.lines()
        .map(|line| {
            Line::from(vec![
                Span::styled(prefix, Style::default().fg(Color::DarkGray)),
                Span::styled(line.to_string(), Style::default().fg(color)),
            ])
        })
        .collect()
}

fn status_line(label: &'static str, text: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(text.to_string()),
    ])
}

fn tool_status_line(status: &forge_ui_model::UiToolStatus) -> Line<'static> {
    let (phase, color) = match status.phase {
        UiToolPhase::Started => ("running", Color::Yellow),
        UiToolPhase::Finished => ("done", Color::Green),
        UiToolPhase::Failed => ("failed", Color::Red),
    };
    let summary = status
        .summary
        .as_deref()
        .map(preview_text)
        .unwrap_or_default();
    let detail_hint = if summary.is_empty() {
        "details in side pane".to_string()
    } else {
        format!("{summary} - details in side pane")
    };

    Line::from(vec![
        Span::styled(
            "Tool ",
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(status.name.clone(), Style::default().fg(Color::White)),
        Span::raw(" "),
        Span::styled(phase, Style::default().fg(color)),
        Span::raw(" - "),
        Span::styled(detail_hint, Style::default().fg(Color::DarkGray)),
    ])
}

fn render_tool_detail_lines(model: &UiModel) -> Vec<Line<'static>> {
    if let Some(detail) = model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::ToolDetail(detail) => Some(detail),
        _ => None,
    }) {
        return render_tool_detail(detail);
    }

    if let Some(output) = model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::ToolOutput(output) => Some(output),
        _ => None,
    }) {
        let mut lines = vec![section_header("Latest tool output", Color::Green)];
        lines.extend(render_body_lines(output, "  ", Color::White));
        return lines;
    }

    vec![
        section_header("Tool activity", Color::Cyan),
        Line::from(Span::styled(
            "No tool activity yet.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "Arguments, output, and errors appear here as soon as tools run.",
            Style::default().fg(Color::DarkGray),
        )),
    ]
}

fn render_tool_detail(detail: &UiToolDetail) -> Vec<Line<'static>> {
    let mut lines = vec![section_header(
        if detail.is_error {
            "Latest tool error"
        } else {
            "Latest tool activity"
        },
        if detail.is_error {
            Color::Red
        } else {
            Color::Cyan
        },
    )];
    lines.push(Line::from(vec![
        Span::styled("Tool ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            detail.name.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    if let Some(call_id) = &detail.call_id {
        lines.push(Line::from(vec![
            Span::styled("Call ", Style::default().fg(Color::DarkGray)),
            Span::raw(call_id.clone()),
        ]));
    }

    if let Some(arguments) = &detail.arguments {
        lines.push(section_header("Arguments", Color::Yellow));
        lines.extend(render_body_lines(arguments, "  ", Color::White));
    }

    if let Some(output) = &detail.output {
        lines.push(section_header(
            if detail.is_error { "Error" } else { "Output" },
            if detail.is_error {
                Color::Red
            } else {
                Color::Green
            },
        ));
        lines.extend(render_body_lines(output, "  ", Color::White));
    }

    if detail.arguments.is_none() && detail.output.is_none() {
        lines.push(Line::from(Span::styled(
            "Waiting for tool payload...",
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines
}

fn section_header(label: &'static str, color: Color) -> Line<'static> {
    Line::from(Span::styled(
        label,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))
}

fn preview_text(text: &str) -> String {
    const LIMIT: usize = 96;
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= LIMIT {
        return normalized;
    }

    let mut preview = normalized.chars().take(LIMIT).collect::<String>();
    preview.push_str("...");
    preview
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use forge_ui_model::{
        UiBlock, UiModel, UiRetryDelay, UiToolDetail, UiToolPhase, UiToolStatus, UiTurnStatus,
        submitted_user_turn,
    };

    #[test]
    fn test_conversation_layout_renders_submitted_user_and_pending_turn() {
        let fixture = submitted_user_turn("Hello from TUI");

        let actual = render_conversation_to_string(&fixture, 92, 12);

        assert!(actual.contains("Session"));
        assert!(actual.contains("Conversation"));
        assert!(actual.contains("Tool details"));
        assert!(actual.contains("Waiting - preparing response"));
        assert!(actual.contains("You"));
        assert!(actual.contains("Hello from TUI"));
        assert!(actual.contains("Assistant"));
        assert!(actual.contains("pending: waiting for provider response"));
        assert!(!actual.contains("events="));
        assert!(!actual.contains("state="));
    }

    #[test]
    fn test_conversation_layout_renders_running_turn_state() {
        let fixture = UiModel::new(vec![UiBlock::TurnStatus(UiTurnStatus::running())]);

        let actual = render_conversation_to_string(&fixture, 90, 10);

        assert!(actual.contains("Working - assistant is responding"));
        assert!(actual.contains("Assistant"));
        assert!(actual.contains("running: provider stream running"));
    }

    #[test]
    fn test_conversation_layout_renders_markdown_without_debug_tag() {
        let fixture = UiModel::new(vec![UiBlock::Markdown {
            text: "# Plan\n- inspect\n```rust\nlet ok = true;\n```".to_string(),
            partial: false,
        }]);

        let actual = render_conversation_to_string(&fixture, 100, 14);
        let expected = vec![
            "Assistant",
            "Plan",
            "- inspect",
            "code",
            "let ok = true;",
            "end code",
        ];

        for fragment in expected {
            assert!(
                actual.contains(fragment),
                "missing rendered fragment: {fragment}"
            );
        }
        assert!(!actual.contains("[markdown]"));
        assert!(!actual.contains("markdown~"));
    }

    #[test]
    fn test_conversation_layout_renders_tool_cards_and_structured_detail() {
        let fixture = UiModel::new(vec![
            UiBlock::Markdown { text: "Checking project".to_string(), partial: false },
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

        let actual = render_conversation_to_string(&fixture, 104, 16);
        let expected = vec![
            "Assistant",
            "Checking project",
            "Tool shell done",
            "details in side pane",
            "Latest tool activity",
            "Tool shell",
            "Call call-1",
            "Arguments",
            "{\"command\":\"true\"}",
            "Output",
            "exit 0",
        ];

        for fragment in expected {
            assert!(
                actual.contains(fragment),
                "missing rendered fragment: {fragment}"
            );
        }
        assert!(!actual.contains("[tool]"));
        assert!(!actual.contains("detail/output"));
    }

    #[test]
    fn test_conversation_layout_renders_retry_without_placeholder_detail_copy() {
        let fixture = UiModel::new(vec![UiBlock::Retry {
            cause: "network".to_string(),
            delay: UiRetryDelay::from_duration(Duration::from_millis(250)),
        }]);

        let actual = render_conversation_to_string(&fixture, 78, 11);
        let expected = vec![
            "Forge conversation",
            "Retry",
            "waiting 250ms - network",
            "Tool activity",
            "No tool activity yet.",
        ];

        for fragment in expected {
            assert!(
                actual.contains(fragment),
                "missing rendered fragment: {fragment}"
            );
        }
        assert!(!actual.contains("select a tool event"));
    }

    #[test]
    fn test_conversation_layout_clears_running_tool_after_finish() {
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

        let actual = render_conversation_to_string(&fixture, 90, 10);

        assert!(actual.contains("Ready - tool activity complete"));
        assert!(!actual.contains("Working - assistant is responding"));
        assert!(!actual.contains("tools_running="));
    }

    #[test]
    fn test_conversation_layout_keeps_latest_transcript_visible_when_content_overflows() {
        let mut setup = Vec::new();
        for index in 0..24 {
            setup.push(UiBlock::Markdown {
                text: format!("older assistant line {index}"),
                partial: false,
            });
        }
        setup.push(UiBlock::Markdown {
            text: "NEWEST ASSISTANT ANSWER".to_string(),
            partial: false,
        });
        setup.push(UiBlock::ToolStatus(UiToolStatus {
            name: "shell".to_string(),
            phase: UiToolPhase::Finished,
            summary: Some("LATEST".to_string()),
        }));
        let fixture = UiModel::new(setup);

        let actual = render_conversation_to_string(&fixture, 100, 12);

        assert!(actual.contains("hidden"));
        assert!(actual.contains("NEWEST ASSISTANT ANSWER"));
        assert!(actual.contains("LATEST"));
        assert!(!actual.contains("older assistant line 0"));
    }

    #[test]
    fn test_tool_detail_layout_keeps_latest_output_visible_when_detail_overflows() {
        let mut long_output = (0..30)
            .map(|index| format!("older output line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        long_output.push_str("\nZENDMARK");
        let fixture = UiModel::new(vec![UiBlock::ToolDetail(UiToolDetail {
            call_id: Some("call-overflow".to_string()),
            name: "shell".to_string(),
            arguments: Some("{\"command\":\"long-output\"}".to_string()),
            output: Some(long_output),
            is_error: false,
        })]);

        let actual = render_conversation_to_string(&fixture, 110, 12);

        assert!(actual.contains("hidden"));
        assert!(actual.contains("ZENDMARK"));
        assert!(!actual.contains("older output line 0"));
    }

    #[test]
    fn test_interactive_shell_initial_frame_is_meaningful_conversation_ui() {
        let actual = render_interactive_shell_to_string(92, 14);

        assert!(actual.contains("Forge conversation"));
        assert!(actual.contains("Ready - start a conversation"));
        assert!(actual.contains("Conversation"));
        assert!(actual.contains("Start a conversation"));
        assert!(actual.contains("Tool details"));
        assert!(actual.contains("No tool activity yet."));
        assert!(actual.contains("Message"));
        assert!(actual.contains("Enter send"));
        assert!(actual.contains("--tui opt-in conversation UI"));
        assert!(!actual.contains("events"));
        assert!(!actual.contains("detail/output"));
    }

    #[test]
    fn test_tui_session_queues_and_renders_nonblank_post_submit_frame() {
        let output = Vec::new();
        let mut fixture = TuiSession::new(output, 92, 10);
        let setup = UiModel::new(vec![UiBlock::Completion]);

        fixture
            .queue_and_render(setup)
            .expect("expected TUI session render to write to the in-memory buffer");
        let actual = String::from_utf8(fixture.into_output())
            .expect("expected rendered TUI frame to be valid UTF-8");

        assert!(actual.contains("Forge conversation"));
        assert!(actual.contains("Complete - response finished"));
        assert!(actual.contains("Assistant"));
        assert!(actual.contains("complete"));
        assert!(!actual.trim().is_empty());
    }
}
