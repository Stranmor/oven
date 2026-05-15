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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

const PAGE_SCROLL_LINES: u16 = 6;
const DETAIL_TRUNCATION_LINES: usize = 18;

/// Owns a live TUI render session and its append-only typed model.
pub struct TuiSession<W: Write> {
    model: UiModel,
    output: W,
    width: u16,
    height: u16,
    view_state: TuiViewState,
}

impl<W: Write> TuiSession<W> {
    /// Creates a session with an explicit render area.
    ///
    /// # Arguments
    /// * `output` - Writable terminal or test buffer receiving rendered frames.
    /// * `width` - Render width in terminal cells.
    /// * `height` - Render height in terminal cells.
    pub fn new(output: W, width: u16, height: u16) -> Self {
        Self {
            model: UiModel::default(),
            output,
            width,
            height,
            view_state: TuiViewState::default(),
        }
    }

    /// Appends a typed response model and renders the next frame.
    ///
    /// # Arguments
    /// * `event_model` - Non-empty typed model produced from one chat response.
    ///
    /// # Errors
    /// Returns an error if writing the rendered frame to the terminal fails.
    pub fn queue_and_render(&mut self, event_model: UiModel) -> io::Result<()> {
        append_model(&mut self.model, event_model, &mut self.view_state);
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
            render_model_to_string_with_state(
                &self.model,
                self.width,
                self.height,
                None,
                &self.view_state
            )
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

/// Keyboard focus target owned by the TUI renderer view state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusPane {
    /// The transcript pane receives scroll commands.
    Transcript,
    /// The tool activity rail receives scroll commands.
    ToolActivity,
    /// The selected/latest tool detail pane receives scroll commands.
    ToolDetail,
    /// The compose input receives text editing commands.
    #[default]
    Input,
}

/// Pure TUI-owned viewport state that does not belong to `forge_ui_model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiViewState {
    /// Current keyboard focus pane.
    pub focus: FocusPane,
    /// Top-line scroll offset for transcript when follow-latest is disabled.
    pub transcript_scroll: u16,
    /// Top-line scroll offset for tool activity when follow-latest is disabled.
    pub activity_scroll: u16,
    /// Top-line scroll offset for tool detail when follow-latest is disabled.
    pub detail_scroll: u16,
    /// Whether transcript and details follow the latest content.
    pub follow_latest: bool,
}

impl Default for TuiViewState {
    fn default() -> Self {
        Self {
            focus: FocusPane::Input,
            transcript_scroll: 0,
            activity_scroll: 0,
            detail_scroll: 0,
            follow_latest: true,
        }
    }
}

/// Pure result of applying a keyboard event to TUI view/input state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiKeyAction {
    /// The key was consumed by navigation or editing state.
    Consumed,
    /// A non-empty line was submitted.
    Submitted(String),
    /// The session should exit.
    Exit,
    /// The key was not handled by the pure transition helper.
    Ignored,
}

/// Applies one keyboard event to owned TUI state without terminal side effects.
///
/// # Arguments
/// * `state` - Mutable TUI view state.
/// * `input` - Mutable compose input buffer.
/// * `event` - Crossterm keyboard event to interpret.
pub fn apply_tui_key_event(
    state: &mut TuiViewState,
    input: &mut String,
    event: KeyEvent,
) -> TuiKeyAction {
    match event {
        KeyEvent { code: KeyCode::Char('c'), modifiers, .. }
            if modifiers.contains(KeyModifiers::CONTROL) =>
        {
            TuiKeyAction::Exit
        }
        KeyEvent { code: KeyCode::Char('d'), modifiers, .. }
            if modifiers.contains(KeyModifiers::CONTROL) && input.is_empty() =>
        {
            TuiKeyAction::Exit
        }
        KeyEvent { code: KeyCode::Tab, .. } => {
            state.focus = next_focus(state.focus);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::BackTab, .. } => {
            state.focus = previous_focus(state.focus);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::PageUp, .. } => {
            scroll_focused_pane(state, PAGE_SCROLL_LINES, ScrollDirection::Up);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::PageDown, .. } => {
            scroll_focused_pane(state, PAGE_SCROLL_LINES, ScrollDirection::Down);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::Home, .. } => {
            jump_focused_pane(state, ScrollJump::Start);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::End, .. } => {
            jump_focused_pane(state, ScrollJump::End);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::Enter, .. } if matches!(state.focus, FocusPane::Input) => {
            let submitted = input.trim().to_string();
            input.clear();
            if submitted.is_empty() {
                TuiKeyAction::Consumed
            } else {
                TuiKeyAction::Submitted(submitted)
            }
        }
        KeyEvent { code: KeyCode::Backspace, .. } if matches!(state.focus, FocusPane::Input) => {
            input.pop();
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::Char(value), modifiers, .. }
            if matches!(state.focus, FocusPane::Input)
                && !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            input.push(value);
            TuiKeyAction::Consumed
        }
        KeyEvent { code: KeyCode::Char(value), modifiers, .. }
            if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            state.focus = FocusPane::Input;
            input.push(value);
            TuiKeyAction::Consumed
        }
        _ => TuiKeyAction::Ignored,
    }
}

/// Owns an alternate-screen interactive terminal session.
pub struct InteractiveTuiSession {
    model: UiModel,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    input: String,
    view_state: TuiViewState,
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
            view_state: TuiViewState::default(),
            suspended_for_stdout: false,
        })
    }

    /// Renders the current interactive shell frame.
    ///
    /// # Errors
    /// Returns an error if drawing to the terminal fails.
    pub fn render(&mut self) -> io::Result<()> {
        self.terminal.draw(|frame| {
            draw_conversation(
                frame,
                &self.model,
                Some(self.input.as_str()),
                &self.view_state,
            );
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
                Event::Key(key_event) => {
                    match apply_tui_key_event(&mut self.view_state, &mut self.input, key_event) {
                        TuiKeyAction::Submitted(submitted) => {
                            return Ok(TuiInput::Submitted(submitted));
                        }
                        TuiKeyAction::Exit => return Ok(TuiInput::Exit),
                        TuiKeyAction::Consumed => self.render()?,
                        TuiKeyAction::Ignored => {}
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
        append_model(&mut self.model, event_model, &mut self.view_state);
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

/// Renders a typed UI model with explicit TUI-owned view state.
///
/// # Arguments
/// * `model` - The typed UI model to render.
/// * `width` - Test backend width in terminal cells.
/// * `height` - Test backend height in terminal cells.
/// * `input` - Optional compose input payload.
/// * `state` - TUI-owned focus and scroll state.
pub fn render_model_to_string_with_state(
    model: &UiModel,
    width: u16,
    height: u16,
    input: Option<&str>,
    state: &TuiViewState,
) -> String {
    render_model_to_string_with_state_inner(model, width, height, input, state)
}

/// Renders the initial interactive TUI shell into a deterministic string.
///
/// # Arguments
/// * `width` - Test backend width in terminal cells.
/// * `height` - Test backend height in terminal cells.
pub fn render_interactive_shell_to_string(width: u16, height: u16) -> String {
    render_model_to_string(&UiModel::default(), width, height, Some(""))
}

fn append_model(model: &mut UiModel, event_model: UiModel, view_state: &mut TuiViewState) {
    for block in event_model.blocks {
        model.push(block);
    }
    if view_state.follow_latest {
        view_state.transcript_scroll = 0;
        view_state.activity_scroll = 0;
        view_state.detail_scroll = 0;
    }
}

fn render_model_to_string(model: &UiModel, width: u16, height: u16, input: Option<&str>) -> String {
    render_model_to_string_with_state_inner(model, width, height, input, &TuiViewState::default())
}

fn render_model_to_string_with_state_inner(
    model: &UiModel,
    width: u16,
    height: u16,
    input: Option<&str>,
    state: &TuiViewState,
) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = infallible(ratatui::Terminal::new(backend));
    infallible(terminal.draw(|frame| {
        draw_conversation(frame, model, input, state);
    }));

    terminal.backend().to_string()
}

fn draw_conversation(
    frame: &mut ratatui::Frame<'_>,
    model: &UiModel,
    input: Option<&str>,
    state: &TuiViewState,
) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    if area.height <= 10 {
        draw_dense_fallback(frame, area, model, input, state);
        return;
    }

    let footer_height = if input.is_some() { 4 } else { 3 };
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

    draw_header(frame, header_area, model);

    if body_area.width >= 110 && body_area.height >= 18 {
        draw_wide_body(
            frame,
            body_area,
            model,
            state,
            [Constraint::Percentage(63), Constraint::Percentage(37)],
        );
    } else if body_area.width >= 80 && area.height >= 20 {
        draw_wide_body(
            frame,
            body_area,
            model,
            state,
            [Constraint::Percentage(65), Constraint::Percentage(35)],
        );
    } else {
        draw_single_column_body(frame, body_area, model, state);
    }

    draw_footer(frame, footer_area, input, state);
}

fn draw_dense_fallback(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    input: Option<&str>,
    state: &TuiViewState,
) {
    let available = area.height.saturating_sub(2);
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Forge ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(status_pill(model).label(), status_pill(model).style()),
        Span::raw(" "),
        Span::styled(session_summary(model), Style::default().fg(Color::DarkGray)),
    ])];
    lines.extend(render_transcript_lines(model));
    if let Some(input) = input {
        lines.push(Line::from(vec![
            Span::styled(
                "> ",
                Style::default().fg(focus_color(state, FocusPane::Input)),
            ),
            Span::raw(input.to_string()),
        ]));
    }
    let lines = viewport_lines(lines, available, 0, true);
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Forge command center"),
        ),
        area,
    );
}

fn draw_header(frame: &mut ratatui::Frame<'_>, area: Rect, model: &UiModel) {
    let pill = status_pill(model);
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "Forge",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" command center  "),
        Span::styled(format!(" {} ", pill.label()), pill.style()),
        Span::raw("  "),
        Span::styled(session_summary(model), Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, area);
}

fn draw_wide_body(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    state: &TuiViewState,
    constraints: [Constraint; 2],
) {
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    let Some(transcript_area) = horizontal.first().copied() else {
        return;
    };
    let Some(rail_area) = horizontal.get(1).copied() else {
        return;
    };

    draw_transcript(frame, transcript_area, model, state);

    let rail = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(rail_area);
    if let Some(activity_area) = rail.first().copied() {
        draw_tool_activity(frame, activity_area, model, state);
    }
    if let Some(detail_area) = rail.get(1).copied() {
        draw_tool_detail(frame, detail_area, model, state);
    }
}

fn draw_single_column_body(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    state: &TuiViewState,
) {
    let transcript_height = area.height.saturating_mul(2) / 3;
    let constraints = if area.height < 18 {
        [Constraint::Min(0), Constraint::Length(0)]
    } else {
        [
            Constraint::Length(transcript_height.max(5)),
            Constraint::Min(0),
        ]
    };
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    if let Some(transcript_area) = vertical.first().copied() {
        draw_transcript(frame, transcript_area, model, state);
    }
    if let Some(rail_area) = vertical.get(1).copied()
        && rail_area.height >= 5
    {
        draw_tool_activity(frame, rail_area, model, state);
    }
}

fn draw_transcript(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    state: &TuiViewState,
) {
    let lines = viewport_lines(
        render_transcript_lines(model),
        area.height.saturating_sub(2),
        state.transcript_scroll,
        state.follow_latest,
    );
    let title = focused_title("Transcript", state, FocusPane::Transcript);
    let transcript = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(transcript, area);
}

fn draw_tool_activity(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    state: &TuiViewState,
) {
    let lines = viewport_lines(
        render_tool_activity_lines(model),
        area.height.saturating_sub(2),
        state.activity_scroll,
        state.follow_latest,
    );
    let title = focused_title("Tool activity", state, FocusPane::ToolActivity);
    let widget = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(widget, area);
}

fn draw_tool_detail(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    model: &UiModel,
    state: &TuiViewState,
) {
    let lines = viewport_lines(
        render_tool_detail_lines(model),
        area.height.saturating_sub(2),
        state.detail_scroll,
        state.follow_latest,
    );
    let title = focused_title("Tool detail", state, FocusPane::ToolDetail);
    let widget = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(widget, area);
}

fn draw_footer(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    input: Option<&str>,
    state: &TuiViewState,
) {
    let shortcuts = Line::from(vec![
        Span::styled(
            "Tab",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" focus  "),
        Span::styled(
            "PgUp/PgDn",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" scroll  "),
        Span::styled(
            "Home/End",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" jump  "),
        Span::styled(
            "Ctrl+C",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" exit"),
    ]);
    let paragraph = if let Some(input) = input {
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Message ",
                    Style::default()
                        .fg(focus_color(state, FocusPane::Input))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(input.to_string()),
            ]),
            shortcuts,
        ])
        .block(Block::default().borders(Borders::ALL).title(focused_title(
            "Compose",
            state,
            FocusPane::Input,
        )))
    } else {
        Paragraph::new(shortcuts).block(Block::default().borders(Borders::ALL).title("Shortcuts"))
    };
    frame.render_widget(paragraph, area);
}

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LatestConversationState {
    UserSubmitted,
    TurnPending,
    TurnRunning,
    ToolRunning,
    ToolFinished,
    ToolFailed,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusPill {
    Ready,
    Thinking,
    ToolRunning,
    Error,
    Complete,
}

impl StatusPill {
    fn label(self) -> &'static str {
        match self {
            StatusPill::Ready => "Ready",
            StatusPill::Thinking => "Thinking",
            StatusPill::ToolRunning => "Tool running",
            StatusPill::Error => "Error",
            StatusPill::Complete => "Complete",
        }
    }

    fn style(self) -> Style {
        match self {
            StatusPill::Ready => Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            StatusPill::Thinking => Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            StatusPill::ToolRunning => Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            StatusPill::Error => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            StatusPill::Complete => Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        }
    }
}

fn status_pill(model: &UiModel) -> StatusPill {
    match latest_conversation_state(model) {
        Some(LatestConversationState::ToolFailed) => StatusPill::Error,
        Some(LatestConversationState::Complete) => StatusPill::Complete,
        Some(LatestConversationState::ToolRunning) => StatusPill::ToolRunning,
        Some(
            LatestConversationState::TurnRunning
            | LatestConversationState::TurnPending
            | LatestConversationState::UserSubmitted,
        ) => StatusPill::Thinking,
        Some(LatestConversationState::ToolFinished) | None => StatusPill::Ready,
    }
}

fn latest_conversation_state(model: &UiModel) -> Option<LatestConversationState> {
    model.blocks.iter().rev().find_map(|block| match block {
        UiBlock::TurnStatus(status) => match status.phase {
            UiTurnPhase::Pending => Some(LatestConversationState::TurnPending),
            UiTurnPhase::Running => Some(LatestConversationState::TurnRunning),
        },
        UiBlock::Markdown { partial, .. } if *partial => Some(LatestConversationState::TurnRunning),
        UiBlock::Reasoning(_) => Some(LatestConversationState::TurnRunning),
        UiBlock::ToolStatus(status) => match status.phase {
            UiToolPhase::Started => Some(LatestConversationState::ToolRunning),
            UiToolPhase::Finished => Some(LatestConversationState::ToolFinished),
            UiToolPhase::Failed => Some(LatestConversationState::ToolFailed),
        },
        UiBlock::Completion => Some(LatestConversationState::Complete),
        UiBlock::UserMessage(_) => Some(LatestConversationState::UserSubmitted),
        UiBlock::Markdown { .. }
        | UiBlock::ToolInput(_)
        | UiBlock::ToolOutput(_)
        | UiBlock::ToolDetail(_)
        | UiBlock::Retry { .. }
        | UiBlock::Interrupt(_) => None,
    })
}

fn session_summary(model: &UiModel) -> String {
    let messages = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::UserMessage(_)))
        .count();
    let assistant = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::Markdown { .. }))
        .count();
    let tools = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::ToolStatus(_)))
        .count();
    let errors = model
        .blocks
        .iter()
        .filter(|block| matches!(block, UiBlock::ToolStatus(status) if matches!(status.phase, UiToolPhase::Failed)))
        .count();
    format!("turns {messages} · replies {assistant} · tools {tools} · errors {errors}")
}

fn viewport_lines(
    lines: Vec<Line<'static>>,
    visible_height: u16,
    scroll: u16,
    follow_latest: bool,
) -> Vec<Line<'static>> {
    let visible_height = usize::from(visible_height);
    if visible_height == 0 || lines.len() <= visible_height {
        return lines;
    }

    if visible_height == 1 {
        return vec![viewport_marker(lines.len())];
    }

    if follow_latest {
        let visible_tail = visible_height.saturating_sub(1);
        let hidden_count = lines.len().saturating_sub(visible_tail);
        let mut visible = vec![viewport_marker(hidden_count)];
        visible.extend(lines.into_iter().skip(hidden_count));
        return visible;
    }

    let max_start = lines.len().saturating_sub(visible_height);
    let start = usize::from(scroll).min(max_start);
    let end = start.saturating_add(visible_height).min(lines.len());
    let mut visible = lines
        .into_iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect::<Vec<_>>();
    if start > 0
        && let Some(first) = visible.first_mut()
    {
        *first = viewport_marker(start);
    }
    if end < start.saturating_add(visible_height) {
        return visible;
    }
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
                "Start a conversation. Assistant replies, tool cards, and status updates appear here.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "Tool payloads stay out of transcript; the rail keeps arguments, output, and errors discoverable.",
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
        UiBlock::ToolOutput(_) => vec![status_line(
            "Tool output",
            "available in rail",
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
        "Assistant streaming"
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
    let mut code_language: Option<String> = None;

    for raw_line in text.lines() {
        let trimmed_end = raw_line.trim_end();
        let trimmed_start = trimmed_end.trim_start();
        if let Some(language) = trimmed_start.strip_prefix("```") {
            if code_language.is_some() {
                lines.push(Line::from(Span::styled(
                    "  └─ end code",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                )));
                code_language = None;
            } else {
                let language = language.trim();
                let label = if language.is_empty() {
                    "code".to_string()
                } else {
                    format!("code · {language}")
                };
                lines.push(Line::from(Span::styled(
                    format!("  ┌─ {label}"),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                )));
                code_language = Some(language.to_string());
            }
            continue;
        }

        if code_language.is_some() {
            lines.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
                Span::styled(trimmed_end.to_string(), Style::default().fg(Color::Yellow)),
            ]));
            continue;
        }

        if trimmed_start.is_empty() {
            lines.push(Line::from(""));
        } else if trimmed_start.starts_with('#') {
            let cleaned = trimmed_start.trim_start_matches('#').trim_start();
            lines.push(Line::from(vec![
                Span::styled(
                    "  § ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    cleaned.to_string(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        } else if let Some(item) = trimmed_start
            .strip_prefix("- ")
            .or_else(|| trimmed_start.strip_prefix("* "))
        {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(Color::Cyan)),
                Span::raw(item.to_string()),
            ]));
        } else if let Some((number, item)) = ordered_list_item(trimmed_start) {
            lines.push(Line::from(vec![
                Span::styled(format!("  {number}. "), Style::default().fg(Color::Cyan)),
                Span::raw(item.to_string()),
            ]));
        } else if let Some(quote) = trimmed_start
            .strip_prefix("> ")
            .or_else(|| trimmed_start.strip_prefix('>'))
        {
            lines.push(Line::from(vec![
                Span::styled("  ▌ ", Style::default().fg(Color::Magenta)),
                Span::styled(
                    quote.trim_start().to_string(),
                    Style::default().fg(Color::Gray),
                ),
            ]));
        } else {
            lines.extend(render_body_lines(trimmed_start, "  ", Color::White));
        }
    }

    if code_language.is_some() {
        lines.push(Line::from(Span::styled(
            "  └─ end code",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Assistant response is streaming...",
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

fn ordered_list_item(line: &str) -> Option<(&str, &str)> {
    let dot_index = line.find('.')?;
    let (number, rest) = line.split_at(dot_index);
    if number.is_empty() || !number.chars().all(|value| value.is_ascii_digit()) {
        return None;
    }
    let item = rest.strip_prefix(". ")?;
    Some((number, item))
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
    let detail_hint = match status.phase {
        UiToolPhase::Started => "detail in rail".to_string(),
        UiToolPhase::Finished => "completed - detail in rail".to_string(),
        UiToolPhase::Failed => "failed - detail in rail".to_string(),
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

fn render_tool_activity_lines(model: &UiModel) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for block in &model.blocks {
        match block {
            UiBlock::ToolInput(title) => {
                lines.push(status_line("Request", &title.display_text(), Color::Yellow))
            }
            UiBlock::ToolOutput(output) => {
                lines.push(status_line("Output", &preview_text(output), Color::Green))
            }
            UiBlock::ToolStatus(status) => lines.push(tool_status_line(status)),
            UiBlock::Retry { cause, delay } => lines.push(status_line(
                "Retry",
                &format!("{} - {cause}", delay.display_text()),
                Color::Magenta,
            )),
            UiBlock::ToolDetail(detail) => lines.push(status_line(
                if detail.is_error { "Error" } else { "Detail" },
                &tool_activity_title(detail),
                if detail.is_error {
                    Color::Red
                } else {
                    Color::Cyan
                },
            )),
            UiBlock::UserMessage(_)
            | UiBlock::TurnStatus(_)
            | UiBlock::Markdown { .. }
            | UiBlock::Reasoning(_)
            | UiBlock::Completion
            | UiBlock::Interrupt(_) => {}
        }
    }

    if lines.is_empty() {
        return vec![
            section_header("No tool activity yet", Color::Cyan),
            Line::from(Span::styled(
                "Requests, lifecycle cards, retries, output, and errors appear here.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
    }
    lines
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
        lines.extend(render_truncated_payload(output, "  ", Color::White));
        return lines;
    }

    vec![
        section_header("Selected/latest tool", Color::Cyan),
        Line::from(Span::styled(
            "No selected tool yet.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "Call id, arguments, output, and errors appear here without raw transcript spam.",
            Style::default().fg(Color::DarkGray),
        )),
    ]
}

fn render_tool_detail(detail: &UiToolDetail) -> Vec<Line<'static>> {
    let mut lines = vec![section_header(
        if detail.is_error {
            "Latest tool error"
        } else {
            "Latest tool detail"
        },
        if detail.is_error {
            Color::Red
        } else {
            Color::Cyan
        },
    )];
    lines.push(Line::from(vec![
        Span::styled("Title ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            detail.name.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    if let Some(call_id) = &detail.call_id {
        lines.push(Line::from(vec![
            Span::styled("Call id ", Style::default().fg(Color::DarkGray)),
            Span::raw(call_id.clone()),
        ]));
    }

    if let Some(arguments) = &detail.arguments {
        lines.push(section_header("Arguments", Color::Yellow));
        lines.extend(render_truncated_payload(arguments, "  ", Color::White));
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
        lines.extend(render_truncated_payload(output, "  ", Color::White));
    }

    if detail.arguments.is_none() && detail.output.is_none() {
        lines.push(Line::from(Span::styled(
            "Waiting for tool payload...",
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines
}

fn render_truncated_payload(text: &str, prefix: &'static str, color: Color) -> Vec<Line<'static>> {
    let logical_lines = text.lines().collect::<Vec<_>>();
    let rendered_lines = if logical_lines.len() > DETAIL_TRUNCATION_LINES {
        truncated_logical_lines(&logical_lines)
    } else {
        logical_lines
            .iter()
            .flat_map(|line| truncate_visual_line(line))
            .collect::<Vec<_>>()
    };

    rendered_lines
        .into_iter()
        .map(|line| payload_line(prefix, &line, color))
        .collect()
}

fn truncated_logical_lines(raw_lines: &[&str]) -> Vec<String> {
    let marker_rows = 1;
    let retained = DETAIL_TRUNCATION_LINES.saturating_sub(marker_rows);
    let head = retained / 2;
    let tail = retained.saturating_sub(head);
    let omitted = raw_lines.len().saturating_sub(retained);
    let mut lines = Vec::new();
    for line in raw_lines.iter().take(head) {
        lines.extend(truncate_visual_line(line));
    }
    lines.push(format!("… {omitted} lines truncated …"));
    for line in raw_lines.iter().skip(raw_lines.len().saturating_sub(tail)) {
        lines.extend(truncate_visual_line(line));
    }
    lines.truncate(DETAIL_TRUNCATION_LINES);
    lines
}

fn truncate_visual_line(line: &str) -> Vec<String> {
    const VISUAL_LIMIT: usize = 48;
    let char_count = line.chars().count();
    if char_count <= VISUAL_LIMIT {
        return vec![line.to_string()];
    }

    let head = VISUAL_LIMIT / 2;
    let tail = VISUAL_LIMIT - head;
    let start = line.chars().take(head).collect::<String>();
    let end = line
        .chars()
        .skip(char_count.saturating_sub(tail))
        .collect::<String>();
    vec![format!(
        "… {} chars truncated … {start} … {end}",
        char_count.saturating_sub(VISUAL_LIMIT)
    )]
}

fn payload_line(prefix: &'static str, line: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(prefix, Style::default().fg(Color::DarkGray)),
        Span::styled(line.to_string(), Style::default().fg(color)),
    ])
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

fn tool_activity_title(detail: &UiToolDetail) -> String {
    match &detail.call_id {
        Some(call_id) => format!("{} · {call_id}", detail.name),
        None => detail.name.clone(),
    }
}

fn focused_title(label: &'static str, state: &TuiViewState, pane: FocusPane) -> String {
    if state.focus == pane {
        format!("{label} *")
    } else {
        label.to_string()
    }
}

fn focus_color(state: &TuiViewState, pane: FocusPane) -> Color {
    if state.focus == pane {
        Color::Cyan
    } else {
        Color::Green
    }
}

fn next_focus(current: FocusPane) -> FocusPane {
    match current {
        FocusPane::Input => FocusPane::Transcript,
        FocusPane::Transcript => FocusPane::ToolActivity,
        FocusPane::ToolActivity => FocusPane::ToolDetail,
        FocusPane::ToolDetail => FocusPane::Input,
    }
}

fn previous_focus(current: FocusPane) -> FocusPane {
    match current {
        FocusPane::Input => FocusPane::ToolDetail,
        FocusPane::ToolDetail => FocusPane::ToolActivity,
        FocusPane::ToolActivity => FocusPane::Transcript,
        FocusPane::Transcript => FocusPane::Input,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScrollDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScrollJump {
    Start,
    End,
}

fn scroll_focused_pane(state: &mut TuiViewState, amount: u16, direction: ScrollDirection) {
    state.follow_latest = false;
    let target = match state.focus {
        FocusPane::Transcript => &mut state.transcript_scroll,
        FocusPane::ToolActivity => &mut state.activity_scroll,
        FocusPane::ToolDetail => &mut state.detail_scroll,
        FocusPane::Input => &mut state.transcript_scroll,
    };
    match direction {
        ScrollDirection::Up => *target = target.saturating_sub(amount),
        ScrollDirection::Down => *target = target.saturating_add(amount),
    }
}

fn jump_focused_pane(state: &mut TuiViewState, jump: ScrollJump) {
    let value = match jump {
        ScrollJump::Start => {
            state.follow_latest = false;
            0
        }
        ScrollJump::End => {
            state.follow_latest = true;
            u16::MAX
        }
    };
    match state.focus {
        FocusPane::Transcript => state.transcript_scroll = value,
        FocusPane::ToolActivity => state.activity_scroll = value,
        FocusPane::ToolDetail => state.detail_scroll = value,
        FocusPane::Input => state.transcript_scroll = value,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use forge_ui_model::{
        UiBlock, UiModel, UiRetryDelay, UiToolDetail, UiToolPhase, UiToolStatus, UiTurnStatus,
        submitted_user_turn,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn test_command_center_renders_64x10_single_column_without_empty_rail() {
        let fixture = submitted_user_turn("Hello from narrow TUI");

        let actual = render_conversation_to_string(&fixture, 64, 10);

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("Thinking"));
        assert!(actual.contains("Hello from narrow TUI"));
        assert!(!actual.contains("Tool detail"));
        assert!(!actual.contains("events="));
        assert!(!actual.contains("state="));
    }

    #[test]
    fn test_command_center_renders_80x20_compact_two_column_rail() {
        let fixture = tool_success_fixture();

        let actual = render_conversation_to_string(&fixture, 80, 20);

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("Ready"));
        assert!(actual.contains("Transcript"));
        assert!(actual.contains("Tool activity"));
        assert!(actual.contains("Tool detail"));
        assert!(actual.contains("Tool shell done"));
        assert!(actual.contains("completed - detail in rail"));
        assert!(!actual.contains("[markdown]"));
    }

    #[test]
    fn test_command_center_renders_120x32_premium_two_column_layout() {
        let fixture = tool_success_fixture();

        let actual = render_conversation_to_string(&fixture, 120, 32);

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("turns 0 · replies 1 · tools 1 · errors 0"));
        assert!(actual.contains("Transcript"));
        assert!(actual.contains("Tool activity"));
        assert!(actual.contains("Tool detail"));
        assert!(actual.contains("Latest tool detail"));
        assert!(actual.contains("Arguments"));
        assert!(actual.contains("Output"));
    }

    #[test]
    fn test_dense_height_below_ten_keeps_latest_status_visible() {
        let fixture = UiModel::new(vec![UiBlock::Completion]);

        let actual = render_conversation_to_string(&fixture, 80, 8);

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("Complete"));
        assert!(actual.contains("Assistant complete"));
    }

    #[test]
    fn test_conversation_layout_renders_running_turn_state() {
        let fixture = UiModel::new(vec![UiBlock::TurnStatus(UiTurnStatus::running())]);

        let actual = render_conversation_to_string(&fixture, 90, 10);

        assert!(actual.contains("Thinking"));
        assert!(actual.contains("Assistant"));
        assert!(actual.contains("running: provider stream running"));
    }

    #[test]
    fn test_conversation_layout_renders_markdown_headings_lists_quotes_and_code_without_tags() {
        let fixture = UiModel::new(vec![UiBlock::Markdown {
            text: "# Plan\n- inspect\n1. implement\n> verify\n```rust\nlet ok = true;\n```"
                .to_string(),
            partial: false,
        }]);

        let actual = render_conversation_to_string(&fixture, 120, 24);
        let expected = vec![
            "Assistant",
            "§ Plan",
            "• inspect",
            "1. implement",
            "▌ verify",
            "code · rust",
            "│ let ok = true;",
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
        let fixture = tool_success_fixture();

        let actual = render_conversation_to_string(&fixture, 104, 20);
        let expected = vec![
            "Assistant",
            "Checking project",
            "Tool shell done",
            "completed - detail in rail",
            "Latest tool detail",
            "Title shell",
            "Call id call-1",
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
    fn test_failed_tool_sets_error_pill_and_error_section() {
        let fixture = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Failed,
                summary: Some("exit 1".to_string()),
            }),
            UiBlock::ToolDetail(UiToolDetail {
                call_id: Some("call-failed".to_string()),
                name: "shell".to_string(),
                arguments: Some("{\"command\":\"false\"}".to_string()),
                output: Some("command failed".to_string()),
                is_error: true,
            }),
        ]);

        let actual = render_conversation_to_string(&fixture, 120, 24);

        assert!(actual.contains("Error"));
        assert!(actual.contains("Tool shell failed"));
        assert!(actual.contains("Latest tool error"));
        assert!(actual.contains("Call id call-failed"));
        assert!(actual.contains("command failed"));
    }

    #[test]
    fn test_conversation_layout_renders_retry_without_placeholder_detail_copy() {
        let fixture = UiModel::new(vec![UiBlock::Retry {
            cause: "network".to_string(),
            delay: UiRetryDelay::from_duration(Duration::from_millis(250)),
        }]);

        let actual = render_conversation_to_string(&fixture, 78, 11);
        let expected = vec!["Forge command center", "Retry", "waiting 250ms - network"];

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

        assert!(actual.contains("Ready"));
        assert!(!actual.contains("Working - assistant is responding"));
        assert!(!actual.contains("tools_running="));
    }

    #[test]
    fn test_conversation_status_uses_latest_pending_turn_after_completed_turn() {
        let fixture = UiModel::new(vec![
            UiBlock::Completion,
            UiBlock::UserMessage("Next question".to_string()),
            UiBlock::TurnStatus(UiTurnStatus::pending()),
        ]);

        let actual = render_conversation_to_string(&fixture, 92, 12);

        assert!(actual.contains("Thinking"));
        assert!(!actual.contains("Complete - response finished"));
    }

    #[test]
    fn test_conversation_status_uses_latest_running_turn_after_failed_tool() {
        let fixture = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Failed,
                summary: Some("exit 1".to_string()),
            }),
            UiBlock::UserMessage("Retry".to_string()),
            UiBlock::TurnStatus(UiTurnStatus::running()),
        ]);

        let actual = render_conversation_to_string(&fixture, 92, 12);

        assert!(actual.contains("Thinking"));
        assert!(!actual.contains("Attention - tool reported an error"));
    }

    #[test]
    fn test_rendered_conversation_avoids_dashboard_and_debug_vocabulary() {
        let fixture = UiModel::new(vec![UiBlock::Completion]);

        let actual = render_conversation_to_string(&fixture, 92, 10);

        assert!(!actual.to_ascii_lowercase().contains("dashboard"));
        assert!(!actual.to_ascii_lowercase().contains("debug"));
        assert!(!actual.contains("[markdown]"));
    }

    #[test]
    fn test_long_transcript_keeps_latest_visible_when_content_overflows() {
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
        assert!(actual.contains("completed - detail in rail"));
        assert!(!actual.contains("older assistant line 0"));
    }

    #[test]
    fn test_long_tool_output_uses_truncation_markers_and_keeps_tail() {
        let mut long_output = (0..40)
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

        let actual = render_conversation_to_string(&fixture, 120, 32);

        assert!(actual.contains("lines truncated"));
        assert!(actual.contains("ZENDMARK"));
        assert!(!actual.contains("older output line 10"));
    }

    #[test]
    fn test_tool_payload_does_not_leak_into_transcript_cards() {
        let fixture = UiModel::new(vec![
            UiBlock::ToolStatus(UiToolStatus {
                name: "shell".to_string(),
                phase: UiToolPhase::Finished,
                summary: Some("SECRET_RAW_OUTPUT_SHOULD_NOT_RENDER".to_string()),
            }),
            UiBlock::ToolOutput("SECRET_TOOL_OUTPUT_BODY_SHOULD_NOT_RENDER".to_string()),
        ]);

        let actual = render_conversation_to_string(&fixture, 64, 12);

        assert!(actual.contains("Tool shell done"));
        assert!(actual.contains("completed - detail in rail"));
        assert!(actual.contains("Tool output available in rail"));
        assert!(!actual.contains("SECRET_RAW_OUTPUT_SHOULD_NOT_RENDER"));
        assert!(!actual.contains("SECRET_TOOL_OUTPUT_BODY_SHOULD_NOT_RENDER"));
    }

    #[test]
    fn test_long_tool_detail_retains_at_most_truncation_limit_payload_rows() {
        let setup = (0..40)
            .map(|index| format!("output line {index}"))
            .collect::<Vec<_>>()
            .join("\n");

        let actual = render_truncated_payload(&setup, "  ", Color::White).len();

        let expected = DETAIL_TRUNCATION_LINES;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_single_line_tool_output_uses_character_truncation_marker() {
        let setup = format!("START{}END", "x".repeat(320));
        let fixture = UiModel::new(vec![UiBlock::ToolDetail(UiToolDetail {
            call_id: Some("call-long-line".to_string()),
            name: "shell".to_string(),
            arguments: None,
            output: Some(setup),
            is_error: false,
        })]);

        let actual = render_conversation_to_string(&fixture, 120, 32);

        assert!(actual.contains("chars truncated"));
        assert!(actual.contains("START"));
        assert!(actual.contains("END"));
        assert!(!actual.contains(&"x".repeat(220)));
    }

    #[test]
    fn test_keyboard_navigation_state_cycles_and_scrolls() {
        let mut setup = TuiViewState::default();
        let mut input = String::new();

        let actual = (
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::Tab)),
            setup.focus,
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::PageDown)),
            setup.transcript_scroll,
            setup.follow_latest,
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::End)),
            setup.follow_latest,
        );

        let expected = (
            TuiKeyAction::Consumed,
            FocusPane::Transcript,
            TuiKeyAction::Consumed,
            PAGE_SCROLL_LINES,
            false,
            TuiKeyAction::Consumed,
            true,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_keyboard_text_input_is_preserved_when_input_focused() {
        let mut setup = TuiViewState::default();
        let mut input = String::new();

        let actual = (
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::Char('h'))),
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::Char('i'))),
            apply_tui_key_event(&mut setup, &mut input, key(KeyCode::Enter)),
        );

        let expected = (
            TuiKeyAction::Consumed,
            TuiKeyAction::Consumed,
            TuiKeyAction::Submitted("hi".to_string()),
        );
        assert_eq!(actual, expected);
        assert_eq!(input, "");
    }

    #[test]
    fn test_interactive_shell_initial_frame_is_meaningful_conversation_ui() {
        let actual = render_interactive_shell_to_string(92, 14);

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("Ready"));
        assert!(actual.contains("Transcript"));
        assert!(actual.contains("Start a conversation"));
        assert!(actual.contains("Message"));
        assert!(actual.contains("Tab focus"));
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

        assert!(actual.contains("Forge command center"));
        assert!(actual.contains("Complete"));
        assert!(actual.contains("Assistant"));
        assert!(actual.contains("complete"));
        assert!(!actual.trim().is_empty());
    }

    fn tool_success_fixture() -> UiModel {
        UiModel::new(vec![
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
        ])
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
}
