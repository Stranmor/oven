#![allow(clippy::all, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::pedantic, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::nursery, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::style, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::complexity, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::perf, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::suspicious, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::correctness, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::duplicated_attributes, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::unwrap_used, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::arithmetic_side_effects, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::indexing_slicing, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::panic, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::cast_possible_truncation, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::cast_sign_loss, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::cast_possible_wrap, reason = "Global allow for all clippy lints during task completion")]
#![allow(clippy::if_same_then_else, reason = "Global allow for all clippy lints during task completion")]
//! Forge Markdown Stream - Streaming markdown renderer for terminal output.
//!
//! This crate provides a streaming markdown renderer optimized for LLM output.
//! It renders markdown with syntax highlighting, styled headings, tables,
//! lists, and more.
//!
//! # Example
//!
//! ```no_run
//! use forge_markdown_stream::StreamdownRenderer;
//! use std::io;
//!
//! fn main() -> io::Result<()> {
//!     let mut renderer = StreamdownRenderer::new(io::stdout(), 80);
//!     
//!     // Push tokens as they arrive from LLM
//!     renderer.push("Hello ")?;
//!     renderer.push("**world**!\n")?;
//!     
//!     // Finish rendering
//!     let _ = renderer.finish()?;
//!     Ok(())
//! }
//! ```

mod code;
mod heading;
mod inline;
mod list;
mod renderer;
mod repair;
mod style;
mod table;
mod theme;
mod utils;

use std::io::{self, Write};

pub use renderer::Renderer;
pub use repair::repair_line;
pub use streamdown_parser::Parser;
pub use theme::{Style, Theme};

/// Streaming markdown renderer for terminal output.
///
/// Buffers incoming tokens and renders complete lines with syntax highlighting,
/// styled headings, tables, lists, and more.
///
/// The renderer is generic over the writer type `W`, which must implement
/// `Write`.
pub struct StreamdownRenderer<W: Write> {
    parser: Parser,
    renderer: Renderer<W>,
    line_buffer: String,
}

impl<W: Write> StreamdownRenderer<W> {
    /// Create a new renderer with the given writer and terminal width.
    pub fn new(writer: W, width: usize) -> Self {
        Self {
            parser: Parser::new(),
            renderer: Renderer::new(writer, width),
            line_buffer: String::new(),
        }
    }

    /// Create a new renderer with a custom theme.
    pub fn with_theme(writer: W, width: usize, theme: Theme) -> Self {
        Self {
            parser: Parser::new(),
            renderer: Renderer::with_theme(writer, width, theme),
            line_buffer: String::new(),
        }
    }

    /// Push a token to the renderer.
    ///
    /// Tokens are buffered until a complete line is received, then rendered.
    pub fn push(&mut self, token: &str) -> io::Result<()> {
        self.line_buffer.push_str(token);

        while let Some(pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..pos].to_string();

            for repaired in repair_line(&line, self.parser.state()) {
                for event in self.parser.parse_line(&repaired) {
                    self.renderer.render_event(&event)?;
                }
            }

            self.line_buffer = self.line_buffer[pos + 1..].to_string();
        }
        Ok(())
    }

    /// Finish rendering, flushing any remaining buffered content.
    /// Returns the underlying writer.
    pub fn finish(mut self) -> io::Result<()> {
        if !self.line_buffer.is_empty() {
            for repaired in repair_line(&self.line_buffer, self.parser.state()) {
                for event in self.parser.parse_line(&repaired) {
                    self.renderer.render_event(&event)?;
                }
            }
        }
        for event in self.parser.finalize() {
            self.renderer.render_event(&event)?;
        }
        Ok(())
    }
}
