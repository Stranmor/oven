#![allow(clippy::unwrap_used, reason = "Allow")]
#![allow(clippy::arithmetic_side_effects, reason = "Allow")]
#![allow(clippy::indexing_slicing, reason = "Allow")]
#![allow(clippy::panic, reason = "Allow")]
#![allow(clippy::cast_possible_truncation, reason = "Allow")]
#![allow(clippy::cast_sign_loss, reason = "Allow")]
#![allow(clippy::cast_possible_wrap, reason = "Allow")]
#![allow(clippy::if_same_then_else, reason = "Allow")]
pub mod code;
pub mod diff;
pub mod grep;
pub mod markdown;

pub use code::SyntaxHighlighter;
pub use diff::DiffFormat;
pub use grep::GrepFormat;
pub use markdown::MarkdownFormat;
