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
pub mod banner;
pub mod built_in_commands;
mod cli;
mod completer;
mod conversation_selector;
mod display_constants;
mod editor;
mod image_paste;
mod info;
mod input;
mod model;
mod oauth_callback;
mod porcelain;
mod prompt;
mod sandbox;
mod state;
mod stream_renderer;
mod sync_display;
mod title_display;
mod tools_display;
pub mod tracker;
mod ui;
mod utils;
mod vscode;
mod zsh;

mod update;

use std::sync::LazyLock;

pub use cli::{Cli, TopLevelCommand};
pub use sandbox::Sandbox;
pub use title_display::*;
pub use ui::UI;

pub static TRACKER: LazyLock<forge_tracker::Tracker> =
    LazyLock::new(forge_tracker::Tracker::default);
