#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::cast_possible_wrap
)]
mod can_track;
mod client_id;
mod collect;
mod dispatch;
mod error;
mod event;
mod log;
mod rate_limit;
pub use can_track::VERSION;
pub use dispatch::Tracker;
use error::Result;
pub use event::{Event, EventKind, ToolCallPayload};
pub use log::{Guard, init_tracing};
