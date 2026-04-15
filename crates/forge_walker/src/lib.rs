#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod walker;

pub use walker::{File, Walker};
