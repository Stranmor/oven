#![allow(
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

mod error;
mod parser;
mod schema_coercion;

pub use error::{JsonRepairError, Result};
pub use parser::json_repair;
pub use schema_coercion::coerce_to_schema;
