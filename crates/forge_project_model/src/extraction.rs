//! Rust AST and dependency extraction entry points.

mod dependency;
mod range;
mod rust_symbols;

pub use dependency::{extract_cargo_dependency_edges, extract_rust_import_edges};
pub use rust_symbols::{RustExtraction, extract_rust_symbols};
