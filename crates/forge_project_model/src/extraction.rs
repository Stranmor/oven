//! Rust AST and dependency extraction entry points.

mod call_graph;
mod cargo_metadata;
mod dependency;
mod range;
mod rust_symbols;

pub use call_graph::extract_rust_call_edges;
pub(crate) use cargo_metadata::{
    CargoManifestInput, StaticCargoMetadata, extract_static_cargo_metadata,
};
pub use dependency::{extract_cargo_dependency_edges, extract_rust_import_edges};
pub use rust_symbols::{RustExtraction, extract_rust_symbols};
