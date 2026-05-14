//! Rust import and Cargo dependency graph extraction.

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use syn::{Item, UseTree, Visibility};
use toml_edit::DocumentMut;

use crate::types::{EdgeConfidence, GraphEdge, GraphEdgeKind};
use crate::util::{edge, edge_sort_key, provenance};

/// Extracts Rust import and module dependency graph edges.
///
/// # Arguments
///
/// * `path` - Relative path used as the source node.
/// * `content` - Rust source content.
///
/// # Errors
///
/// Returns an error when Rust source cannot be parsed by `syn`.
pub fn extract_rust_import_edges(path: &str, content: &str) -> Result<Vec<GraphEdge>> {
    let syntax = syn::parse_file(content).with_context(|| format!("parse Rust imports {path}"))?;
    let mut edges = Vec::new();
    for item in syntax.items {
        match item {
            Item::Use(item_use) => {
                let visibility = if matches!(item_use.vis, Visibility::Public(_)) {
                    "pub use"
                } else {
                    "use"
                };
                for target in flatten_use_tree(&item_use.tree) {
                    edges.push(edge(
                        path,
                        &target,
                        GraphEdgeKind::Imports,
                        0.9,
                        EdgeConfidence::HeuristicHigh,
                        provenance(path, None, None, visibility, &format!("{path}->{target}")),
                    ));
                }
            }
            Item::Mod(item_mod) => {
                let name = item_mod.ident.to_string();
                edges.push(edge(
                    path,
                    &name,
                    GraphEdgeKind::ModuleDeclares,
                    0.95,
                    EdgeConfidence::HeuristicHigh,
                    provenance(path, None, None, "mod", &format!("{path}->{name}")),
                ));
            }
            Item::ExternCrate(item_extern) => {
                let name = item_extern.ident.to_string();
                edges.push(edge(
                    path,
                    &name,
                    GraphEdgeKind::ExternCrate,
                    0.95,
                    EdgeConfidence::HeuristicHigh,
                    provenance(path, None, None, "extern crate", &format!("{path}->{name}")),
                ));
            }
            _ => {}
        }
    }
    edges.sort_by_key(edge_sort_key);
    Ok(edges)
}

/// Extracts Cargo dependency graph edges from a Cargo.toml document.
///
/// # Arguments
///
/// * `path` - Relative Cargo.toml path.
/// * `content` - Cargo.toml content.
///
/// # Errors
///
/// Returns an error when TOML cannot be parsed.
pub fn extract_cargo_dependency_edges(path: &str, content: &str) -> Result<Vec<GraphEdge>> {
    let document = content
        .parse::<DocumentMut>()
        .with_context(|| format!("parse Cargo manifest {path}"))?;
    let mut edges = Vec::new();
    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(table) = document.get(section).and_then(|item| item.as_table()) {
            for name in table.iter().map(|(name, _)| name).collect::<BTreeSet<_>>() {
                edges.push(edge(
                    path,
                    name,
                    GraphEdgeKind::CargoDependency,
                    1.0,
                    EdgeConfidence::ExactCompiler,
                    provenance(path, None, None, section, &format!("{path}->{name}")),
                ));
            }
        }
    }
    edges.sort_by_key(edge_sort_key);
    Ok(edges)
}

fn flatten_use_tree(tree: &UseTree) -> Vec<String> {
    fn walk(prefix: String, tree: &UseTree, output: &mut Vec<String>) {
        match tree {
            UseTree::Path(path) => walk(
                join_path(&prefix, &path.ident.to_string()),
                &path.tree,
                output,
            ),
            UseTree::Name(name) => output.push(join_path(&prefix, &name.ident.to_string())),
            UseTree::Rename(rename) => output.push(join_path(&prefix, &rename.ident.to_string())),
            UseTree::Glob(_) => output.push(join_path(&prefix, "*")),
            UseTree::Group(group) => {
                for item in &group.items {
                    walk(prefix.clone(), item, output);
                }
            }
        }
    }
    let mut output = Vec::new();
    walk(String::new(), tree, &mut output);
    output.sort();
    output
}

fn join_path(prefix: &str, part: &str) -> String {
    if prefix.is_empty() {
        part.to_string()
    } else {
        format!("{prefix}::{part}")
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn extracts_import_and_cargo_dependency_graph() -> Result<()> {
        let setup = "use serde::{Serialize, Deserialize};\npub use crate::model::Widget;\nmod model;\nextern crate core;\n";
        let mut actual = extract_rust_import_edges("src/lib.rs", setup)?;
        actual.extend(extract_cargo_dependency_edges(
            "Cargo.toml",
            "[dependencies]\nserde = \"1\"\n",
        )?);
        actual.sort_by_key(edge_sort_key);
        let expected = vec![
            GraphEdgeKind::CargoDependency,
            GraphEdgeKind::ExternCrate,
            GraphEdgeKind::Imports,
            GraphEdgeKind::ModuleDeclares,
            GraphEdgeKind::Imports,
            GraphEdgeKind::Imports,
        ];
        assert_eq!(
            actual
                .iter()
                .map(|edge| edge.kind.clone())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            actual.iter().any(|edge| edge.to == "serde::Serialize"),
            true
        );
        Ok(())
    }
}
