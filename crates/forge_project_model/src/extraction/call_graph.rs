//! Rust call graph extraction from syntax-level call expressions.

use crate::extraction::RustExtraction;
use crate::types::{EdgeConfidence, GraphEdge, GraphEdgeKind, SymbolKind, SymbolNode};
use crate::util::{edge, edge_sort_key, provenance};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use syn::visit::Visit;
use syn::{ExprCall, ExprMethodCall, ImplItem, Item, TraitItem};

/// Extracts syntax-level Rust call graph edges without claiming compiler exactness.
///
/// # Arguments
///
/// * `path` - Relative source path used for provenance.
/// * `content` - Rust source content.
/// * `symbols` - Symbols previously extracted from the same source file.
///
/// # Errors
///
/// Returns an error when Rust source cannot be parsed by `syn`.
pub fn extract_rust_call_edges(
    path: &str,
    content: &str,
    symbols: &[SymbolNode],
) -> Result<Vec<GraphEdge>> {
    let syntax = syn::parse_file(content).with_context(|| format!("parse Rust calls {path}"))?;
    let lookup = SymbolLookup::new(symbols);
    let mut edges = Vec::new();
    for item in &syntax.items {
        extract_item_calls(path, item, &lookup, &mut edges);
    }
    edges.sort_by_key(edge_sort_key);
    edges.dedup_by(|left, right| {
        left.from == right.from
            && left.to == right.to
            && left.kind == right.kind
            && left.confidence_kind == right.confidence_kind
    });
    Ok(edges)
}

fn extract_item_calls(
    path: &str,
    item: &Item,
    lookup: &SymbolLookup<'_>,
    edges: &mut Vec<GraphEdge>,
) {
    match item {
        Item::Fn(item_fn) => {
            if let Some(source) = lookup.function(&item_fn.sig.ident.to_string()) {
                collect_block_calls(path, source, &item_fn.block, lookup, edges);
            }
        }
        Item::Impl(item_impl) => {
            for item in &item_impl.items {
                if let ImplItem::Fn(function) = item
                    && let Some(source) = lookup.method(&function.sig.ident.to_string())
                {
                    collect_block_calls(path, source, &function.block, lookup, edges);
                }
            }
        }
        Item::Trait(item_trait) => {
            for item in &item_trait.items {
                if let TraitItem::Fn(function) = item
                    && let (Some(source), Some(default_block)) = (
                        lookup.method(&function.sig.ident.to_string()),
                        function.default.as_ref(),
                    )
                {
                    collect_block_calls(path, source, default_block, lookup, edges);
                }
            }
        }
        Item::Mod(item_mod) => {
            if let Some((_, items)) = &item_mod.content {
                for item in items {
                    extract_item_calls(path, item, lookup, edges);
                }
            }
        }
        _ => {}
    }
}

fn collect_block_calls(
    path: &str,
    source: &SymbolNode,
    block: &syn::Block,
    lookup: &SymbolLookup<'_>,
    edges: &mut Vec<GraphEdge>,
) {
    let mut visitor = CallVisitor { path, source, lookup, edges };
    visitor.visit_block(block);
}

struct CallVisitor<'a> {
    path: &'a str,
    source: &'a SymbolNode,
    lookup: &'a SymbolLookup<'a>,
    edges: &'a mut Vec<GraphEdge>,
}

impl<'ast> Visit<'ast> for CallVisitor<'_> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(name) = callable_name(&node.func) {
            self.push_call(name);
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.push_call(node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }
}

impl CallVisitor<'_> {
    fn push_call(&mut self, name: String) {
        let (target, confidence, confidence_kind) = self
            .lookup
            .callable(&name)
            .map(|symbol| (symbol.id.clone(), 0.85, EdgeConfidence::HeuristicHigh))
            .unwrap_or_else(|| {
                (
                    format!("unresolved-call:{name}"),
                    0.45,
                    EdgeConfidence::HeuristicLow,
                )
            });
        self.edges.push(edge(
            &self.source.id,
            &target,
            GraphEdgeKind::Calls,
            confidence,
            confidence_kind,
            provenance(
                self.path,
                Some(self.source.start_line),
                Some(self.source.end_line),
                "rust-ast-call-heuristic",
                &format!("{}->{target}", self.source.id),
            ),
        ));
    }
}

fn callable_name(expression: &syn::Expr) -> Option<String> {
    match expression {
        syn::Expr::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        _ => None,
    }
}

struct SymbolLookup<'a> {
    by_name: BTreeMap<&'a str, Vec<&'a SymbolNode>>,
}

impl<'a> SymbolLookup<'a> {
    fn new(symbols: &'a [SymbolNode]) -> Self {
        let mut by_name = BTreeMap::<&str, Vec<&SymbolNode>>::new();
        for symbol in symbols {
            by_name
                .entry(symbol.name.as_str())
                .or_default()
                .push(symbol);
        }
        Self { by_name }
    }

    fn function(&self, name: &str) -> Option<&'a SymbolNode> {
        self.by_name.get(name).and_then(|symbols| {
            symbols
                .iter()
                .copied()
                .find(|symbol| matches!(symbol.kind, SymbolKind::Function | SymbolKind::Test))
        })
    }

    fn method(&self, name: &str) -> Option<&'a SymbolNode> {
        self.by_name.get(name).and_then(|symbols| {
            symbols
                .iter()
                .copied()
                .find(|symbol| symbol.kind == SymbolKind::Method)
        })
    }

    fn callable(&self, name: &str) -> Option<&'a SymbolNode> {
        self.by_name.get(name).and_then(|symbols| {
            symbols.iter().copied().find(|symbol| {
                matches!(
                    symbol.kind,
                    SymbolKind::Function | SymbolKind::Method | SymbolKind::Test
                )
            })
        })
    }
}

pub(super) fn extract_calls_into(
    path: &str,
    content: &str,
    extraction: &mut RustExtraction,
) -> Result<()> {
    extraction
        .edges
        .extend(extract_rust_call_edges(path, content, &extraction.symbols)?);
    extraction.edges.sort_by_key(edge_sort_key);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract_rust_symbols;
    use pretty_assertions::assert_eq;

    #[test]
    fn records_function_and_method_calls_without_exact_compiler_confidence() -> Result<()> {
        let setup = "pub fn helper() {}

pub struct Root;

impl Root {
    pub fn leaf(&self) {}
    pub fn run(&self) {
        helper();
        self.leaf();
        missing();
    }
}
";
        let symbols = extract_rust_symbols("src/lib.rs", setup)?.symbols;
        let actual = extract_rust_call_edges("src/lib.rs", setup, &symbols)?;
        let expected = vec![EdgeConfidence::HeuristicHigh, EdgeConfidence::HeuristicLow];
        assert_eq!(
            actual
                .iter()
                .map(|edge| edge.confidence_kind.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            actual.iter().any(
                |edge| edge.kind == GraphEdgeKind::Calls && edge.to.contains("Function:helper")
            ),
            true
        );
        assert_eq!(
            actual
                .iter()
                .all(|edge| edge.confidence_kind != EdgeConfidence::ExactCompiler),
            true
        );
        Ok(())
    }
}
