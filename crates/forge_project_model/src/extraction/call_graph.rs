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
    let mut resolver = SourceSymbolResolver::new(&lookup);
    let mut edges = Vec::new();
    for item in &syntax.items {
        extract_item_calls(path, item, &lookup, &mut resolver, &mut edges);
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
    resolver: &mut SourceSymbolResolver<'_>,
    edges: &mut Vec<GraphEdge>,
) {
    match item {
        Item::Fn(item_fn) => {
            if let Some(source) = resolver.function(item_fn) {
                collect_block_calls(path, source, &item_fn.block, lookup, edges);
            }
        }
        Item::Impl(item_impl) => {
            let source_impl = resolver.impl_block();
            for item in &item_impl.items {
                if let ImplItem::Fn(function) = item
                    && let Some(source) = resolver.impl_method(source_impl, function)
                {
                    collect_block_calls(path, source, &function.block, lookup, edges);
                }
            }
        }
        Item::Trait(item_trait) => {
            let source_trait = resolver.trait_item(item_trait);
            for item in &item_trait.items {
                if let TraitItem::Fn(function) = item
                    && let (Some(source), Some(default_block)) = (
                        resolver.trait_method(source_trait, function),
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
                    extract_item_calls(path, item, lookup, resolver, edges);
                }
            }
        }
        _ => {}
    }
}

struct SourceSymbolResolver<'a> {
    lookup: &'a SymbolLookup<'a>,
    function_positions: BTreeMap<String, usize>,
    impl_position: usize,
    trait_positions: BTreeMap<String, usize>,
}

impl<'a> SourceSymbolResolver<'a> {
    fn new(lookup: &'a SymbolLookup<'a>) -> Self {
        Self {
            lookup,
            function_positions: BTreeMap::new(),
            impl_position: 0,
            trait_positions: BTreeMap::new(),
        }
    }

    fn function(&mut self, function: &syn::ItemFn) -> Option<&'a SymbolNode> {
        self.next_named_symbol(
            &function.sig.ident.to_string(),
            |symbol| matches!(symbol.kind, SymbolKind::Function | SymbolKind::Test),
            SourceCursorKind::Function,
        )
    }

    fn impl_block(&mut self) -> Option<&'a SymbolNode> {
        let symbol = self
            .lookup
            .by_kind(&SymbolKind::Impl)
            .into_iter()
            .nth(self.impl_position)?;
        self.impl_position = self
            .impl_position
            .checked_add(1)
            .expect("source impl cursor should not overflow");
        Some(symbol)
    }

    fn trait_item(&mut self, item_trait: &syn::ItemTrait) -> Option<&'a SymbolNode> {
        self.next_named_symbol(
            &item_trait.ident.to_string(),
            |symbol| symbol.kind == SymbolKind::Trait,
            SourceCursorKind::Trait,
        )
    }

    fn impl_method(
        &self,
        source_impl: Option<&SymbolNode>,
        function: &syn::ImplItemFn,
    ) -> Option<&'a SymbolNode> {
        self.child_method(source_impl, &function.sig.ident.to_string())
    }

    fn trait_method(
        &self,
        source_trait: Option<&SymbolNode>,
        function: &syn::TraitItemFn,
    ) -> Option<&'a SymbolNode> {
        self.child_method(source_trait, &function.sig.ident.to_string())
    }

    fn child_method(&self, parent: Option<&SymbolNode>, name: &str) -> Option<&'a SymbolNode> {
        let parent_id = parent.map(|symbol| symbol.id.as_str())?;
        self.lookup.method_in_parent(parent_id, name)
    }

    fn next_named_symbol(
        &mut self,
        name: &str,
        matches_kind: impl Fn(&SymbolNode) -> bool,
        cursor_kind: SourceCursorKind,
    ) -> Option<&'a SymbolNode> {
        let position = match cursor_kind {
            SourceCursorKind::Function => {
                self.function_positions.entry(name.to_string()).or_default()
            }
            SourceCursorKind::Trait => self.trait_positions.entry(name.to_string()).or_default(),
        };
        let candidates = self.lookup.by_name.get(name)?;
        let symbol = candidates
            .iter()
            .copied()
            .filter(|symbol| matches_kind(symbol))
            .nth(*position)?;
        *position = position
            .checked_add(1)
            .expect("source symbol cursor should not overflow");
        Some(symbol)
    }
}

enum SourceCursorKind {
    Function,
    Trait,
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
        for symbols in by_name.values_mut() {
            symbols.sort_by_key(|symbol| (symbol.start_line, symbol.end_line, symbol.id.as_str()));
        }
        Self { by_name }
    }

    fn method_in_parent(&self, parent_id: &str, name: &str) -> Option<&'a SymbolNode> {
        self.by_name.get(name).and_then(|symbols| {
            symbols.iter().copied().find(|symbol| {
                symbol.kind == SymbolKind::Method && symbol.parent.as_deref() == Some(parent_id)
            })
        })
    }

    fn by_kind(&self, kind: &SymbolKind) -> Vec<&'a SymbolNode> {
        let mut symbols = self
            .by_name
            .values()
            .flat_map(|symbols| symbols.iter().copied())
            .filter(|symbol| &symbol.kind == kind)
            .collect::<Vec<_>>();
        symbols.sort_by_key(|symbol| (symbol.start_line, symbol.end_line, symbol.id.as_str()));
        symbols
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

    #[test]
    fn call_edges_use_the_actual_duplicate_method_source() -> Result<()> {
        let setup = "pub fn helper_a() {}
pub fn helper_b() {}

pub struct Alpha;
pub struct Beta;

impl Alpha {
    pub fn run(&self) {
        helper_a();
    }
}

impl Beta {
    pub fn run(&self) {
        helper_b();
    }
}
";
        let symbols = extract_rust_symbols("src/lib.rs", setup)?.symbols;
        let actual = extract_rust_call_edges("src/lib.rs", setup, &symbols)?;
        let beta_run = symbols
            .iter()
            .find(|symbol| symbol.id.contains("Impl:impl Beta") && symbol.name == "run")
            .expect("fixture should include Beta::run")
            .id
            .clone();
        let expected = true;
        assert_eq!(
            actual
                .iter()
                .any(|edge| edge.from == beta_run && edge.to.contains("Function:helper_b")),
            expected
        );
        Ok(())
    }
}
