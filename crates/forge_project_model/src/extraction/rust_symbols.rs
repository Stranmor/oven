//! Rust AST symbol extraction.

use anyhow::{Context, Result};
use syn::Item;

use super::call_graph::extract_calls_into;
use super::range::SymbolRangeResolver;
use crate::types::{EdgeConfidence, GraphEdge, GraphEdgeKind, SymbolKind, SymbolNode};
use crate::util::{edge, edge_sort_key, provenance};

/// Extracted Rust symbol and edge bundle.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RustExtraction {
    /// Extracted symbols.
    pub symbols: Vec<SymbolNode>,
    /// Extracted symbol hierarchy edges.
    pub edges: Vec<GraphEdge>,
}

/// Extracts Rust AST symbols for structs, enums, traits, impls, functions,
/// methods, tests, and modules.
///
/// # Arguments
///
/// * `path` - Relative path used in identifiers and provenance.
/// * `content` - Rust source content.
///
/// # Errors
///
/// Returns an error when Rust source cannot be parsed by `syn`.
pub fn extract_rust_symbols(path: &str, content: &str) -> Result<RustExtraction> {
    let syntax = syn::parse_file(content).with_context(|| format!("parse Rust source {path}"))?;
    let mut extraction = RustExtraction::default();
    let mut ranges = SymbolRangeResolver::default();
    extract_items(
        path,
        content,
        None,
        &syntax.items,
        &mut extraction,
        &mut ranges,
    )?;
    extract_calls_into(path, content, &mut extraction)?;
    extraction
        .symbols
        .sort_by(|left, right| left.id.cmp(&right.id));
    extraction.edges.sort_by_key(edge_sort_key);
    Ok(extraction)
}

fn extract_items(
    path: &str,
    content: &str,
    parent: Option<String>,
    items: &[Item],
    extraction: &mut RustExtraction,
    ranges: &mut SymbolRangeResolver,
) -> Result<()> {
    for item in items {
        match item {
            Item::Struct(item) => {
                push_symbol(
                    path,
                    content,
                    parent.clone(),
                    item.ident.to_string(),
                    SymbolKind::Struct,
                    extraction,
                    ranges,
                );
            }
            Item::Enum(item) => {
                push_symbol(
                    path,
                    content,
                    parent.clone(),
                    item.ident.to_string(),
                    SymbolKind::Enum,
                    extraction,
                    ranges,
                );
            }
            Item::Trait(item_trait) => {
                let symbol = push_symbol(
                    path,
                    content,
                    parent.clone(),
                    item_trait.ident.to_string(),
                    SymbolKind::Trait,
                    extraction,
                    ranges,
                );
                for method in item_trait.items.iter().filter_map(|item| match item {
                    syn::TraitItem::Fn(function) => Some(function.sig.ident.to_string()),
                    _ => None,
                }) {
                    push_child_symbol(
                        path,
                        content,
                        symbol.id.clone(),
                        method,
                        SymbolKind::Method,
                        extraction,
                        ranges,
                    );
                }
            }
            Item::Impl(item_impl) => {
                let name = impl_name(item_impl);
                let symbol = push_symbol(
                    path,
                    content,
                    parent.clone(),
                    name,
                    SymbolKind::Impl,
                    extraction,
                    ranges,
                );
                for method in item_impl.items.iter().filter_map(|item| match item {
                    syn::ImplItem::Fn(function) => Some(function.sig.ident.to_string()),
                    _ => None,
                }) {
                    push_child_symbol(
                        path,
                        content,
                        symbol.id.clone(),
                        method,
                        SymbolKind::Method,
                        extraction,
                        ranges,
                    );
                }
            }
            Item::Fn(item_fn) => {
                let kind = if has_test_attribute(&item_fn.attrs) {
                    SymbolKind::Test
                } else {
                    SymbolKind::Function
                };
                push_symbol(
                    path,
                    content,
                    parent.clone(),
                    item_fn.sig.ident.to_string(),
                    kind,
                    extraction,
                    ranges,
                );
            }
            Item::Mod(item_mod) => {
                let symbol = push_symbol(
                    path,
                    content,
                    parent.clone(),
                    item_mod.ident.to_string(),
                    SymbolKind::Module,
                    extraction,
                    ranges,
                );
                if let Some((_, items)) = &item_mod.content {
                    extract_items(path, content, Some(symbol.id), items, extraction, ranges)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn push_symbol(
    path: &str,
    content: &str,
    parent: Option<String>,
    name: String,
    kind: SymbolKind,
    extraction: &mut RustExtraction,
    ranges: &mut SymbolRangeResolver,
) -> SymbolNode {
    let (start_line, end_line) = ranges.line_range(content, &name, &kind);
    let base_id = symbol_id(path, parent.as_deref(), &name, &kind);
    let id = unique_symbol_id(&base_id, extraction);
    let symbol = SymbolNode {
        id: id.clone(),
        name,
        kind,
        path: path.to_string(),
        parent: parent.clone(),
        start_line,
        end_line,
        provenance: provenance(path, Some(start_line), Some(end_line), "rust-ast", &id),
    };
    extraction.edges.push(edge(
        path,
        &id,
        GraphEdgeKind::Contains,
        1.0,
        EdgeConfidence::HeuristicHigh,
        provenance(
            path,
            Some(start_line),
            Some(end_line),
            "rust-ast",
            &format!("{path}->{id}"),
        ),
    ));
    if let Some(parent) = parent {
        extraction.edges.push(edge(
            &id,
            &parent,
            GraphEdgeKind::ChildOf,
            1.0,
            EdgeConfidence::HeuristicHigh,
            provenance(
                path,
                Some(start_line),
                Some(end_line),
                "rust-ast",
                &format!("{id}->{parent}"),
            ),
        ));
    }
    extraction.symbols.push(symbol.clone());
    symbol
}

fn push_child_symbol(
    path: &str,
    content: &str,
    parent: String,
    name: String,
    kind: SymbolKind,
    extraction: &mut RustExtraction,
    ranges: &mut SymbolRangeResolver,
) -> SymbolNode {
    push_symbol(path, content, Some(parent), name, kind, extraction, ranges)
}

fn impl_name(item_impl: &syn::ItemImpl) -> String {
    let type_name = match item_impl.self_ty.as_ref() {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_else(|| "Self".to_string()),
        _ => "Self".to_string(),
    };
    if let Some((_, trait_path, _)) = &item_impl.trait_ {
        let trait_name = trait_path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_else(|| "Trait".to_string());
        format!("impl {trait_name} for {type_name}")
    } else {
        format!("impl {type_name}")
    }
}

fn has_test_attribute(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("test"))
}

fn symbol_id(path: &str, parent: Option<&str>, name: &str, kind: &SymbolKind) -> String {
    match parent {
        Some(parent) => format!("symbol:{path}:{parent}:{kind:?}:{name}"),
        None => format!("symbol:{path}:{kind:?}:{name}"),
    }
}

fn unique_symbol_id(base_id: &str, extraction: &RustExtraction) -> String {
    if !extraction.symbols.iter().any(|symbol| symbol.id == base_id) {
        return base_id.to_string();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base_id}#{suffix}");
        if !extraction
            .symbols
            .iter()
            .any(|symbol| symbol.id == candidate)
        {
            return candidate;
        }
        suffix = suffix
            .checked_add(1)
            .expect("symbol id disambiguation suffix should not overflow");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn extracts_rust_symbols_with_line_ranges_and_provenance() -> Result<()> {
        let setup = "pub struct Root {\n    value: usize,\n}\n\nimpl Root {\n    pub fn new() -> Self { Self { value: 0 } }\n}\n\n#[test]\nfn root_test() {}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = vec![
            SymbolKind::Struct,
            SymbolKind::Impl,
            SymbolKind::Method,
            SymbolKind::Test,
        ];
        assert_eq!(
            actual
                .symbols
                .iter()
                .map(|symbol| symbol.kind.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            actual.symbols.iter().all(|symbol| symbol.start_line > 0
                && symbol.end_line >= symbol.start_line
                && symbol.provenance.path == "src/lib.rs"),
            true
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_line_ranges_are_bound_to_the_actual_ast_item() -> Result<()> {
        let setup = "pub struct Alpha;\n\nimpl Alpha {\n    pub fn duplicate() -> Self { Self }\n}\n\npub struct Beta;\n\nimpl Beta {\n    pub fn duplicate() -> Self { Self }\n}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = vec![4, 10];
        assert_eq!(
            actual
                .symbols
                .iter()
                .filter(|symbol| symbol.kind == SymbolKind::Method && symbol.name == "duplicate")
                .map(|symbol| symbol.start_line)
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_line_ranges_handle_qualified_signatures() -> Result<()> {
        let setup = "pub(crate) struct Scoped;\n\npub async fn async_root() {}\n\nunsafe fn unsafe_root() {}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = BTreeSet::from([
            ("Scoped".to_string(), SymbolKind::Struct, 1),
            ("async_root".to_string(), SymbolKind::Function, 3),
            ("unsafe_root".to_string(), SymbolKind::Function, 5),
        ]);
        assert_eq!(
            actual
                .symbols
                .iter()
                .map(|symbol| (symbol.name.clone(), symbol.kind.clone(), symbol.start_line))
                .collect::<BTreeSet<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_line_ranges_ignore_comment_shadow_matches() -> Result<()> {
        let setup = "// pub struct Real;\n\npub struct Real;\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = vec![3];
        assert_eq!(
            actual
                .symbols
                .iter()
                .filter(|symbol| symbol.kind == SymbolKind::Struct && symbol.name == "Real")
                .map(|symbol| symbol.start_line)
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_line_ranges_ignore_shadow_matches_in_comments_and_literals() -> Result<()> {
        let setup = "// pub struct LineComment;\n\n/* pub enum BlockComment { Fake } */\n\n/// pub trait DocComment {}\npub struct LineComment;\n\nconst TEXT: &str = \"pub enum Literal { Fake }\";\n\npub enum BlockComment {\n    Real,\n}\n\n#[doc = \"pub trait AttributeDoc {}\"]\npub trait DocComment {}\n\npub fn string_literal_shadow() {\n    let _shadow = r#\"pub struct LineComment;\"#;\n}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = BTreeSet::from([
            ("LineComment".to_string(), SymbolKind::Struct, 6),
            ("BlockComment".to_string(), SymbolKind::Enum, 10),
            ("DocComment".to_string(), SymbolKind::Trait, 15),
            (
                "string_literal_shadow".to_string(),
                SymbolKind::Function,
                17,
            ),
        ]);
        assert_eq!(
            actual
                .symbols
                .iter()
                .map(|symbol| (symbol.name.clone(), symbol.kind.clone(), symbol.start_line))
                .collect::<BTreeSet<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_line_ranges_handle_lifetime_parameters() -> Result<()> {
        let setup = "pub struct Borrowed<'a> {\n    value: &'a str,\n}\n\nimpl<'a> Borrowed<'a> {\n    pub fn new(value: &'a str) -> Self {\n        Self { value }\n    }\n}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let expected = BTreeSet::from([
            ("Borrowed".to_string(), SymbolKind::Struct, 1),
            ("impl Borrowed".to_string(), SymbolKind::Impl, 5),
            ("new".to_string(), SymbolKind::Method, 6),
        ]);
        assert_eq!(
            actual
                .symbols
                .iter()
                .map(|symbol| (symbol.name.clone(), symbol.kind.clone(), symbol.start_line))
                .collect::<BTreeSet<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn rust_symbol_ids_are_unique_for_multiple_impl_blocks_on_the_same_type() -> Result<()> {
        let setup = "pub struct Root;\n\nimpl Root {\n    pub fn first() {}\n}\n\nimpl Root {\n    pub fn second() {}\n}\n";
        let actual = extract_rust_symbols("src/lib.rs", setup)?;
        let ids = actual
            .symbols
            .iter()
            .map(|symbol| symbol.id.clone())
            .collect::<Vec<_>>();
        let unique_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
        let expected = ids.len();
        assert_eq!(unique_ids.len(), expected);
        assert_eq!(
            ids.iter()
                .filter(|id| id.as_str() == "symbol:src/lib.rs:Impl:impl Root"
                    || id.as_str() == "symbol:src/lib.rs:Impl:impl Root#2")
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                "symbol:src/lib.rs:Impl:impl Root".to_string(),
                "symbol:src/lib.rs:Impl:impl Root#2".to_string()
            ]
        );
        Ok(())
    }
}
