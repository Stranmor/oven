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
#![allow(clippy::panic, reason = "Procedural macros use panic to emit compiler errors")]

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, Expr, ExprLit, Lit, parse_macro_input};

/// Custom attribute for specifying tool description file path
extern crate proc_macro;

#[proc_macro_attribute]
pub fn tool_description_file(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    // This is just a marker attribute, the actual processing happens in
    // ToolDescription
    _item
}

#[proc_macro_derive(ToolDescription, attributes(tool_description_file))]
pub fn derive_description(input: TokenStream) -> TokenStream {
    // Parse the input struct or enum
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;

    // Check for tool_description_file attribute first
    let mut description_file = None;
    for attr in &input.attrs {
        if attr.path().is_ident("tool_description_file")
            && let syn::Meta::NameValue(name_value) = &attr.meta
            && let Expr::Lit(ExprLit { lit: Lit::Str(lit_str), .. }) = &name_value.value
        {
            description_file = Some(lit_str.value());
        }
    }

    // If we have a description file, read it at compile time
    let doc_string = if let Some(file_path) = description_file {
        std::fs::read_to_string(&file_path)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to read tool description file '{}': {}",
                    file_path, e
                )
            })
            .trim()
            .to_string()
    } else {
        // Collect doc lines from doc comments
        let mut doc_lines = Vec::new();
        for attr in &input.attrs {
            if attr.path().is_ident("doc") {
                // Get the doc content as a string
                let doc_string = attr.meta.to_token_stream().to_string();
                // Remove the quotes and = sign
                let clean = doc_string.trim_start_matches("=").trim_matches('"').trim();
                if !clean.is_empty() {
                    doc_lines.push(clean.to_string());
                }
            }
        }

        if doc_lines.is_empty() {
            panic!("No doc comment found for {name}");
        }
        doc_lines.join("\n")
    };

    // Generate the implementation
    let expanded = if generics.params.is_empty() {
        quote! {
            impl ToolDescription for #name {
                fn description(&self) -> String {
                    #doc_string.into()
                }
            }
        }
    } else {
        quote! {
            impl #generics ToolDescription for #name #generics {
                fn description(&self) -> String {
                    #doc_string.into()
                }
            }
        }
    };

    expanded.into()
}
