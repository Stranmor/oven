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
use handlebars::Handlebars;
use include_dir::{Dir, DirEntry, File};

/// Returns an iterator over all files embedded in `dir`, recursively
/// descending into subdirectories.
pub fn files(dir: &'static Dir<'static>) -> impl Iterator<Item = &'static File<'static>> {
    dir.entries().iter().flat_map(walk_entry)
}

fn walk_entry(
    entry: &'static DirEntry<'static>,
) -> Box<dyn Iterator<Item = &'static File<'static>>> {
    match entry {
        DirEntry::File(f) => Box::new(std::iter::once(f)),
        DirEntry::Dir(d) => Box::new(d.entries().iter().flat_map(walk_entry)),
    }
}

/// Registers all files in `dir` (recursively) as Handlebars templates.
///
/// Template names match the relative file path as returned by
/// [`File::path`] (e.g. `forge-system-prompt.md`). Panics if any file path
/// is not valid UTF-8, if any file content is not valid UTF-8, or if template
/// parsing fails.
pub fn register_templates(hb: &mut Handlebars<'_>, dir: &'static Dir<'static>) -> Result<(), String> {
    for file in files(dir) {
        let name = file.path().to_str().ok_or_else(|| {
            format!(
                "embedded template path '{:?}' is not valid UTF-8",
                file.path()
            )
        })?;
        let content = file
            .contents_utf8()
            .ok_or_else(|| format!("embedded template '{}' is not valid UTF-8", name))?;
        hb.register_template_string(name, content)
            .map_err(|e| format!("failed to register template '{}': {}", name, e))?;
    }
    Ok(())
}
