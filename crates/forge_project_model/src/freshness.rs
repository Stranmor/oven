//! Freshness comparison for deterministic project manifests.

use std::collections::BTreeMap;

use crate::types::{FreshnessState, ProjectManifest};

/// Compares two manifests and returns deterministic freshness invalidation
/// details.
///
/// # Arguments
///
/// * `previous` - Baseline manifest.
/// * `current` - Current manifest.
pub fn compare_freshness(previous: &ProjectManifest, current: &ProjectManifest) -> FreshnessState {
    let previous_files = previous
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.content_hash.as_str()))
        .collect::<BTreeMap<_, _>>();
    let current_files = current
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.content_hash.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut changed = Vec::new();
    let mut deleted = Vec::new();
    let mut added = Vec::new();
    let mut unchanged = Vec::new();

    for (path, previous_hash) in &previous_files {
        match current_files.get(path) {
            Some(current_hash) if current_hash == previous_hash => {
                unchanged.push((*path).to_string())
            }
            Some(_) => changed.push((*path).to_string()),
            None => deleted.push((*path).to_string()),
        }
    }
    for path in current_files.keys() {
        if !previous_files.contains_key(path) {
            added.push((*path).to_string());
        }
    }
    let fresh = changed.is_empty() && deleted.is_empty() && added.is_empty();
    FreshnessState { changed, deleted, added, unchanged, fresh }
}
