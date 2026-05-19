//! Ignore-aware project indexing, persistence, sharding, and episodes.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;

use crate::extraction::{
    extract_cargo_dependency_edges, extract_rust_import_edges, extract_rust_symbols,
};
use crate::freshness::compare_freshness;
use crate::policy::LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME;
use crate::types::{
    ContextPack, ContextPackArtifactId, FileNode, FileNodeKind, FreshnessProofLevel,
    FreshnessState, Language, ManifestFreshnessEvaluation, ProjectManifest, ShardManifest,
    ShardStrategy, SourceFile, SymbolNode, ToolEpisode,
};
use crate::util::{
    detect_language, edge_sort_key, hash_text, line_count, manifest_hash, normalize_path,
    provenance, ranges_overlap,
};

/// Project indexer that owns filesystem scanning and deterministic storage.
pub struct ProjectIndexer {
    root: PathBuf,
    model_dir: PathBuf,
}

impl ProjectIndexer {
    /// Creates a project indexer.
    ///
    /// # Arguments
    ///
    /// * `root` - Project root used for ignore-aware walking.
    /// * `model_dir` - Directory where deterministic JSON and JSONL model files
    ///   are stored.
    pub fn new(root: impl Into<PathBuf>, model_dir: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), model_dir: model_dir.into() }
    }

    /// Builds a deterministic project manifest from the configured root.
    ///
    /// # Errors
    ///
    /// Returns an error when walking, reading, parsing, or hashing project
    /// files fails.
    pub fn index(&self) -> Result<ProjectManifest> {
        let mut files = Vec::new();
        let mut rust_sources = BTreeMap::new();
        let mut cargo_tomls = Vec::new();

        for result in WalkBuilder::new(&self.root)
            .hidden(false)
            .git_ignore(true)
            .build()
        {
            let entry = result.context("walk project tree")?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let relative = normalize_path(
                path.strip_prefix(&self.root)
                    .context("strip project root")?,
            );
            if is_model_storage_path(&relative) || is_ignore_control_file(&relative) {
                continue;
            }
            let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
            if std::str::from_utf8(&bytes).is_err() {
                continue;
            }
            let content = String::from_utf8(bytes).context("validated UTF-8 source content")?;
            let language = detect_language(&relative);
            let content_hash = hash_text(&content);
            let lines = line_count(&content);
            let source = SourceFile {
                path: relative.clone(),
                language: language.clone(),
                bytes: content.len() as u64,
                lines,
                content_hash: content_hash.clone(),
                provenance: provenance(&relative, None, None, "indexer", &content_hash),
            };
            if language == Language::Rust {
                rust_sources.insert(relative.clone(), content);
            } else if relative.ends_with("Cargo.toml") {
                cargo_tomls.push((relative.clone(), content));
            }
            files.push(source);
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));

        let file_nodes = build_file_nodes(&files);
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        for (path, content) in &rust_sources {
            let extracted = extract_rust_symbols(path, content)?;
            symbols.extend(extracted.symbols);
            edges.extend(extracted.edges);
            edges.extend(extract_rust_import_edges(path, content)?);
        }
        for (path, content) in &cargo_tomls {
            edges.extend(extract_cargo_dependency_edges(path, content)?);
        }
        symbols.sort_by(|left, right| left.id.cmp(&right.id));
        edges.sort_by_key(edge_sort_key);
        let shards = build_shards(
            &files,
            &symbols,
            &self.root,
            &ShardStrategy::RustSemanticWithLineFallback,
        )?;
        let manifest_hash = manifest_hash(&files);
        Ok(ProjectManifest {
            version: 1,
            root: self.root.clone(),
            files,
            file_nodes,
            symbols,
            edges,
            shards,
            manifest_hash,
        })
    }

    /// Writes the manifest to `project_manifest.json` using stable pretty JSON.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Manifest to persist.
    ///
    /// # Errors
    ///
    /// Returns an error when the model directory cannot be created or JSON
    /// cannot be written.
    pub fn write_manifest(&self, manifest: &ProjectManifest) -> Result<PathBuf> {
        fs::create_dir_all(&self.model_dir).context("create model dir")?;
        let path = self.model_dir.join(LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME);
        let json = serde_json::to_string_pretty(manifest).context("serialize manifest")?;
        fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        Ok(path)
    }

    /// Reads the deterministic project manifest from storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest cannot be read or decoded.
    pub fn read_manifest(&self) -> Result<ProjectManifest> {
        let path = self.model_dir.join(LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME);
        let json = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&json).context("deserialize manifest")
    }

    /// Computes the deterministic artifact identifier for a context pack.
    ///
    /// # Arguments
    ///
    /// * `pack` - Redaction-safe context pack to identify.
    ///
    /// # Errors
    ///
    /// Returns an error when stable JSON serialization fails.
    pub fn context_pack_artifact_id(&self, pack: &ContextPack) -> Result<ContextPackArtifactId> {
        ContextPackArtifactId::new(hash_text(&pack.to_stable_json()?))
    }

    /// Atomically writes a non-empty context pack artifact and validates readback.
    ///
    /// # Arguments
    ///
    /// * `pack` - Redaction-safe context pack artifact to persist.
    ///
    /// # Errors
    ///
    /// Returns an error when the pack has no evidence, storage cannot be
    /// created, JSON cannot be written atomically, or readback differs from the
    /// input pack.
    pub fn write_context_pack(&self, pack: &ContextPack) -> Result<PathBuf> {
        if pack.evidence.is_empty() {
            anyhow::bail!("context pack artifact requires non-empty evidence");
        }
        let id = self.context_pack_artifact_id(pack)?;
        let directory = self.context_pack_dir();
        fs::create_dir_all(&directory).context("create context pack dir")?;
        let path = self.context_pack_path(&id);
        let json = pack.to_stable_json()?;
        let temp_path = self.write_context_pack_temp(&directory, &id, &json)?;
        fs::rename(&temp_path, &path)
            .with_context(|| format!("rename {} to {}", temp_path.display(), path.display()))?;
        sync_directory(&directory)?;
        let actual = self.read_context_pack(&id)?;
        if &actual != pack {
            anyhow::bail!("context pack artifact readback mismatch: {}", id);
        }
        Ok(path)
    }

    /// Reads a persisted context pack artifact by deterministic identifier.
    ///
    /// # Arguments
    ///
    /// * `id` - Hash-only context pack artifact identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact cannot be read, decoded, or its
    /// content-derived identifier does not match `id`.
    pub fn read_context_pack(&self, id: &ContextPackArtifactId) -> Result<ContextPack> {
        let path = self.context_pack_path(id);
        let json = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let pack =
            serde_json::from_str::<ContextPack>(&json).context("deserialize context pack")?;
        let actual_id = self.context_pack_artifact_id(&pack)?;
        if &actual_id != id {
            anyhow::bail!(
                "context pack artifact id mismatch: expected {}, got {}",
                id,
                actual_id
            );
        }
        Ok(pack)
    }

    /// Lists deterministic context pack artifact identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact directory cannot be listed or an
    /// artifact filename is not a valid hash-only identifier.
    pub fn list_context_pack_artifacts(&self) -> Result<Vec<ContextPackArtifactId>> {
        let directory = self.context_pack_dir();
        if !directory.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in
            fs::read_dir(&directory).with_context(|| format!("read {}", directory.display()))?
        {
            let entry = entry.context("read context pack artifact entry")?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
                anyhow::bail!(
                    "context pack artifact filename is not UTF-8: {}",
                    path.display()
                );
            };
            let Some(id) = file_name.strip_suffix(".json") else {
                continue;
            };
            ids.push(ContextPackArtifactId::new(id.to_string())?);
        }
        ids.sort();
        Ok(ids)
    }

    fn context_pack_dir(&self) -> PathBuf {
        self.model_dir.join("context_packs")
    }

    fn context_pack_path(&self, id: &ContextPackArtifactId) -> PathBuf {
        self.context_pack_dir()
            .join(format!("{}.json", id.as_str()))
    }

    fn write_context_pack_temp(
        &self,
        directory: &Path,
        id: &ContextPackArtifactId,
        json: &str,
    ) -> Result<PathBuf> {
        for attempt in 0..100u32 {
            let temp_path = directory.join(format!(
                ".{}.{}.{}.tmp",
                id.as_str(),
                std::process::id(),
                attempt
            ));
            let mut file = match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| format!("create {}", temp_path.display()));
                }
            };
            file.write_all(json.as_bytes())
                .with_context(|| format!("write {}", temp_path.display()))?;
            file.write_all(b"\n")
                .with_context(|| format!("write newline {}", temp_path.display()))?;
            file.flush()
                .with_context(|| format!("flush {}", temp_path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync {}", temp_path.display()))?;
            return Ok(temp_path);
        }
        anyhow::bail!("could not create unique context pack temp file");
    }

    ///
    /// # Arguments
    ///
    /// * `episode` - Episode record whose payloads are fingerprints, not raw
    ///   secret-bearing data.
    ///
    /// # Errors
    ///
    /// Returns an error when the JSONL file cannot be opened or written.
    pub fn append_episode(&self, episode: &ToolEpisode) -> Result<PathBuf> {
        fs::create_dir_all(&self.model_dir).context("create model dir")?;
        let path = self.model_dir.join("tool_episodes.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        let json = serde_json::to_string(episode).context("serialize episode")?;
        writeln!(file, "{json}").context("append episode")?;
        Ok(path)
    }

    /// Reads all tool episodes from deterministic JSONL storage.
    ///
    /// # Errors
    ///
    /// Returns an error when JSONL cannot be read or any line cannot be
    /// decoded.
    pub fn read_episodes(&self) -> Result<Vec<ToolEpisode>> {
        let path = self.model_dir.join("tool_episodes.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut episodes = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line.context("read episode line")?;
            if line.trim().is_empty() {
                continue;
            }
            episodes.push(serde_json::from_str(&line).context("deserialize episode")?);
        }
        Ok(episodes)
    }

    /// Evaluates a persisted manifest against the current ignore-aware filesystem.
    ///
    /// # Arguments
    ///
    /// * `previous` - Persisted manifest used as the freshness baseline.
    ///
    /// # Errors
    ///
    /// Returns an error when current indexing fails.
    pub fn evaluate_manifest_freshness(
        &self,
        previous: &ProjectManifest,
    ) -> Result<ManifestFreshnessEvaluation> {
        Ok(ManifestFreshnessEvaluation {
            state: compare_freshness(previous, &self.index()?),
            proof_level: FreshnessProofLevel::FullFilesystem,
        })
    }

    /// Computes full freshness by comparing a previous manifest with a freshly
    /// rebuilt filesystem manifest.
    ///
    /// # Arguments
    ///
    /// * `previous` - Manifest used as the freshness baseline.
    ///
    /// # Errors
    ///
    /// Returns an error when current indexing fails.
    pub fn freshness(&self, previous: &ProjectManifest) -> Result<FreshnessState> {
        Ok(self.evaluate_manifest_freshness(previous)?.state)
    }

    /// Computes hot-path freshness for files already listed in a manifest
    /// without walking or rebuilding the workspace manifest.
    ///
    /// This check is intentionally bounded to persisted manifest evidence: it
    /// detects changed and deleted indexed files by rehashing listed paths, but
    /// it does not discover newly added files. Full added-file discovery remains
    /// part of explicit indexing/sync through [`Self::freshness`].
    ///
    /// # Arguments
    ///
    /// * `previous` - Persisted manifest used as the hot-path evidence set.
    ///
    /// # Errors
    ///
    /// Returns an error when an indexed file exists but cannot be read or is no
    /// longer valid UTF-8.
    pub fn known_file_freshness(&self, previous: &ProjectManifest) -> Result<FreshnessState> {
        let mut changed = Vec::new();
        let mut deleted = Vec::new();
        let mut unchanged = Vec::new();

        for file in &previous.files {
            let path = self.root.join(&file.path);
            if !path.exists() {
                deleted.push(file.path.clone());
                continue;
            }
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let content = String::from_utf8(bytes).context("validated UTF-8 source content")?;
            if hash_text(&content) == file.content_hash {
                unchanged.push(file.path.clone());
            } else {
                changed.push(file.path.clone());
            }
        }

        changed.sort();
        deleted.sort();
        unchanged.sort();
        Ok(FreshnessState {
            fresh: changed.is_empty() && deleted.is_empty(),
            changed,
            deleted,
            added: Vec::new(),
            unchanged,
        })
    }
    /// Evaluates known manifest files without added-file discovery.
    ///
    /// # Arguments
    ///
    /// * `previous` - Persisted manifest used as the freshness baseline.
    ///
    /// # Errors
    ///
    /// Returns an error when known-file hashing fails.
    pub fn evaluate_known_file_freshness(
        &self,
        previous: &ProjectManifest,
    ) -> Result<ManifestFreshnessEvaluation> {
        Ok(ManifestFreshnessEvaluation {
            state: self.known_file_freshness(previous)?,
            proof_level: FreshnessProofLevel::IndexedFilesOnly,
        })
    }
}

fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)
        .with_context(|| format!("open {}", directory.display()))?
        .sync_all()
        .with_context(|| format!("sync {}", directory.display()))
}

fn build_file_nodes(files: &[SourceFile]) -> Vec<FileNode> {
    let mut nodes = BTreeMap::new();
    for file in files {
        let mut parent = None;
        let mut current = String::new();
        for part in file
            .path
            .split('/')
            .collect::<Vec<_>>()
            .split_last()
            .map(|(_, dirs)| dirs)
            .unwrap_or(&[])
        {
            current = if current.is_empty() {
                (*part).to_string()
            } else {
                format!("{current}/{part}")
            };
            nodes.entry(current.clone()).or_insert_with(|| FileNode {
                path: current.clone(),
                kind: FileNodeKind::Directory,
                parent: parent.clone(),
                provenance: provenance(&current, None, None, "file-tree", &current),
            });
            parent = Some(current.clone());
        }
        nodes.insert(
            file.path.clone(),
            FileNode {
                path: file.path.clone(),
                kind: FileNodeKind::File,
                parent,
                provenance: file.provenance.clone(),
            },
        );
    }
    nodes.into_values().collect()
}

fn build_shards(
    files: &[SourceFile],
    symbols: &[SymbolNode],
    root: &Path,
    strategy: &ShardStrategy,
) -> Result<Vec<ShardManifest>> {
    let mut shards = Vec::new();
    for file in files {
        let path = root.join(&file.path);
        let content = fs::read_to_string(&path)
            .with_context(|| format!("read shard source {}", path.display()))?;
        let lines = content.lines().collect::<Vec<_>>();
        let ranges = shard_ranges(file, symbols, strategy);
        for (start_line, end_line) in ranges {
            let start_index = usize::try_from(start_line.saturating_sub(1))
                .context("compute shard start index")?;
            let end_index = usize::try_from(end_line).context("compute shard end index")?;
            let shard_text = lines
                .get(start_index..end_index.min(lines.len()))
                .unwrap_or_default()
                .join("\n");
            let content_hash = hash_text(&shard_text);
            let id = format!("shard:{}:{}-{}", file.path, start_line, end_line);
            let symbol_ids = overlapping_symbol_ids(symbols, &file.path, start_line, end_line);
            shards.push(ShardManifest {
                id: id.clone(),
                path: file.path.clone(),
                start_line,
                end_line,
                content_hash: content_hash.clone(),
                symbol_ids,
                provenance: provenance(
                    &file.path,
                    Some(start_line),
                    Some(end_line),
                    "sharder",
                    &id,
                ),
            });
        }
    }
    shards.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(shards)
}

fn shard_ranges(
    file: &SourceFile,
    symbols: &[SymbolNode],
    strategy: &ShardStrategy,
) -> Vec<(u32, u32)> {
    match strategy {
        ShardStrategy::RustSemanticWithLineFallback if file.language == Language::Rust => {
            let mut semantic_ranges = symbols
                .iter()
                .filter(|symbol| symbol.path == file.path)
                .filter(|symbol| symbol.start_line > 0 && symbol.end_line >= symbol.start_line)
                .map(|symbol| (symbol.start_line, symbol.end_line.min(file.lines)))
                .collect::<Vec<_>>();
            semantic_ranges.sort_unstable();
            semantic_ranges.dedup();
            if semantic_ranges.is_empty() {
                fixed_line_ranges(file.lines, strategy.default_chunk_size())
            } else {
                semantic_ranges
            }
        }
        ShardStrategy::RustSemanticWithLineFallback | ShardStrategy::FixedLineChunks { .. } => {
            fixed_line_ranges(file.lines, strategy.default_chunk_size())
        }
    }
}

fn fixed_line_ranges(lines: u32, chunk_size: usize) -> Vec<(u32, u32)> {
    let chunk_size = u32::try_from(chunk_size.max(1)).unwrap_or(u32::MAX);
    let mut ranges = Vec::new();
    let mut start_line = 1u32;
    while start_line <= lines.max(1) {
        let end_line = start_line
            .saturating_add(chunk_size)
            .saturating_sub(1)
            .min(lines.max(1));
        ranges.push((start_line, end_line));
        let Some(next_start) = end_line.checked_add(1) else {
            break;
        };
        start_line = next_start;
    }
    ranges
}

fn overlapping_symbol_ids(
    symbols: &[SymbolNode],
    file_path: &str,
    start_line: u32,
    end_line: u32,
) -> Vec<String> {
    symbols
        .iter()
        .filter(|symbol| {
            symbol.path == file_path
                && ranges_overlap(start_line, end_line, symbol.start_line, symbol.end_line)
        })
        .map(|symbol| symbol.id.clone())
        .collect()
}

fn is_model_storage_path(path: &str) -> bool {
    path.contains(".forge_project_model/")
        || path.ends_with("project_manifest.json")
        || path.ends_with("tool_episodes.jsonl")
}

fn is_ignore_control_file(path: &str) -> bool {
    matches!(path, ".gitignore" | ".ignore" | ".git/info/exclude")
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        ContextPackSelection, GraphEdgeKind, RetrievalQuery, StaleEvidencePolicy, SymbolKind,
        compare_freshness, fingerprint, retrieve,
    };

    pub(crate) fn fixture_project() -> Result<(TempDir, PathBuf)> {
        let temp = TempDir::new()?;
        let root = temp.path().join("project");
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".ignore"), "ignored.rs\n")?;
        fs::write(root.join("ignored.rs"), "pub struct Ignored;\n")?;
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1\"\n",
        )?;
        fs::write(
            root.join("src").join("lib.rs"),
            "use serde::{Serialize, Deserialize};\npub use crate::model::Widget;\nmod model;\nextern crate core;\n\npub struct Root {\n    value: usize,\n}\n\nimpl Root {\n    pub fn new() -> Self {\n        Self { value: 0 }\n    }\n}\n\n#[test]\nfn root_test() {\n    assert_eq!(1, 1);\n}\n",
        )?;
        fs::write(
            root.join("src").join("model.rs"),
            "pub enum Widget {\n    One,\n}\n\npub trait Named {\n    fn name(&self) -> &str;\n}\n\nimpl Named for Widget {\n    fn name(&self) -> &str {\n        \"widget\"\n    }\n}\n",
        )?;
        Ok((temp, root))
    }

    #[test]
    fn indexes_manifest_with_ignore_hashes_shards_and_file_nodes() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let actual = setup.index()?;
        let expected = vec![
            "Cargo.toml".to_string(),
            "src/lib.rs".to_string(),
            "src/model.rs".to_string(),
        ];
        assert_eq!(
            actual
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            actual.files.iter().any(|file| file.path == "ignored.rs"),
            false
        );
        assert_eq!(
            actual
                .file_nodes
                .iter()
                .any(|node| node.path == "src" && node.kind == FileNodeKind::Directory),
            true
        );
        assert_eq!(
            actual.shards.iter().any(|shard| shard.path == "src/lib.rs"),
            true
        );
        assert_eq!(
            actual.shards.iter().any(|shard| shard.path == "src/lib.rs"
                && shard.start_line == 6
                && shard.end_line == 8),
            true
        );
        assert_eq!(actual.manifest_hash.len(), 64);
        Ok(())
    }

    #[test]
    fn manifest_freshness_evaluation_detects_changed_deleted_and_added_files() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(root.join("src").join("lib.rs"), "pub struct Changed;\n")?;
        fs::remove_file(root.join("src").join("model.rs"))?;
        fs::write(root.join("src").join("added.rs"), "pub fn added() {}\n")?;

        let actual = setup.evaluate_manifest_freshness(&previous)?;
        let expected = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: vec!["src/lib.rs".to_string()],
                deleted: vec!["src/model.rs".to_string()],
                added: vec!["src/added.rs".to_string()],
                unchanged: vec!["Cargo.toml".to_string()],
                fresh: false,
            },
            proof_level: FreshnessProofLevel::FullFilesystem,
        };
        assert_eq!(actual, expected);
        assert_eq!(actual.can_inject(), false);
        Ok(())
    }

    #[test]
    fn known_file_freshness_evaluation_does_not_overclaim_full_freshness() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(root.join("src").join("added.rs"), "pub fn added() {}\n")?;

        let actual = setup.evaluate_known_file_freshness(&previous)?;
        let expected = ManifestFreshnessEvaluation {
            state: FreshnessState {
                changed: Vec::new(),
                deleted: Vec::new(),
                added: Vec::new(),
                unchanged: vec![
                    "Cargo.toml".to_string(),
                    "src/lib.rs".to_string(),
                    "src/model.rs".to_string(),
                ],
                fresh: true,
            },
            proof_level: FreshnessProofLevel::IndexedFilesOnly,
        };
        assert_eq!(actual, expected);
        assert_eq!(actual.can_inject(), false);
        Ok(())
    }

    #[test]
    fn detects_changed_deleted_and_added_freshness() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(root.join("src").join("lib.rs"), "pub struct Changed;\n")?;
        fs::remove_file(root.join("src").join("model.rs"))?;
        fs::write(root.join("src").join("added.rs"), "pub fn added() {}\n")?;
        let current = setup.index()?;
        let actual = compare_freshness(&previous, &current);
        let expected = FreshnessState {
            changed: vec!["src/lib.rs".to_string()],
            deleted: vec!["src/model.rs".to_string()],
            added: vec!["src/added.rs".to_string()],
            unchanged: vec!["Cargo.toml".to_string()],
            fresh: false,
        };
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn known_file_freshness_checks_manifest_files_without_added_file_discovery() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(root.join("src").join("lib.rs"), "pub struct Changed;\n")?;
        fs::remove_file(root.join("src").join("model.rs"))?;
        fs::write(root.join("src").join("added.rs"), "pub fn added() {}\n")?;

        let actual = setup.known_file_freshness(&previous)?;
        let expected = FreshnessState {
            changed: vec!["src/lib.rs".to_string()],
            deleted: vec!["src/model.rs".to_string()],
            added: Vec::new(),
            unchanged: vec!["Cargo.toml".to_string()],
            fresh: false,
        };
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn appends_and_reads_tool_episode_jsonl() -> Result<()> {
        let fixture = TempDir::new()?;
        let setup = ProjectIndexer::new(fixture.path(), fixture.path().join("model"));
        let episode = ToolEpisode {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            tool: "tester".to_string(),
            input_fingerprint: fingerprint("secret input"),
            output_fingerprint: fingerprint("secret output"),
            status: "ok".to_string(),
            provenance: provenance("tool", None, None, "test", "episode"),
        };
        setup.append_episode(&episode)?;
        let actual = setup.read_episodes()?;
        let expected = vec![episode];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn serializes_manifest_with_provenance_deterministically() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let path = setup.write_manifest(&manifest)?;
        let actual = setup.read_manifest()?;
        let expected = manifest;
        assert_eq!(actual, expected);
        assert_eq!(path.ends_with("project_manifest.json"), true);
        Ok(())
    }

    fn fixture_context_pack(
        setup: &ProjectIndexer,
        manifest: &ProjectManifest,
    ) -> Result<ContextPack> {
        let query = RetrievalQuery {
            text: Some("Root".to_string()),
            path: None,
            path_prefix: None,
            symbol: Some("Root".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };
        ContextPack::from_selection(
            manifest,
            ContextPackSelection {
                retrieval_results: retrieve(manifest, &query),
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: setup.evaluate_manifest_freshness(manifest)?.state,
                stale_policy: StaleEvidencePolicy::Reject,
            },
        )
    }

    #[test]
    fn writes_context_pack_artifact_deterministically_and_round_trips() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let pack = fixture_context_pack(&setup, &manifest)?;

        let first_path = setup.write_context_pack(&pack)?;
        let first_bytes = fs::read(&first_path)?;
        let second_path = setup.write_context_pack(&pack)?;
        let second_bytes = fs::read(&second_path)?;
        let id = setup.context_pack_artifact_id(&pack)?;
        let actual = (
            first_path.clone(),
            second_path,
            first_bytes,
            second_bytes,
            setup.read_context_pack(&id)?,
            setup.list_context_pack_artifacts()?,
        );
        let expected = (
            first_path.clone(),
            first_path,
            actual.2.clone(),
            actual.2.clone(),
            pack,
            vec![id.clone()],
        );

        assert_eq!(actual, expected);
        assert_eq!(id.as_str().len(), 64);
        assert!(id.as_str().bytes().all(|byte| byte.is_ascii_hexdigit()));
        Ok(())
    }

    #[test]
    fn context_pack_artifact_rejects_empty_and_path_influenced_ids() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let empty = ContextPack::from_selection(
            &manifest,
            ContextPackSelection {
                retrieval_results: Vec::new(),
                shards: Vec::new(),
                evidence: Vec::new(),
                freshness: setup.evaluate_manifest_freshness(&manifest)?.state,
                stale_policy: StaleEvidencePolicy::Reject,
            },
        )?;

        let actual = setup.write_context_pack(&empty).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        assert!(setup.list_context_pack_artifacts()?.is_empty());
        assert!(ContextPackArtifactId::new("../escape".to_string()).is_err());
        assert!(ContextPackArtifactId::new("src/lib.rs".to_string()).is_err());
        Ok(())
    }

    #[test]
    fn read_context_pack_rejects_corrupt_or_mismatched_artifacts() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let pack = fixture_context_pack(&setup, &manifest)?;
        let id = setup.context_pack_artifact_id(&pack)?;
        let path = setup.write_context_pack(&pack)?;
        let mut mutated = pack.clone();
        mutated.manifest_hash = "different".to_string();
        fs::write(&path, mutated.to_stable_json()?)?;

        let actual = setup.read_context_pack(&id).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn fixture_includes_graph_retrieval_inputs() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: Some("Root".to_string()),
            path: None,
            path_prefix: None,
            symbol: Some("Root".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };
        let actual = retrieve(&manifest, &query);
        let expected = BTreeSet::from([GraphEdgeKind::Contains, GraphEdgeKind::CargoDependency]);
        assert_eq!(
            manifest
                .edges
                .iter()
                .map(|edge| edge.kind.clone())
                .filter(|kind| expected.contains(kind))
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            manifest
                .symbols
                .iter()
                .any(|symbol| symbol.kind == SymbolKind::Struct && symbol.name == "Root"),
            true
        );
        assert_eq!(
            actual
                .iter()
                .any(|result| result.symbol.as_deref() == Some("Root")),
            true
        );
        Ok(())
    }
}
