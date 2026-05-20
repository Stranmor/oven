//! Ignore-aware project indexing, persistence, sharding, and episodes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;

use crate::durable_vector_index::{VectorIndexArtifact, VectorIndexArtifactId};
use crate::extraction::{
    CargoManifestInput, extract_cargo_dependency_edges, extract_rust_import_edges,
    extract_rust_symbols, extract_static_cargo_metadata,
};
use crate::freshness::compare_freshness;
use crate::ingestion::ingest_external_fact_artifacts;
use crate::policy::{
    LOCAL_PROJECT_MODEL_EXTERNAL_FACT_REPORT_FILE_NAME, LOCAL_PROJECT_MODEL_MANIFEST_FILE_NAME,
};
use crate::types::{
    ContextPack, ContextPackArtifactId, ExternalFactArtifactIngestionReport,
    ExternalFactProductionBaseline, FileNode, FileNodeKind, FreshnessProofLevel, FreshnessState,
    Language, ManifestFreshnessEvaluation, ProjectManifest, ShardManifest, ShardStrategy,
    SourceFile, SymbolNode, ToolEpisode,
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
        let root = root.into();
        let model_dir = model_dir.into();
        let model_dir = if model_dir.is_absolute() {
            model_dir
        } else {
            root.join(model_dir)
        };
        Self { root, model_dir }
    }

    /// Returns the project root used by this indexer.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the project-model storage directory used by this indexer.
    ///
    /// The returned path is useful for typed artifact producers that must write
    /// through crate-owned persistence helpers without mutating a manifest.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Builds a deterministic project manifest from the configured root.
    ///
    /// # Errors
    ///
    /// Returns an error when walking, reading, parsing, or hashing project
    /// files fails.
    pub fn index(&self) -> Result<ProjectManifest> {
        Ok(self.index_with_external_fact_report()?.0)
    }

    /// Builds a deterministic project manifest and returns the external fact
    /// artifact ingestion report produced while indexing.
    ///
    /// # Errors
    ///
    /// Returns an error when base indexing fails or the model-dir external fact
    /// store cannot be listed.
    pub fn index_with_external_fact_report(
        &self,
    ) -> Result<(ProjectManifest, ExternalFactArtifactIngestionReport)> {
        let mut manifest = self.index_base_manifest()?;
        let report = ingest_external_fact_artifacts(&mut manifest, &self.external_facts_dir())?;
        Ok((manifest, report))
    }

    /// Applies durable external fact artifacts to a caller-provided frozen base manifest.
    ///
    /// This API is the no-second-walk ingestion path for future opt-in exact-fact
    /// wiring: callers pass the already built filesystem baseline and this method
    /// only reads the external fact artifact store before returning the final
    /// manifest and ingestion report. It does not rebuild the base manifest,
    /// discover newly added files, collect source text, or invoke producers.
    ///
    /// # Arguments
    ///
    /// * `frozen_manifest` - Base manifest previously produced by explicit indexing.
    ///
    /// # Errors
    ///
    /// Returns an error when the external fact artifact store cannot be listed or
    /// accepted artifacts cannot be applied to the cloned frozen manifest.
    pub fn ingest_external_fact_artifacts_from_manifest(
        &self,
        frozen_manifest: &ProjectManifest,
    ) -> Result<(ProjectManifest, ExternalFactArtifactIngestionReport)> {
        let mut manifest = frozen_manifest.clone();
        let report = ingest_external_fact_artifacts(&mut manifest, &self.external_facts_dir())?;
        Ok((manifest, report))
    }

    /// Builds a base project manifest and transient manifest-owned Rust source texts.
    ///
    /// The returned manifest is the base filesystem manifest before external
    /// fact artifact ingestion. Source texts are read only for Rust files already
    /// listed in that base manifest, are verified against manifest hashes, and
    /// are kept in memory without persistence or export side effects.
    ///
    /// # Errors
    ///
    /// Returns an error when base indexing fails, a manifest Rust path is not a
    /// safe project-relative path, the resolved file escapes the project root,
    /// the file is missing, unreadable, non-UTF-8, symlinked, or the source text
    /// fingerprint no longer matches the manifest file hash.
    pub fn external_fact_production_baseline(&self) -> Result<ExternalFactProductionBaseline> {
        let manifest = self.index_base_manifest()?;
        let rust_source_texts = self.collect_manifest_owned_rust_source_texts(&manifest)?;
        Ok(ExternalFactProductionBaseline { manifest, rust_source_texts })
    }

    fn collect_manifest_owned_rust_source_texts(
        &self,
        manifest: &ProjectManifest,
    ) -> Result<BTreeMap<String, String>> {
        let mut rust_source_texts = BTreeMap::new();
        for file in manifest
            .files
            .iter()
            .filter(|file| file.language == Language::Rust)
        {
            let source_text = self.read_manifest_owned_source_text(file)?;
            rust_source_texts.insert(file.path.clone(), source_text);
        }
        Ok(rust_source_texts)
    }

    fn read_manifest_owned_source_text(&self, file: &SourceFile) -> Result<String> {
        validate_manifest_relative_path(&file.path)?;
        let root = self
            .root
            .canonicalize()
            .with_context(|| format!("canonicalize project root {}", self.root.display()))?;
        let path = self.root.join(&file.path);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("inspect manifest source {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            anyhow::bail!("manifest Rust source path is a symlink: {}", file.path);
        }
        if !metadata.is_file() {
            anyhow::bail!("manifest Rust source path is not a file: {}", file.path);
        }
        let canonical_path = path
            .canonicalize()
            .with_context(|| format!("canonicalize manifest source {}", path.display()))?;
        if !canonical_path.starts_with(&root) {
            anyhow::bail!(
                "manifest Rust source path escapes project root: {}",
                file.path
            );
        }
        let bytes = fs::read(&canonical_path)
            .with_context(|| format!("read manifest source {}", canonical_path.display()))?;
        let source_text = String::from_utf8(bytes)
            .with_context(|| format!("manifest Rust source is not UTF-8: {}", file.path))?;
        let actual_hash = hash_text(&source_text);
        if actual_hash != file.content_hash {
            anyhow::bail!(
                "manifest Rust source hash mismatch for {}: expected {}, got {}",
                file.path,
                file.content_hash,
                actual_hash
            );
        }
        Ok(source_text)
    }

    fn index_base_manifest(&self) -> Result<ProjectManifest> {
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
            if path.starts_with(&self.model_dir) {
                continue;
            }
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
                cargo_tomls.push(CargoManifestInput { path: relative.clone(), content });
            }
            files.push(source);
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));

        let file_nodes = build_file_nodes(&files);
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        let known_files = files
            .iter()
            .map(|file| file.path.clone())
            .collect::<BTreeSet<_>>();
        let cargo_metadata = extract_static_cargo_metadata(&cargo_tomls, &known_files)?;
        for (path, content) in &rust_sources {
            let extracted = extract_rust_symbols(path, content)?;
            symbols.extend(extracted.symbols);
            edges.extend(extracted.edges);
            edges.extend(extract_rust_import_edges(path, content)?);
        }
        for cargo_manifest in &cargo_tomls {
            edges.extend(extract_cargo_dependency_edges(
                &cargo_manifest.path,
                &cargo_manifest.content,
            )?);
        }
        symbols.sort_by(|left, right| left.id.cmp(&right.id));
        edges.sort_by_key(edge_sort_key);
        let shards = build_shards(
            &files,
            &symbols,
            &self.root,
            &ShardStrategy::RustSemanticWithLineFallback,
        )?;
        let external_fact_batches = Vec::new();
        let external_facts_fingerprint =
            crate::util::external_facts_fingerprint(&external_fact_batches);
        let manifest_hash =
            manifest_hash(&files, &external_fact_batches, &external_facts_fingerprint);
        Ok(ProjectManifest {
            version: 1,
            root: self.root.clone(),
            files,
            file_nodes,
            symbols,
            cargo_workspace: cargo_metadata.workspace,
            cargo_packages: cargo_metadata.packages,
            cargo_package_dependencies: cargo_metadata.dependencies,
            edges,
            external_fact_batches,
            external_facts_fingerprint,
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

    /// Writes the redaction-safe external fact artifact ingestion report using stable pretty JSON.
    ///
    /// # Arguments
    ///
    /// * `report` - Ingestion report produced by `index_with_external_fact_report`.
    ///
    /// # Errors
    ///
    /// Returns an error when the model directory cannot be created or JSON
    /// cannot be written.
    pub fn write_external_fact_artifact_ingestion_report(
        &self,
        report: &ExternalFactArtifactIngestionReport,
    ) -> Result<PathBuf> {
        fs::create_dir_all(&self.model_dir).context("create model dir")?;
        let path = self
            .model_dir
            .join(LOCAL_PROJECT_MODEL_EXTERNAL_FACT_REPORT_FILE_NAME);
        let json =
            serde_json::to_string_pretty(report).context("serialize external fact report")?;
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

    /// Reads the deterministic external fact artifact ingestion report from storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the report cannot be read or decoded.
    pub fn read_external_fact_artifact_ingestion_report(
        &self,
    ) -> Result<ExternalFactArtifactIngestionReport> {
        let path = self
            .model_dir
            .join(LOCAL_PROJECT_MODEL_EXTERNAL_FACT_REPORT_FILE_NAME);
        let json = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&json).context("deserialize external fact report")
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

    /// Computes the deterministic artifact identifier for a vector index.
    ///
    /// # Arguments
    ///
    /// * `artifact` - Durable vector artifact to identify.
    ///
    /// # Errors
    ///
    /// Returns an error when stable JSON serialization fails.
    pub fn vector_index_artifact_id(
        &self,
        artifact: &VectorIndexArtifact,
    ) -> Result<VectorIndexArtifactId> {
        artifact.artifact_id()
    }

    /// Atomically writes a durable vector index artifact and validates readback.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current manifest used to validate source evidence.
    /// * `artifact` - Durable vector artifact to persist.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, atomic storage, directory sync, or
    /// readback validation fails.
    pub fn write_vector_index(
        &self,
        manifest: &ProjectManifest,
        artifact: &VectorIndexArtifact,
    ) -> Result<PathBuf> {
        artifact.validate(manifest)?;
        let id = self.vector_index_artifact_id(artifact)?;
        let directory = self.vector_index_dir();
        fs::create_dir_all(&directory).context("create vector index dir")?;
        let path = self.vector_index_path(&id);
        let json = artifact.to_stable_json()?;
        let temp_path = write_temp_artifact(&directory, id.as_str(), &json)?;
        fs::rename(&temp_path, &path)
            .with_context(|| format!("rename {} to {}", temp_path.display(), path.display()))?;
        sync_directory(&directory)?;
        let actual = self.read_vector_index(manifest, &id)?;
        if &actual != artifact {
            anyhow::bail!("vector index artifact readback mismatch: {}", id);
        }
        Ok(path)
    }

    /// Reads and validates a persisted vector index artifact by deterministic identifier.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current manifest used to validate source evidence.
    /// * `id` - Hash-only vector index artifact identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when read, decode, source validation, fingerprint, or
    /// content-derived identifier validation fails.
    pub fn read_vector_index(
        &self,
        manifest: &ProjectManifest,
        id: &VectorIndexArtifactId,
    ) -> Result<VectorIndexArtifact> {
        let path = self.vector_index_path(id);
        let json = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let artifact = serde_json::from_str::<VectorIndexArtifact>(&json)
            .context("deserialize vector index")?;
        artifact.validate(manifest)?;
        let actual_id = self.vector_index_artifact_id(&artifact)?;
        if &actual_id != id {
            anyhow::bail!(
                "vector index artifact id mismatch: expected {}, got {}",
                id,
                actual_id
            );
        }
        Ok(artifact)
    }

    /// Lists deterministic vector index artifact identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact directory cannot be listed or an
    /// artifact filename is not a valid hash-only identifier.
    pub fn list_vector_indexes(&self) -> Result<Vec<VectorIndexArtifactId>> {
        let directory = self.vector_index_dir();
        if !directory.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in
            fs::read_dir(&directory).with_context(|| format!("read {}", directory.display()))?
        {
            let entry = entry.context("read vector index artifact entry")?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
                anyhow::bail!(
                    "vector index artifact filename is not UTF-8: {}",
                    path.display()
                );
            };
            let Some(id) = file_name.strip_suffix(".json") else {
                continue;
            };
            ids.push(VectorIndexArtifactId::new(id.to_string())?);
        }
        ids.sort();
        Ok(ids)
    }

    fn external_facts_dir(&self) -> PathBuf {
        self.model_dir.join("external_facts")
    }

    fn vector_index_dir(&self) -> PathBuf {
        self.model_dir.join("vector_indexes")
    }

    fn vector_index_path(&self, id: &VectorIndexArtifactId) -> PathBuf {
        self.vector_index_dir()
            .join(format!("{}.json", id.as_str()))
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

fn validate_manifest_relative_path(path: &str) -> Result<()> {
    let candidate = Path::new(path);
    if path.is_empty() {
        anyhow::bail!("manifest source path is empty");
    }
    if candidate.is_absolute() {
        anyhow::bail!("manifest source path is absolute: {path}");
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                anyhow::bail!("manifest source path is not a safe relative path: {path}");
            }
        }
    }
    Ok(())
}

fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)
        .with_context(|| format!("open {}", directory.display()))?
        .sync_all()
        .with_context(|| format!("sync {}", directory.display()))
}

fn write_temp_artifact(directory: &Path, id: &str, json: &str) -> Result<PathBuf> {
    for attempt in 0..100u32 {
        let temp_path = directory.join(format!(".{id}.{}.{}.tmp", std::process::id(), attempt));
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
    anyhow::bail!("could not create unique artifact temp file");
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
        CargoDependencyDeclaration, CargoDependencyKind, CargoPackageDependency,
        CargoTargetDeclaration, CargoTargetKind, ContextPackSelection, EdgeConfidence,
        ExternalFactBatch, ExternalFactBatchMetadata, ExternalFactSource, GraphEdgeKind,
        RetrievalQuery, StaleEvidencePolicy, SymbolKind, TypedExternalFacts,
        TypedExternalReferenceFact, TypedExternalSymbolFact, VectorIndexArtifact,
        compare_freshness, external_fact_artifact_fingerprint, external_fact_batch_fingerprint,
        fingerprint, retrieve, vector_entries_from_manifest_embeddings,
        write_external_fact_artifact,
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

    fn external_artifact_batch(
        manifest: &ProjectManifest,
        source_label: &str,
        external_symbol_id: &str,
    ) -> ExternalFactBatch {
        let facts = TypedExternalFacts {
            symbols: vec![TypedExternalSymbolFact {
                id: external_symbol_id.to_string(),
                name: "external_new".to_string(),
                kind: SymbolKind::Method,
                path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 12,
                source: ExternalFactSource::Lsp,
            }],
            references: vec![TypedExternalReferenceFact {
                from: external_symbol_id.to_string(),
                to: "symbol:src/lib.rs:Struct:Root".to_string(),
                kind: GraphEdgeKind::References,
                path: "src/lib.rs".to_string(),
                start_line: Some(10),
                end_line: Some(10),
                source: ExternalFactSource::Lsp,
            }],
        };
        let mut batch = ExternalFactBatch {
            metadata: ExternalFactBatchMetadata {
                source: ExternalFactSource::Lsp,
                source_label: source_label.to_string(),
                tool_version: Some("fixture-1".to_string()),
                producer_snapshot_fingerprint: fingerprint("indexer-fixture-1"),
                workspace_root: manifest.root.to_string_lossy().to_string(),
                source_artifact_fingerprint: String::new(),
                manifest_hash_input: manifest.manifest_hash.clone(),
                batch_fingerprint: String::new(),
            },
            facts,
        };
        batch.metadata.source_artifact_fingerprint = external_fact_artifact_fingerprint(&batch);
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
        batch
    }

    fn write_external_artifact(
        setup: &ProjectIndexer,
        name: &str,
        batch: &ExternalFactBatch,
    ) -> Result<PathBuf> {
        let directory = setup.model_dir.join("external_facts");
        fs::create_dir_all(&directory)?;
        let path = directory.join(name);
        fs::write(&path, serde_json::to_string_pretty(batch)?)?;
        Ok(path)
    }

    #[test]
    fn external_fact_production_baseline_excludes_external_facts_while_index_ingests_them()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index_base_manifest()?;
        let batch =
            external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:baseline_external");
        write_external_artifact(&setup, "accepted.json", &batch)?;

        let actual = setup.external_fact_production_baseline()?;
        let normal = setup.index()?;

        assert_eq!(actual.manifest, base);
        assert_eq!(actual.manifest.external_fact_batches, Vec::new());
        assert_eq!(normal.external_fact_batches, vec![batch.metadata]);
        assert_eq!(
            normal
                .symbols
                .iter()
                .any(|symbol| symbol.id == "lsp:src/lib.rs:baseline_external"),
            true
        );
        Ok(())
    }

    #[test]
    fn external_fact_production_baseline_collects_only_manifest_rust_source_texts() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        fs::write(root.join("notes.md"), "# Notes\n")?;
        fs::create_dir_all(setup.model_dir.join("external_facts"))?;
        fs::write(
            setup.model_dir.join("external_facts").join("ignored.rs"),
            "pub struct Stored;\n",
        )?;

        let actual = setup.external_fact_production_baseline()?;
        let expected = BTreeSet::from(["src/lib.rs".to_string(), "src/model.rs".to_string()]);

        assert_eq!(
            actual
                .rust_source_texts
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            actual
                .manifest
                .files
                .iter()
                .any(|file| file.path == "notes.md" && file.language == Language::Markdown),
            true
        );
        assert_eq!(actual.rust_source_texts.contains_key("notes.md"), false);
        assert_eq!(actual.rust_source_texts.contains_key("ignored.rs"), false);
        assert_eq!(
            actual
                .rust_source_texts
                .iter()
                .all(|(path, source_text)| actual
                    .manifest
                    .files
                    .iter()
                    .any(|file| file.path == *path
                        && file.language == Language::Rust
                        && hash_text(source_text) == file.content_hash)),
            true
        );
        Ok(())
    }

    #[test]
    fn external_fact_production_baseline_rejects_manifest_source_hash_mismatch() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index_base_manifest()?;
        fs::write(root.join("src").join("lib.rs"), "pub struct Changed;\n")?;

        let actual = setup
            .collect_manifest_owned_rust_source_texts(&manifest)
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn external_fact_production_baseline_rejects_unsafe_manifest_paths() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index_base_manifest()?;
        let fixture_file = manifest
            .files
            .iter()
            .find(|file| file.path == "src/lib.rs")
            .expect("fixture should include src/lib.rs");
        let mut parent_escape = fixture_file.clone();
        parent_escape.path = "../escape.rs".to_string();
        let mut absolute_escape = fixture_file.clone();
        absolute_escape.path = root
            .join("src")
            .join("lib.rs")
            .to_string_lossy()
            .to_string();

        let actual = (
            setup
                .read_manifest_owned_source_text(&parent_escape)
                .is_err(),
            setup
                .read_manifest_owned_source_text(&absolute_escape)
                .is_err(),
        );
        let expected = (true, true);

        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn external_fact_production_baseline_rejects_manifest_symlink_source() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index_base_manifest()?;
        let fixture_file = manifest
            .files
            .iter()
            .find(|file| file.path == "src/lib.rs")
            .expect("fixture should include src/lib.rs");
        std::os::unix::fs::symlink(
            root.join("src").join("lib.rs"),
            root.join("src").join("link.rs"),
        )?;
        let mut symlink_file = fixture_file.clone();
        symlink_file.path = "src/link.rs".to_string();

        let actual = setup
            .read_manifest_owned_source_text(&symlink_file)
            .is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn external_fact_production_baseline_has_no_external_fact_store_side_effects() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let directory = setup.model_dir.join("external_facts");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("partial.json"), "{")?;

        let actual = (
            setup
                .external_fact_production_baseline()?
                .manifest
                .external_fact_batches,
            setup.model_dir.join("project_manifest.json").exists(),
            setup
                .model_dir
                .join("external_fact_artifact_ingestion_report.json")
                .exists(),
            fs::read_to_string(directory.join("partial.json"))?,
        );
        let expected = (Vec::new(), false, false, "{".to_string());

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn ingest_external_fact_artifacts_from_manifest_does_not_rewalk_filesystem() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model-no-second-walk"));
        let base = setup.index()?;
        let batch =
            external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:no_second_walk");
        write_external_artifact(&setup, "accepted.json", &batch)?;
        fs::write(
            root.join("src").join("added_after_base.rs"),
            "pub fn late() {}\n",
        )?;

        let (actual, report) = setup.ingest_external_fact_artifacts_from_manifest(&base)?;

        assert_eq!(report.accepted_artifacts, 1usize);
        assert_eq!(actual.files, base.files);
        assert_eq!(
            actual
                .files
                .iter()
                .any(|file| file.path == "src/added_after_base.rs"),
            false
        );
        assert_eq!(actual.external_fact_batches, vec![batch.metadata]);
        Ok(())
    }

    #[test]
    fn repeated_manifest_artifact_ingestion_is_idempotent_without_duplicate_edges_or_batches()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model-idempotent-ingest"));
        let base = setup.index()?;
        let batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:idempotent");
        write_external_artifact(&setup, "accepted.json", &batch)?;

        let (left, left_report) = setup.ingest_external_fact_artifacts_from_manifest(&base)?;
        let (right, right_report) = setup.ingest_external_fact_artifacts_from_manifest(&base)?;

        assert_eq!(left, right);
        assert_eq!(left_report.accepted_artifacts, 1usize);
        assert_eq!(right_report.accepted_artifacts, 1usize);
        assert_eq!(left.external_fact_batches.len(), 1usize);
        assert_eq!(
            left.edges
                .iter()
                .filter(|edge| edge.from == "lsp:src/lib.rs:idempotent")
                .count(),
            1usize
        );
        Ok(())
    }

    #[test]
    fn external_artifact_batch_is_accepted_and_produces_exact_edges_and_metadata() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let batch =
            external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:Root::external_new");
        write_external_artifact(&setup, "accepted.json", &batch)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 1usize);
        assert_eq!(actual.external_fact_batches, vec![batch.metadata.clone()]);
        assert_eq!(
            actual.edges.iter().any(|edge| {
                edge.from == "lsp:src/lib.rs:Root::external_new"
                    && edge.to == "symbol:src/lib.rs:Struct:Root"
                    && edge.confidence_kind == EdgeConfidence::ExactCompiler
            }),
            true
        );
        Ok(())
    }

    #[test]
    fn external_artifact_two_valid_same_base_batches_are_both_accepted() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let first = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:first");
        let second = external_artifact_batch(&base, "rust-analyzer-alt", "lsp:src/lib.rs:second");
        write_external_artifact(&setup, "b.json", &second)?;
        write_external_artifact(&setup, "a.json", &first)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 2usize);
        assert_eq!(actual.external_fact_batches.len(), 2usize);
        assert_eq!(
            actual
                .external_fact_batches
                .iter()
                .map(|metadata| metadata.manifest_hash_input.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([base.manifest_hash])
        );
        Ok(())
    }

    #[test]
    fn external_artifact_stale_base_hash_is_rejected_without_manifest_mutation() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let mut stale = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:stale");
        stale.metadata.manifest_hash_input = fingerprint("stale");
        stale.metadata.source_artifact_fingerprint = external_fact_artifact_fingerprint(&stale);
        stale.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&stale.metadata, &stale.facts);
        write_external_artifact(&setup, "stale.json", &stale)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(actual, base);
        Ok(())
    }

    #[test]
    fn external_artifact_fingerprint_mismatch_is_rejected_without_manifest_mutation() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let mut batch =
            external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:fingerprint");
        batch.metadata.source_artifact_fingerprint = fingerprint("wrong-artifact");
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
        write_external_artifact(&setup, "fingerprint.json", &batch)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(actual, base);
        Ok(())
    }

    #[test]
    fn external_artifact_endpoint_mismatch_is_rejected_without_manifest_mutation() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let mut batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:endpoint");
        batch
            .facts
            .references
            .first_mut()
            .expect("fixture should include reference")
            .to = "symbol:missing".to_string();
        batch.metadata.source_artifact_fingerprint = external_fact_artifact_fingerprint(&batch);
        batch.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&batch.metadata, &batch.facts);
        write_external_artifact(&setup, "endpoint.json", &batch)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(actual, base);
        Ok(())
    }

    #[test]
    fn external_artifact_directory_order_does_not_change_final_manifest_hash() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let left_setup = ProjectIndexer::new(&root, fixture.path().join("model-left"));
        let right_setup = ProjectIndexer::new(&root, fixture.path().join("model-right"));
        let base = left_setup.index()?;
        let left_first = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:first");
        let left_second =
            external_artifact_batch(&base, "rust-analyzer-alt", "lsp:src/lib.rs:second");
        let right_first = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:first");
        let right_second =
            external_artifact_batch(&base, "rust-analyzer-alt", "lsp:src/lib.rs:second");
        write_external_artifact(&left_setup, "a.json", &left_first)?;
        write_external_artifact(&left_setup, "b.json", &left_second)?;
        write_external_artifact(&right_setup, "b.json", &right_second)?;
        write_external_artifact(&right_setup, "a.json", &right_first)?;

        let left = left_setup.index()?;
        let right = right_setup.index()?;

        assert_eq!(left.manifest_hash, right.manifest_hash);
        assert_eq!(left.external_fact_batches, right.external_fact_batches);
        Ok(())
    }

    #[test]
    fn external_artifacts_under_model_storage_are_not_indexed_as_source_files() -> Result<()> {
        let (_fixture, root) = fixture_project()?;
        let model_dir = root.join(".forge_project_model");
        let setup = ProjectIndexer::new(&root, &model_dir);
        let base = setup.index()?;
        let batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:model_storage");
        write_external_artifact(&setup, "accepted.json", &batch)?;

        let actual = setup.index()?;

        assert_eq!(
            actual
                .files
                .iter()
                .any(|file| file.path.contains("external_facts")),
            false
        );
        assert_eq!(actual.external_fact_batches.len(), 1usize);
        Ok(())
    }

    #[test]
    fn external_artifacts_under_relative_model_storage_are_not_indexed_as_source_files()
    -> Result<()> {
        let (_fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, "relative-model");
        let base = setup.index()?;
        let batch = external_artifact_batch(
            &base,
            "rust-analyzer",
            "lsp:src/lib.rs:relative_model_storage",
        );
        write_external_artifact(&setup, "accepted.json", &batch)?;

        let actual = setup.index()?;

        assert_eq!(actual.files, base.files);
        assert_eq!(actual.external_fact_batches.len(), 1usize);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn external_artifact_store_reports_non_json_directory_and_symlink_without_accepting()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let directory = setup.model_dir.join("external_facts");
        fs::create_dir_all(directory.join("nested"))?;
        fs::write(directory.join("ignored.txt"), "not json")?;
        std::os::unix::fs::symlink(root.join("src").join("lib.rs"), directory.join("link.json"))?;

        let (actual, report) = setup.index_with_external_fact_report()?;
        let codes = report
            .artifacts
            .iter()
            .flat_map(|artifact| artifact.issues.iter().map(|issue| issue.code.clone()))
            .collect::<BTreeSet<_>>();

        assert_eq!(actual, base);
        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(
            codes,
            BTreeSet::from([
                crate::ExternalFactIngestionIssueCode::NonFileArtifact,
                crate::ExternalFactIngestionIssueCode::NonJsonArtifact,
                crate::ExternalFactIngestionIssueCode::SymlinkArtifact,
            ])
        );
        Ok(())
    }

    #[test]
    fn rejected_external_artifact_duplicate_symbol_does_not_poison_later_acceptance() -> Result<()>
    {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let first = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:first");
        let mut rejected =
            external_artifact_batch(&base, "rust-analyzer-rejected", "lsp:src/lib.rs:second");
        rejected.facts.symbols.push(TypedExternalSymbolFact {
            id: "lsp:src/lib.rs:first".to_string(),
            name: "first_duplicate".to_string(),
            kind: SymbolKind::Method,
            path: "src/lib.rs".to_string(),
            start_line: 10,
            end_line: 12,
            source: ExternalFactSource::Lsp,
        });
        rejected.metadata.source_artifact_fingerprint =
            external_fact_artifact_fingerprint(&rejected);
        rejected.metadata.batch_fingerprint =
            external_fact_batch_fingerprint(&rejected.metadata, &rejected.facts);
        let later_valid =
            external_artifact_batch(&base, "rust-analyzer-later", "lsp:src/lib.rs:second");
        write_external_artifact(&setup, "a-first.json", &first)?;
        write_external_artifact(&setup, "b-rejected.json", &rejected)?;
        write_external_artifact(&setup, "c-later-valid.json", &later_valid)?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 2usize);
        assert_eq!(actual.external_fact_batches.len(), 2usize);
        assert_eq!(
            actual
                .symbols
                .iter()
                .any(|symbol| symbol.id == "lsp:src/lib.rs:second"),
            true
        );
        Ok(())
    }

    #[test]
    fn external_artifact_exact_edge_participates_in_retrieval_graph_expansion() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let batch =
            external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:graph_neighbor");
        write_external_artifact(&setup, "graph.json", &batch)?;
        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: None,
            path: None,
            path_prefix: None,
            symbol: Some("external_new".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };

        let actual = retrieve(&manifest, &query)
            .into_iter()
            .map(|result| result.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(actual.contains("symbol:src/lib.rs:Struct:Root"), true);
        Ok(())
    }

    #[test]
    fn external_fact_artifact_writer_writes_artifact_accepted_by_indexer() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:writer");
        let expected = batch.metadata.clone();

        let actual_path = write_external_fact_artifact(&setup.model_dir, &base, batch)?;
        let (actual, report) = setup.index_with_external_fact_report()?;

        assert!(actual_path.is_file());
        assert_eq!(report.accepted_artifacts, 1usize);
        assert_eq!(actual.external_fact_batches, vec![expected]);
        assert_eq!(
            report
                .artifacts
                .first()
                .expect("accepted report should include artifact")
                .artifact_path,
            actual_path
                .file_name()
                .expect("artifact path should have filename")
                .to_string_lossy()
        );
        Ok(())
    }

    #[test]
    fn external_fact_artifact_writer_output_is_rejected_after_source_file_change() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:stale_writer");
        write_external_fact_artifact(&setup.model_dir, &base, batch)?;
        fs::write(
            root.join("src").join("lib.rs"),
            "use serde::{Serialize, Deserialize};\npub use crate::model::Widget;\nmod model;\nextern crate core;\n\npub struct Root {\n    value: usize,\n}\n\nimpl Root {\n    pub fn new() -> Self {\n        Self { value: 1 }\n    }\n}\n",
        )?;
        let current_base = setup.index_base_manifest()?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(actual, current_base);
        assert_eq!(
            report
                .artifacts
                .first()
                .expect("rejection report should include artifact")
                .issues
                .iter()
                .any(|issue| issue.code
                    == crate::ExternalFactIngestionIssueCode::ManifestBaselineMismatch),
            true
        );
        Ok(())
    }

    #[test]
    fn external_fact_artifact_writer_rejects_mixed_source_batch() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let mut batch = external_artifact_batch(&base, "rust-analyzer", "lsp:src/lib.rs:mixed");
        batch
            .facts
            .references
            .first_mut()
            .expect("fixture should include reference")
            .source = ExternalFactSource::Scip;

        let actual = write_external_fact_artifact(&setup.model_dir, &base, batch).is_err();
        let expected = true;

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn partial_external_fact_json_is_reported_and_not_accepted() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let directory = setup.model_dir.join("external_facts");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("partial.json"), "{")?;

        let (actual, report) = setup.index_with_external_fact_report()?;

        assert_eq!(actual, base);
        assert_eq!(report.accepted_artifacts, 0usize);
        assert_eq!(report.inspected_artifacts, 1usize);
        assert_eq!(
            report
                .artifacts
                .first()
                .and_then(|artifact| artifact.issues.first())
                .expect("partial artifact report should include parse issue")
                .code,
            crate::ExternalFactIngestionIssueCode::ArtifactParseFailed
        );
        Ok(())
    }

    #[test]
    fn writer_to_indexer_ingested_edge_participates_in_retrieval_graph_expansion() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let base = setup.index()?;
        let batch = external_artifact_batch(
            &base,
            "rust-analyzer",
            "lsp:src/lib.rs:writer_graph_neighbor",
        );
        write_external_fact_artifact(&setup.model_dir, &base, batch)?;
        let manifest = setup.index()?;
        let query = RetrievalQuery {
            text: None,
            path: None,
            path_prefix: None,
            symbol: Some("external_new".to_string()),
            limit: 5,
            include_graph_expansion: true,
        };

        let actual = retrieve(&manifest, &query)
            .into_iter()
            .map(|result| result.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(actual.contains("symbol:src/lib.rs:Struct:Root"), true);
        Ok(())
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
    fn cargo_single_package_indexes_static_package_targets_and_dependency_declarations()
    -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let actual = setup.index()?;
        let actual_package = actual
            .cargo_packages
            .iter()
            .find(|package| package.name == "fixture")
            .expect("fixture package should be indexed");
        let actual_dependency = actual
            .cargo_package_dependencies
            .iter()
            .find(|dependency| dependency.dependency_key == "serde")
            .expect("serde dependency should be indexed");

        assert_eq!(actual_package.manifest_path, "Cargo.toml");
        assert_eq!(actual_package.package_root, "");
        assert_eq!(actual_package.version.as_deref(), Some("0.1.0"));
        assert_eq!(actual_package.edition, None);
        assert_eq!(actual_package.provenance.is_complete(), true);
        assert_eq!(
            actual_package
                .targets
                .iter()
                .any(|target| target.kind == CargoTargetKind::Lib
                    && target.path == "src/lib.rs"
                    && target.declaration == CargoTargetDeclaration::ConventionInferred),
            true
        );
        assert_eq!(
            actual_dependency.declaring_package.as_deref(),
            Some("fixture")
        );
        assert_eq!(actual_dependency.kind, CargoDependencyKind::Normal);
        assert_eq!(
            actual_dependency.declaration,
            CargoDependencyDeclaration::DeclaredExternal
        );
        assert_eq!(actual_dependency.version.as_deref(), Some("1"));
        assert_eq!(actual_dependency.provenance.is_complete(), true);
        Ok(())
    }

    #[test]
    fn cargo_inferred_bin_target_preserves_hyphenated_package_name() -> Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path().join("project");
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"hyphen-pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )?;
        fs::write(root.join("src").join("main.rs"), "fn main() {}\n")?;
        let setup = ProjectIndexer::new(&root, temp.path().join("model"));

        let manifest = setup.index()?;
        let actual = manifest
            .cargo_packages
            .iter()
            .find(|package| package.name == "hyphen-pkg")
            .and_then(|package| {
                package
                    .targets
                    .iter()
                    .find(|target| target.kind == CargoTargetKind::Bin)
            })
            .map(|target| target.name.clone());
        let expected = Some("hyphen-pkg".to_string());

        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn cargo_workspace_indexes_members_and_member_packages_deterministically() -> Result<()> {
        let (left_fixture, left_root) = cargo_workspace_fixture(false)?;
        let (right_fixture, right_root) = cargo_workspace_fixture(true)?;
        let left = ProjectIndexer::new(&left_root, left_fixture.path().join("model")).index()?;
        let right = ProjectIndexer::new(&right_root, right_fixture.path().join("model")).index()?;
        let left_workspace = left
            .cargo_workspace
            .as_ref()
            .expect("workspace metadata should be indexed");
        let right_workspace = right
            .cargo_workspace
            .as_ref()
            .expect("workspace metadata should be indexed");

        assert_eq!(right_workspace.provenance.is_complete(), true);
        assert_eq!(
            left_workspace.members,
            vec!["crates/app".to_string(), "crates/util".to_string()]
        );
        assert_eq!(
            left_workspace.package_manifest_paths,
            vec![
                "crates/app/Cargo.toml".to_string(),
                "crates/util/Cargo.toml".to_string()
            ]
        );
        assert_eq!(
            serde_json::to_string(&left.cargo_workspace)?,
            serde_json::to_string(&right.cargo_workspace)?
        );
        assert_eq!(
            serde_json::to_string(&left.cargo_packages)?,
            serde_json::to_string(&right.cargo_packages)?
        );
        assert_eq!(
            serde_json::to_string(&left.cargo_package_dependencies)?,
            serde_json::to_string(&right.cargo_package_dependencies)?
        );
        Ok(())
    }

    #[test]
    fn cargo_dependencies_model_static_kinds_paths_workspace_target_renames_and_features()
    -> Result<()> {
        let (fixture, root) = cargo_workspace_fixture(false)?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let actual = setup.index()?;
        let path_dependency = dependency(&actual, "app", "util");
        let workspace_declaration = actual
            .cargo_package_dependencies
            .iter()
            .find(|dependency| {
                dependency.declaring_package.is_none() && dependency.dependency_key == "serde"
            })
            .expect("workspace serde declaration should be indexed");
        let inherited_dependency = dependency(&actual, "app", "serde");
        let renamed_dependency = dependency(&actual, "app", "anyhow_renamed");
        let optional_dependency = dependency(&actual, "app", "optional_dep");
        let dev_dependency = dependency(&actual, "app", "pretty_assertions");
        let build_dependency = dependency(&actual, "app", "cc");
        let target_dependency = dependency(&actual, "app", "cfg-if");
        let app_package = actual
            .cargo_packages
            .iter()
            .find(|package| package.name == "app")
            .expect("app package should be indexed");

        assert_eq!(
            path_dependency.declaration,
            CargoDependencyDeclaration::DeclaredPath
        );
        assert_eq!(path_dependency.path.as_deref(), Some("../util"));
        assert_eq!(
            path_dependency.linked_package_manifest_path.as_deref(),
            Some("crates/util/Cargo.toml")
        );
        assert_eq!(workspace_declaration.declaring_package, None);
        assert_eq!(workspace_declaration.features, vec!["derive".to_string()]);
        assert_eq!(
            inherited_dependency.declaration,
            CargoDependencyDeclaration::DeclaredWorkspaceInherited
        );
        assert_eq!(inherited_dependency.package_name, "serde");
        assert_eq!(renamed_dependency.dependency_key, "anyhow_renamed");
        assert_eq!(renamed_dependency.package_name, "anyhow");
        assert_eq!(optional_dependency.optional, true);
        assert_eq!(optional_dependency.features, vec!["feature-a".to_string()]);
        assert_eq!(
            app_package
                .features
                .iter()
                .any(|feature| feature.name == "extras"
                    && feature.members == vec!["optional_dep/feature-a".to_string()]),
            true
        );
        assert_eq!(dev_dependency.kind, CargoDependencyKind::Dev);
        assert_eq!(build_dependency.kind, CargoDependencyKind::Build);
        assert_eq!(target_dependency.target.as_deref(), Some("cfg(unix)"));
        assert_eq!(target_dependency.kind, CargoDependencyKind::Normal);
        assert_eq!(
            actual
                .edges
                .iter()
                .any(|edge| edge.kind == GraphEdgeKind::CargoDependency && edge.to == "serde"),
            true
        );
        Ok(())
    }

    #[test]
    fn cargo_manifest_hash_changes_when_static_declarations_change() -> Result<()> {
        let (fixture, root) = cargo_workspace_fixture(false)?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let previous = setup.index()?;
        fs::write(
            root.join("crates").join("app").join("Cargo.toml"),
            app_manifest("0.2"),
        )?;
        let current = setup.index()?;

        assert_ne!(previous.manifest_hash, current.manifest_hash);
        assert_eq!(
            current
                .files
                .iter()
                .any(|file| file.path == "crates/app/Cargo.toml"
                    && !previous
                        .files
                        .iter()
                        .any(|previous_file| previous_file.path == file.path
                            && previous_file.content_hash == file.content_hash)),
            true
        );
        Ok(())
    }

    fn dependency<'a>(
        manifest: &'a ProjectManifest,
        declaring_package: &str,
        dependency_key: &str,
    ) -> &'a CargoPackageDependency {
        manifest
            .cargo_package_dependencies
            .iter()
            .find(|dependency| {
                dependency.declaring_package.as_deref() == Some(declaring_package)
                    && dependency.dependency_key == dependency_key
            })
            .expect("dependency should be indexed")
    }

    fn cargo_workspace_fixture(reversed: bool) -> Result<(TempDir, PathBuf)> {
        let temp = TempDir::new()?;
        let root = temp.path().join("project");
        fs::create_dir_all(root.join("crates").join("app").join("src"))?;
        fs::create_dir_all(root.join("crates").join("util").join("src"))?;
        let members = if reversed {
            "members = [\"crates/util\", \"crates/app\"]"
        } else {
            "members = [\"crates/app\", \"crates/util\"]"
        };
        fs::write(
            root.join("Cargo.toml"),
            format!(
                "[workspace]\n{members}\n\n[workspace.dependencies]\nserde = {{ version = \"1\", features = [\"derive\"] }}\n"
            ),
        )?;
        fs::write(
            root.join("crates").join("app").join("Cargo.toml"),
            app_manifest("0.1"),
        )?;
        fs::write(
            root.join("crates").join("app").join("src").join("lib.rs"),
            "pub fn app() {}\n",
        )?;
        fs::write(
            root.join("crates").join("util").join("Cargo.toml"),
            "[package]\nname = \"util\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )?;
        fs::write(
            root.join("crates").join("util").join("src").join("lib.rs"),
            "pub fn util() {}\n",
        )?;
        Ok((temp, root))
    }

    fn app_manifest(version: &str) -> String {
        format!(
            "[package]\nname = \"app\"\nversion = \"{version}.0\"\nedition = \"2021\"\n\n[dependencies]\nserde = {{ workspace = true }}\nanyhow_renamed = {{ package = \"anyhow\", version = \"1\" }}\n\n[dependencies.util]\npath = \"../util\"\n\n[dependencies.optional_dep]\nversion = \"1\"\noptional = true\nfeatures = [\"feature-a\"]\n\n[dev-dependencies]\npretty_assertions = \"1\"\n\n[build-dependencies]\ncc = \"1\"\n\n[target.'cfg(unix)'.dependencies]\ncfg-if = \"1\"\n\n[features]\nextras = [\"optional_dep/feature-a\"]\n"
        )
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

    fn fixture_vector_index(
        manifest: &ProjectManifest,
        target_symbol: &str,
    ) -> Result<VectorIndexArtifact> {
        let symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == target_symbol)
            .expect("fixture should include target symbol");
        let entries = vector_entries_from_manifest_embeddings(
            manifest,
            BTreeMap::from([(symbol.id.clone(), vec![1.0, 0.0])]),
        )?;
        VectorIndexArtifact::new(manifest, "fixture-model", 2, entries)
    }

    #[test]
    fn writes_vector_index_artifact_deterministically_and_round_trips() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let artifact = fixture_vector_index(&manifest, "Widget")?;

        let first_path = setup.write_vector_index(&manifest, &artifact)?;
        let first_bytes = fs::read(&first_path)?;
        let second_path = setup.write_vector_index(&manifest, &artifact)?;
        let second_bytes = fs::read(&second_path)?;
        let id = setup.vector_index_artifact_id(&artifact)?;
        let actual = (
            first_path.clone(),
            second_path,
            first_bytes,
            second_bytes,
            setup.read_vector_index(&manifest, &id)?,
            setup.list_vector_indexes()?,
        );
        let expected = (
            first_path.clone(),
            first_path,
            actual.2.clone(),
            actual.2.clone(),
            artifact,
            vec![id.clone()],
        );
        assert_eq!(actual, expected);
        assert_eq!(id.as_str().len(), 64);
        assert!(id.as_str().bytes().all(|byte| byte.is_ascii_hexdigit()));
        Ok(())
    }

    #[test]
    fn read_vector_index_rejects_corrupt_or_mismatched_artifacts() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let artifact = fixture_vector_index(&manifest, "Widget")?;
        let id = setup.vector_index_artifact_id(&artifact)?;
        let path = setup.write_vector_index(&manifest, &artifact)?;
        let mut mutated = artifact.clone();
        mutated.manifest_hash = "different".to_string();
        fs::write(&path, mutated.to_stable_json()?)?;

        let actual = setup.read_vector_index(&manifest, &id).is_err();
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn vector_index_artifact_rejects_path_influenced_ids_from_listing() -> Result<()> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        fs::create_dir_all(setup.model_dir.join("vector_indexes"))?;
        fs::write(
            setup
                .model_dir
                .join("vector_indexes")
                .join("src-lib.rs.json"),
            "{}",
        )?;

        let actual = setup.list_vector_indexes().is_err();
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
