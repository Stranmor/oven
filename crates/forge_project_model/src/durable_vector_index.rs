//! Durable provider-neutral vector artifacts and in-memory search.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::types::{ProjectManifest, VectorQuery, VectorSearchHit};
use crate::util::hash_text;
use crate::vector::VectorIndex;

/// Current durable vector artifact format version.
pub const DURABLE_VECTOR_INDEX_VERSION: u32 = 1;

/// Hash-only deterministic identifier for a persisted vector index artifact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VectorIndexArtifactId(String);

impl VectorIndexArtifactId {
    /// Builds a vector index artifact identifier from a lowercase 64-character hex hash.
    ///
    /// # Arguments
    ///
    /// * `value` - Candidate lowercase SHA-256 hex string.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is not a lowercase hash-only artifact identifier.
    pub fn new(value: String) -> Result<Self> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("vector index artifact id must be a lowercase 64-character hex hash");
        }
        Ok(Self(value))
    }

    /// Returns the validated hash-only artifact identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for VectorIndexArtifactId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Durable provider-neutral vector artifact containing precomputed embeddings only.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorIndexArtifact {
    /// Artifact format version.
    pub version: u32,
    /// Manifest hash used to validate source identities and fingerprints.
    pub manifest_hash: String,
    /// Non-empty embedding model identity supplied by an external embedding boundary.
    pub embedding_model_id: String,
    /// Fixed embedding dimension for all vectors.
    pub dimension: usize,
    /// Vector entries sorted by identifier.
    pub entries: Vec<VectorIndexEntry>,
    /// Deterministic fingerprint over artifact fields excluding this fingerprint.
    pub index_fingerprint: String,
}

impl VectorIndexArtifact {
    /// Builds and validates a durable vector artifact from precomputed entries.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current project manifest that owns source identities.
    /// * `embedding_model_id` - Non-empty external embedding model identity.
    /// * `dimension` - Expected embedding dimension.
    /// * `entries` - Precomputed vector entries.
    ///
    /// # Errors
    ///
    /// Returns an error when any durable vector invariant is violated.
    pub fn new(
        manifest: &ProjectManifest,
        embedding_model_id: impl Into<String>,
        dimension: usize,
        mut entries: Vec<VectorIndexEntry>,
    ) -> Result<Self> {
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        let mut artifact = Self {
            version: DURABLE_VECTOR_INDEX_VERSION,
            manifest_hash: manifest.manifest_hash.clone(),
            embedding_model_id: embedding_model_id.into(),
            dimension,
            entries,
            index_fingerprint: String::new(),
        };
        artifact.index_fingerprint = artifact.compute_index_fingerprint()?;
        artifact.validate(manifest)?;
        Ok(artifact)
    }

    /// Validates this artifact against the current manifest.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current project manifest used as source evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is corrupt, stale, or not searchable.
    pub fn validate(&self, manifest: &ProjectManifest) -> Result<()> {
        if self.version != DURABLE_VECTOR_INDEX_VERSION {
            bail!(
                "vector index artifact version is unsupported: {}",
                self.version
            );
        }
        if self.manifest_hash != manifest.manifest_hash {
            bail!("vector index manifest hash mismatch");
        }
        if self.embedding_model_id.trim().is_empty() {
            bail!("vector index embedding model id must be non-empty");
        }
        if self.dimension == 0 {
            bail!("vector index dimension must be non-zero");
        }
        if self.entries.is_empty() {
            bail!("vector index requires non-empty entries");
        }
        for pair in self.entries.windows(2) {
            let [left, right] = pair else {
                continue;
            };
            if left.id >= right.id {
                bail!("vector index entries must be sorted by unique id");
            }
        }
        let evidence = ManifestVectorEvidence::from_manifest(manifest);
        for entry in &self.entries {
            entry.validate(self.dimension, &evidence)?;
        }
        let expected = self.compute_index_fingerprint()?;
        if self.index_fingerprint != expected {
            bail!("vector index fingerprint mismatch");
        }
        Ok(())
    }

    /// Serializes this artifact as stable pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when JSON serialization fails.
    pub fn to_stable_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Computes the hash-only artifact identifier from stable JSON content.
    ///
    /// # Errors
    ///
    /// Returns an error when stable JSON serialization fails.
    pub fn artifact_id(&self) -> Result<VectorIndexArtifactId> {
        VectorIndexArtifactId::new(hash_text(&self.to_stable_json()?))
    }

    fn compute_index_fingerprint(&self) -> Result<String> {
        let fingerprint_payload = VectorIndexFingerprintPayload {
            version: self.version,
            manifest_hash: &self.manifest_hash,
            embedding_model_id: &self.embedding_model_id,
            dimension: self.dimension,
            entries: &self.entries,
        };
        Ok(hash_text(&serde_json::to_string_pretty(
            &fingerprint_payload,
        )?))
    }
}

#[derive(Serialize)]
struct VectorIndexFingerprintPayload<'a> {
    version: u32,
    manifest_hash: &'a str,
    embedding_model_id: &'a str,
    dimension: usize,
    entries: &'a [VectorIndexEntry],
}

/// Durable source identity and precomputed vector entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorIndexEntry {
    /// Stable identifier matching a manifest file path, symbol id, or shard id.
    pub id: String,
    /// Source identity class for this vector.
    pub source_kind: VectorSourceKind,
    /// Relative source path that backs the vector evidence.
    pub source_path: String,
    /// Current manifest fingerprint for the source identity.
    pub source_fingerprint: String,
    /// Precomputed embedding vector.
    pub embedding: Vec<f32>,
}

impl VectorIndexEntry {
    /// Creates a durable vector entry from manifest source evidence and an embedding.
    ///
    /// # Arguments
    ///
    /// * `id` - Manifest file path, symbol id, or shard id.
    /// * `source_kind` - Source identity class.
    /// * `source_path` - Manifest source path associated with the id.
    /// * `source_fingerprint` - Current source fingerprint from the manifest.
    /// * `embedding` - Precomputed embedding vector.
    pub fn new(
        id: impl Into<String>,
        source_kind: VectorSourceKind,
        source_path: impl Into<String>,
        source_fingerprint: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            source_kind,
            source_path: source_path.into(),
            source_fingerprint: source_fingerprint.into(),
            embedding,
        }
    }

    fn validate(&self, dimension: usize, evidence: &ManifestVectorEvidence) -> Result<()> {
        if self.embedding.len() != dimension {
            bail!("vector entry dimension mismatch: {}", self.id);
        }
        validate_embedding_values(&self.embedding)
            .with_context(|| format!("vector entry embedding is invalid for id {}", self.id))?;
        let Some(expected) = evidence.entries.get(&self.id) else {
            bail!("vector entry id is absent from manifest: {}", self.id);
        };
        if expected.source_kind != self.source_kind {
            bail!("vector entry source kind mismatch: {}", self.id);
        }
        if expected.source_path != self.source_path {
            bail!("vector entry source path mismatch: {}", self.id);
        }
        if expected.source_fingerprint != self.source_fingerprint {
            bail!("vector entry source fingerprint is stale: {}", self.id);
        }
        Ok(())
    }
}

/// Manifest source class accepted by durable vector entries.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VectorSourceKind {
    /// Source file entry keyed by `files.path`.
    #[default]
    File,
    /// Symbol entry keyed by `symbols.id`.
    Symbol,
    /// Shard entry keyed by `shards.id`.
    Shard,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestVectorSource {
    source_kind: VectorSourceKind,
    source_path: String,
    source_fingerprint: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ManifestVectorEvidence {
    entries: BTreeMap<String, ManifestVectorSource>,
}

impl ManifestVectorEvidence {
    fn from_manifest(manifest: &ProjectManifest) -> Self {
        let mut entries = BTreeMap::new();
        for file in &manifest.files {
            entries.insert(
                file.path.clone(),
                ManifestVectorSource {
                    source_kind: VectorSourceKind::File,
                    source_path: file.path.clone(),
                    source_fingerprint: file.content_hash.clone(),
                },
            );
        }
        for symbol in &manifest.symbols {
            entries.insert(
                symbol.id.clone(),
                ManifestVectorSource {
                    source_kind: VectorSourceKind::Symbol,
                    source_path: symbol.path.clone(),
                    source_fingerprint: symbol.provenance.fingerprint.clone(),
                },
            );
        }
        for shard in &manifest.shards {
            entries.insert(
                shard.id.clone(),
                ManifestVectorSource {
                    source_kind: VectorSourceKind::Shard,
                    source_path: shard.path.clone(),
                    source_fingerprint: shard.content_hash.clone(),
                },
            );
        }
        Self { entries }
    }
}

/// Validated durable vector index with in-memory cosine search.
#[derive(Clone, Debug, PartialEq)]
pub struct DurableVectorIndex {
    artifact: VectorIndexArtifact,
}

impl DurableVectorIndex {
    /// Builds a searchable durable vector index from a validated artifact.
    ///
    /// # Arguments
    ///
    /// * `manifest` - Current manifest that owns source evidence.
    /// * `artifact` - Durable vector artifact to validate and search.
    ///
    /// # Errors
    ///
    /// Returns an error when artifact validation fails.
    pub fn new(manifest: &ProjectManifest, artifact: VectorIndexArtifact) -> Result<Self> {
        artifact.validate(manifest)?;
        Ok(Self { artifact })
    }

    /// Returns the embedding model identity associated with this index.
    pub fn embedding_model_id(&self) -> &str {
        &self.artifact.embedding_model_id
    }

    /// Returns the validated durable artifact.
    pub fn artifact(&self) -> &VectorIndexArtifact {
        &self.artifact
    }

    /// Searches the index with a precomputed query embedding and typed errors.
    ///
    /// # Arguments
    ///
    /// * `query` - Provider-neutral query embedding.
    ///
    /// # Errors
    ///
    /// Returns an error when the query dimension, values, or norm are invalid.
    pub fn search_validated(&self, query: &VectorQuery) -> Result<Vec<VectorSearchHit>> {
        if query.embedding.len() != self.artifact.dimension {
            bail!("vector query dimension mismatch");
        }
        validate_embedding_values(&query.embedding).context("vector query embedding is invalid")?;
        let mut hits = self
            .artifact
            .entries
            .iter()
            .map(|entry| VectorSearchHit {
                id: entry.id.clone(),
                score: cosine_similarity_unchecked(&query.embedding, &entry.embedding),
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(hits)
    }
}

impl VectorIndex for DurableVectorIndex {
    fn search(&self, query: &VectorQuery) -> Vec<VectorSearchHit> {
        self.search_validated(query).unwrap_or_default()
    }
}

fn validate_embedding_values(values: &[f32]) -> Result<()> {
    if values.is_empty() {
        bail!("embedding must be non-empty");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("embedding values must be finite");
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        bail!("embedding norm must be non-zero");
    }
    Ok(())
}

fn cosine_similarity_unchecked(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

/// Creates vector entries for tests and offline callers from manifest source evidence.
///
/// # Arguments
///
/// * `manifest` - Current project manifest that owns source identities.
/// * `embeddings` - Mapping from manifest id to precomputed embedding.
///
/// # Errors
///
/// Returns an error when any requested id is absent from manifest evidence.
pub fn vector_entries_from_manifest_embeddings(
    manifest: &ProjectManifest,
    embeddings: BTreeMap<String, Vec<f32>>,
) -> Result<Vec<VectorIndexEntry>> {
    let evidence = ManifestVectorEvidence::from_manifest(manifest);
    let mut entries = Vec::new();
    for (id, embedding) in embeddings {
        let Some(source) = evidence.entries.get(&id) else {
            bail!("vector entry id is absent from manifest: {}", id);
        };
        entries.push(VectorIndexEntry::new(
            id,
            source.source_kind.clone(),
            source.source_path.clone(),
            source.source_fingerprint.clone(),
            embedding,
        ));
    }
    let mut unique = BTreeSet::new();
    for entry in &entries {
        if !unique.insert(entry.id.clone()) {
            bail!("vector entry id is duplicated: {}", entry.id);
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use anyhow::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::ProjectIndexer;
    use crate::indexer::tests::fixture_project;

    fn fixture_artifact() -> Result<(tempfile::TempDir, ProjectManifest, VectorIndexArtifact)> {
        let (fixture, root) = fixture_project()?;
        let setup = ProjectIndexer::new(&root, fixture.path().join("model"));
        let manifest = setup.index()?;
        let root_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let model_symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Widget")
            .expect("fixture should include Widget symbol");
        let entries = vector_entries_from_manifest_embeddings(
            &manifest,
            BTreeMap::from([
                (model_symbol.id.clone(), vec![0.0, 1.0]),
                (root_symbol.id.clone(), vec![1.0, 0.0]),
            ]),
        )?;
        let artifact = VectorIndexArtifact::new(&manifest, "fixture-model", 2, entries)?;
        Ok((fixture, manifest, artifact))
    }

    #[test]
    fn vector_artifact_rejects_path_influenced_ids() {
        assert!(VectorIndexArtifactId::new("../escape".to_string()).is_err());
        assert!(VectorIndexArtifactId::new("src/lib.rs".to_string()).is_err());
        assert!(VectorIndexArtifactId::new("A".repeat(64)).is_err());
    }

    #[test]
    fn vector_artifact_rejects_duplicate_missing_stale_and_invalid_entries() -> Result<()> {
        let (_fixture, manifest, artifact) = fixture_artifact()?;
        let mut duplicate = artifact.clone();
        let first_duplicate_entry = duplicate
            .entries
            .first()
            .expect("fixture artifact should include entries")
            .clone();
        duplicate.entries.push(first_duplicate_entry);
        duplicate
            .entries
            .sort_by(|left, right| left.id.cmp(&right.id));
        duplicate.index_fingerprint = duplicate.compute_index_fingerprint()?;

        let mut missing = artifact.clone();
        missing
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .id = "missing".to_string();
        missing
            .entries
            .sort_by(|left, right| left.id.cmp(&right.id));
        missing.index_fingerprint = missing.compute_index_fingerprint()?;

        let mut stale = artifact.clone();
        stale
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .source_fingerprint = "stale".to_string();
        stale.index_fingerprint = stale.compute_index_fingerprint()?;

        let mut dimension = artifact.clone();
        dimension
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .embedding = vec![1.0, 0.0, 0.0];
        dimension.index_fingerprint = dimension.compute_index_fingerprint()?;

        let mut empty = artifact.clone();
        empty
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .embedding = Vec::new();
        empty.index_fingerprint = empty.compute_index_fingerprint()?;

        let mut zero_norm = artifact.clone();
        zero_norm
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .embedding = vec![0.0, 0.0];
        zero_norm.index_fingerprint = zero_norm.compute_index_fingerprint()?;

        let mut non_finite = artifact.clone();
        non_finite
            .entries
            .first_mut()
            .expect("fixture artifact should include entries")
            .embedding = vec![f32::NAN, 0.0];
        non_finite.index_fingerprint = non_finite.compute_index_fingerprint()?;

        let mut empty_model = artifact.clone();
        empty_model.embedding_model_id = String::new();
        empty_model.index_fingerprint = empty_model.compute_index_fingerprint()?;

        let mut empty_entries = artifact.clone();
        empty_entries.entries = Vec::new();
        empty_entries.index_fingerprint = empty_entries.compute_index_fingerprint()?;

        let actual = vec![
            duplicate.validate(&manifest).is_err(),
            missing.validate(&manifest).is_err(),
            stale.validate(&manifest).is_err(),
            dimension.validate(&manifest).is_err(),
            empty.validate(&manifest).is_err(),
            zero_norm.validate(&manifest).is_err(),
            non_finite.validate(&manifest).is_err(),
            empty_model.validate(&manifest).is_err(),
            empty_entries.validate(&manifest).is_err(),
        ];
        let expected = vec![true, true, true, true, true, true, true, true, true];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn vector_artifact_rejects_stale_file_symbol_and_shard_fingerprints() -> Result<()> {
        let (_fixture, manifest, _artifact) = fixture_artifact()?;
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == "src/lib.rs")
            .expect("fixture should include src/lib.rs");
        let symbol = manifest
            .symbols
            .iter()
            .find(|symbol| symbol.name == "Root")
            .expect("fixture should include Root symbol");
        let shard = manifest
            .shards
            .iter()
            .find(|shard| shard.path == "src/model.rs")
            .expect("fixture should include model shard");
        let entries = vec![
            VectorIndexEntry::new(
                file.path.clone(),
                VectorSourceKind::File,
                file.path.clone(),
                file.content_hash.clone(),
                vec![1.0, 0.0],
            ),
            VectorIndexEntry::new(
                symbol.id.clone(),
                VectorSourceKind::Symbol,
                symbol.path.clone(),
                symbol.provenance.fingerprint.clone(),
                vec![0.0, 1.0],
            ),
            VectorIndexEntry::new(
                shard.id.clone(),
                VectorSourceKind::Shard,
                shard.path.clone(),
                shard.content_hash.clone(),
                vec![1.0, 1.0],
            ),
        ];
        let artifact = VectorIndexArtifact::new(&manifest, "fixture-model", 2, entries)?;
        let actual = (0..artifact.entries.len())
            .map(|index| {
                let mut stale = artifact.clone();
                stale
                    .entries
                    .get_mut(index)
                    .expect("fixture artifact should include indexed entry")
                    .source_fingerprint = "stale".to_string();
                stale.index_fingerprint = stale.compute_index_fingerprint()?;
                Ok(stale.validate(&manifest).is_err())
            })
            .collect::<Result<Vec<_>>>()?;
        let expected = vec![true, true, true];
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn vector_artifact_serializes_without_provider_api_process_or_credential_fields() -> Result<()>
    {
        let (_fixture, _manifest, artifact) = fixture_artifact()?;
        let json = artifact.to_stable_json()?;
        let actual = [
            "provider",
            "api_key",
            "credential",
            "router",
            "endpoint",
            "process",
            "command",
        ]
        .iter()
        .any(|field| json.contains(field));
        let expected = false;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn durable_vector_index_search_rejects_invalid_queries_and_sorts_deterministically()
    -> Result<()> {
        let (_fixture, manifest, artifact) = fixture_artifact()?;
        let setup = DurableVectorIndex::new(&manifest, artifact)?;
        let actual = setup.search_validated(&VectorQuery { embedding: vec![1.0, 1.0] })?;
        let expected = actual.iter().map(|hit| hit.id.clone()).collect::<Vec<_>>();
        let mut sorted_expected = expected.clone();
        sorted_expected.sort();
        assert_eq!(expected, sorted_expected);
        assert!(
            setup
                .search_validated(&VectorQuery { embedding: vec![1.0] })
                .is_err()
        );
        assert!(
            setup
                .search_validated(&VectorQuery { embedding: vec![f32::INFINITY, 0.0] })
                .is_err()
        );
        assert!(
            setup
                .search_validated(&VectorQuery { embedding: vec![0.0, 0.0] })
                .is_err()
        );
        Ok(())
    }
}
