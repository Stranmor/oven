//! Typed stable/volatile project-model context cache partitioning.

use std::marker::PhantomData;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::fingerprint;

/// Stable schema version for project-model cache partition payloads.
pub const CACHE_PARTITION_SCHEMA_VERSION: u32 = 1;

/// Error returned when a stable project-model cache partition cannot be proven.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CachePartitionError {
    /// Manifest identity was absent or not proven fresh enough for stable injection.
    #[error("project-model stable cache partition blocked: {0}")]
    ManifestNotKnown(String),
    /// Stable source selection was empty.
    #[error("project-model stable cache partition blocked: no sources selected")]
    NoSourcesSelected,
    /// A selected source did not have exact readback content.
    #[error(
        "project-model stable cache partition blocked: source '{0}' has no verified readback content"
    )]
    UnverifiedSource(String),
    /// The supplied readback digest did not match the exact selected source range.
    #[error(
        "project-model stable cache partition blocked: source '{path}' digest mismatch: expected {expected}, actual {actual}"
    )]
    ReadbackDigestMismatch {
        /// Stable source path.
        path: String,
        /// Expected source-range digest.
        expected: String,
        /// Actual source-range digest.
        actual: String,
    },
    /// Stable render budget was exceeded before the overflow could be classified.
    #[error("project-model stable cache partition blocked: render budget overflow is unclassified")]
    BudgetOverflowUnclassified,
}

/// Typestate marker proving the manifest identity is known and fresh.
#[derive(Debug)]
pub struct ManifestKnown;

/// Typestate marker proving the ordered stable source list has been selected.
#[derive(Debug)]
pub struct SourcesSelected;

/// Typestate marker proving every selected source range was read back and digested.
#[derive(Debug)]
pub struct ReadbackVerified;

/// Typestate marker proving stable whitelisted bytes and identity are sealed.
#[derive(Debug)]
pub struct StablePayloadSealed;

/// Typestate marker proving a volatile sidecar was attached outside the stable bytes.
#[derive(Debug)]
pub struct VolatileSidecarAttached;

/// Cache partition state after manifest validation.
pub type CachePartitionManifestKnown = ProjectModelCachePartition<ManifestKnown>;
/// Cache partition state after source selection.
pub type CachePartitionSourcesSelected = ProjectModelCachePartition<SourcesSelected>;
/// Cache partition state after source readback verification.
pub type CachePartitionReadbackVerified = ProjectModelCachePartition<ReadbackVerified>;
/// Cache partition state after stable payload sealing.
pub type CachePartitionStablePayloadSealed = ProjectModelCachePartition<StablePayloadSealed>;
/// Cache partition state after volatile sidecar attachment.
pub type CachePartitionVolatileSidecarAttached =
    ProjectModelCachePartition<VolatileSidecarAttached>;

/// Stable source range selected for cache-eligible project-model context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelCachePartitionSource {
    /// Ordered relative source path or stable synthetic evidence path.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: Option<u32>,
    /// One-based inclusive end line.
    pub end_line: Option<u32>,
    /// Stable source node identifier.
    pub node_id: String,
    /// Exact readback text for the selected source range.
    pub readback: String,
    /// Expected source-range digest computed by the source owner.
    pub expected_readback_digest: String,
}

impl ProjectModelCachePartitionSource {
    /// Creates a stable selected source range with exact readback proof.
    ///
    /// # Arguments
    /// * `path` - Ordered relative source path or synthetic evidence path.
    /// * `node_id` - Stable source node identifier.
    /// * `readback` - Exact readback text for the selected source range.
    pub fn new(
        path: impl Into<String>,
        node_id: impl Into<String>,
        readback: impl Into<String>,
    ) -> Self {
        let readback = readback.into();
        let expected_readback_digest = fingerprint(&readback);
        Self {
            path: path.into(),
            start_line: None,
            end_line: None,
            node_id: node_id.into(),
            readback,
            expected_readback_digest,
        }
    }

    /// Attaches a one-based inclusive source range.
    ///
    /// # Arguments
    /// * `start_line` - First selected source line.
    /// * `end_line` - Last selected source line.
    pub fn line_range(mut self, start_line: u32, end_line: u32) -> Self {
        self.start_line = Some(start_line);
        self.end_line = Some(end_line);
        self
    }

    /// Overrides the expected readback digest for verification tests and callers
    /// that already own a source-range digest.
    ///
    /// # Arguments
    /// * `digest` - Expected source-range digest.
    pub fn expected_readback_digest(mut self, digest: impl Into<String>) -> Self {
        self.expected_readback_digest = digest.into();
        self
    }
}

/// Stable identity inputs for a sealed project-model payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelCachePartitionIdentity {
    /// Canonical project/repository identity.
    pub canonical_project_identity: String,
    /// Manifest schema version.
    pub manifest_schema_version: u32,
    /// Fresh manifest content hash.
    pub manifest_hash: String,
    /// Retrieval/query plan version.
    pub retrieval_plan_version: String,
    /// Renderer/template version.
    pub renderer_template_version: String,
    /// Optional AGENTS/project-rules digest when rules are semantically injected.
    pub agents_project_rules_digest: Option<String>,
    /// Render budget encoded into stable identity.
    pub render_budget: u32,
    /// Stable truncation policy label.
    pub truncation_policy: String,
    /// Cache-partition schema version.
    pub cache_partition_schema_version: u32,
}

/// Input required to enter the stable project-model cache partition typestate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelCachePartitionInput {
    /// Stable identity inputs for the project-model context pack.
    pub identity: ProjectModelCachePartitionIdentity,
}

impl ProjectModelCachePartitionInput {
    /// Validates manifest identity and enters the manifest-known typestate.
    ///
    /// # Errors
    /// Returns an error when the manifest hash or canonical identity is absent.
    pub fn manifest_known(self) -> Result<CachePartitionManifestKnown, CachePartitionError> {
        if self.identity.canonical_project_identity.trim().is_empty() {
            return Err(CachePartitionError::ManifestNotKnown(
                "canonical project identity is empty".to_string(),
            ));
        }
        if self.identity.manifest_hash.trim().is_empty() {
            return Err(CachePartitionError::ManifestNotKnown(
                "manifest hash is empty".to_string(),
            ));
        }
        Ok(ProjectModelCachePartition {
            identity: self.identity,
            sources: Vec::new(),
            stable_payload: None,
            volatile_sidecar: None,
            state: PhantomData,
        })
    }
}

/// Typed project-model cache partition builder.
pub struct ProjectModelCachePartition<State> {
    identity: ProjectModelCachePartitionIdentity,
    sources: Vec<ProjectModelCachePartitionSource>,
    stable_payload: Option<ProjectModelStablePayload>,
    volatile_sidecar: Option<ProjectModelVolatileSidecar>,
    state: PhantomData<State>,
}

impl ProjectModelCachePartition<ManifestKnown> {
    /// Selects the ordered stable source list.
    ///
    /// # Arguments
    /// * `sources` - Ordered sources selected by the retrieval/query plan.
    ///
    /// # Errors
    /// Returns an error when no stable source is selected.
    pub fn select_sources(
        self,
        sources: Vec<ProjectModelCachePartitionSource>,
    ) -> Result<CachePartitionSourcesSelected, CachePartitionError> {
        if sources.is_empty() {
            return Err(CachePartitionError::NoSourcesSelected);
        }
        Ok(ProjectModelCachePartition {
            identity: self.identity,
            sources,
            stable_payload: None,
            volatile_sidecar: None,
            state: PhantomData,
        })
    }
}

impl ProjectModelCachePartition<SourcesSelected> {
    /// Verifies every selected source range by comparing exact readback digests.
    ///
    /// # Errors
    /// Returns an error when any source lacks readback or has a digest mismatch.
    pub fn verify_readback(self) -> Result<CachePartitionReadbackVerified, CachePartitionError> {
        for source in &self.sources {
            if source.readback.is_empty() {
                return Err(CachePartitionError::UnverifiedSource(source.path.clone()));
            }
            let actual = fingerprint(&source.readback);
            if actual != source.expected_readback_digest {
                return Err(CachePartitionError::ReadbackDigestMismatch {
                    path: source.path.clone(),
                    expected: source.expected_readback_digest.clone(),
                    actual,
                });
            }
        }
        Ok(ProjectModelCachePartition {
            identity: self.identity,
            sources: self.sources,
            stable_payload: None,
            volatile_sidecar: None,
            state: PhantomData,
        })
    }
}

impl ProjectModelCachePartition<ReadbackVerified> {
    /// Seals cache-eligible stable payload bytes from whitelisted stable fields only.
    ///
    /// # Errors
    /// Returns an error when the sealed stable bytes exceed the render budget.
    pub fn seal_stable_payload(
        self,
    ) -> Result<CachePartitionStablePayloadSealed, CachePartitionError> {
        let fields = ProjectModelStablePayloadWhitelistedFields::new(
            self.identity.clone(),
            self.sources.clone(),
        );
        let bytes =
            serde_json::to_vec(&fields).expect("stable project-model payload is serializable");
        if bytes.len() > self.identity.render_budget as usize {
            return Err(CachePartitionError::BudgetOverflowUnclassified);
        }
        let stable_payload = ProjectModelStablePayload {
            identity: fingerprint(&String::from_utf8_lossy(&bytes)),
            bytes,
            fields,
        };
        Ok(ProjectModelCachePartition {
            identity: self.identity,
            sources: self.sources,
            stable_payload: Some(stable_payload),
            volatile_sidecar: None,
            state: PhantomData,
        })
    }
}

impl ProjectModelCachePartition<StablePayloadSealed> {
    /// Attaches uncached volatile sidecar data outside the stable payload bytes.
    ///
    /// # Arguments
    /// * `sidecar` - Volatile project-model context sidecar input.
    pub fn attach_volatile_sidecar(
        self,
        sidecar: ProjectModelVolatileSidecarInput,
    ) -> CachePartitionVolatileSidecarAttached {
        ProjectModelCachePartition {
            identity: self.identity,
            sources: self.sources,
            stable_payload: self.stable_payload,
            volatile_sidecar: Some(ProjectModelVolatileSidecar::from(sidecar)),
            state: PhantomData,
        }
    }

    /// Returns the sealed stable payload.
    pub fn stable_payload(&self) -> &ProjectModelStablePayload {
        self.stable_payload
            .as_ref()
            .expect("stable payload sealed typestate must contain payload")
    }
}

impl ProjectModelCachePartition<VolatileSidecarAttached> {
    /// Returns the sealed stable payload.
    pub fn stable_payload(&self) -> &ProjectModelStablePayload {
        self.stable_payload
            .as_ref()
            .expect("volatile sidecar attached typestate must contain stable payload")
    }

    /// Returns the attached volatile sidecar.
    pub fn volatile_sidecar(&self) -> &ProjectModelVolatileSidecar {
        self.volatile_sidecar
            .as_ref()
            .expect("volatile sidecar attached typestate must contain sidecar")
    }
}

/// Stable cache-eligible project-model payload bytes and identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectModelStablePayload {
    /// Stable cache identity derived from stable payload bytes.
    pub identity: String,
    /// Exact stable payload bytes.
    pub bytes: Vec<u8>,
    /// Whitelisted stable fields used to produce the bytes.
    pub fields: ProjectModelStablePayloadWhitelistedFields,
}

/// Stable payload fields allowed inside the cache-eligible partition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelStablePayloadWhitelistedFields {
    /// Cache-partition schema version.
    pub cache_partition_schema_version: u32,
    /// Canonical project/repository identity.
    pub canonical_project_identity: String,
    /// Manifest schema version.
    pub manifest_schema_version: u32,
    /// Fresh manifest content hash.
    pub manifest_hash: String,
    /// Retrieval/query plan version.
    pub retrieval_plan_version: String,
    /// Renderer/template version.
    pub renderer_template_version: String,
    /// Optional AGENTS/project-rules digest when semantically injected.
    pub agents_project_rules_digest: Option<String>,
    /// Ordered selected source list with readback digest per range.
    pub ordered_sources: Vec<ProjectModelStableSourceIdentity>,
    /// Render budget encoded into stable identity.
    pub render_budget: u32,
    /// Stable truncation policy label.
    pub truncation_policy: String,
}

impl ProjectModelStablePayloadWhitelistedFields {
    fn new(
        identity: ProjectModelCachePartitionIdentity,
        sources: Vec<ProjectModelCachePartitionSource>,
    ) -> Self {
        Self {
            cache_partition_schema_version: identity.cache_partition_schema_version,
            canonical_project_identity: identity.canonical_project_identity,
            manifest_schema_version: identity.manifest_schema_version,
            manifest_hash: identity.manifest_hash,
            retrieval_plan_version: identity.retrieval_plan_version,
            renderer_template_version: identity.renderer_template_version,
            agents_project_rules_digest: identity.agents_project_rules_digest,
            ordered_sources: sources
                .into_iter()
                .map(ProjectModelStableSourceIdentity::from)
                .collect(),
            render_budget: identity.render_budget,
            truncation_policy: identity.truncation_policy,
        }
    }
}

/// Stable identity for one selected source range.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelStableSourceIdentity {
    /// Ordered relative source path or stable synthetic evidence path.
    pub path: String,
    /// One-based inclusive start line.
    pub start_line: Option<u32>,
    /// One-based inclusive end line.
    pub end_line: Option<u32>,
    /// Stable source node identifier.
    pub node_id: String,
    /// Verified source-range readback digest.
    pub readback_digest: String,
}

impl From<ProjectModelCachePartitionSource> for ProjectModelStableSourceIdentity {
    fn from(source: ProjectModelCachePartitionSource) -> Self {
        Self {
            path: source.path,
            start_line: source.start_line,
            end_line: source.end_line,
            node_id: source.node_id,
            readback_digest: source.expected_readback_digest,
        }
    }
}

/// Volatile project-model sidecar input that must never affect stable bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelVolatileSidecarInput {
    /// Live freshness/readiness/semantic diagnostics.
    pub diagnostics: Vec<String>,
    /// Current request time if supplied by the runtime boundary.
    pub current_time: Option<String>,
    /// Runtime context label.
    pub runtime_context: Option<String>,
    /// Session identifier.
    pub session_id: Option<String>,
    /// Model/provider route.
    pub model_provider_route: Option<String>,
    /// Token counters or estimates.
    pub token_counters: Option<String>,
    /// Latency diagnostics.
    pub latency: Option<String>,
    /// Health probe diagnostics.
    pub health_probe: Option<String>,
    /// Readiness warnings.
    pub readiness_warnings: Vec<String>,
    /// Mutable file freshness diagnostics.
    pub mutable_file_freshness: Vec<String>,
    /// Transient errors.
    pub transient_errors: Vec<String>,
}

/// Volatile project-model sidecar rendered separately from stable payload bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectModelVolatileSidecar {
    /// Volatile fields that are explicitly uncached.
    pub input: ProjectModelVolatileSidecarInput,
}

impl From<ProjectModelVolatileSidecarInput> for ProjectModelVolatileSidecar {
    fn from(input: ProjectModelVolatileSidecarInput) -> Self {
        Self { input }
    }
}

/// Stable and volatile context messages produced by the project-model partitioner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableProjectModelContextMessage {
    /// Cache-eligible stable payload message bytes.
    pub stable_payload: ProjectModelStablePayload,
    /// Uncached volatile sidecar message.
    pub volatile_sidecar: ProjectModelVolatileSidecar,
}

impl From<CachePartitionVolatileSidecarAttached> for StableProjectModelContextMessage {
    fn from(partition: CachePartitionVolatileSidecarAttached) -> Self {
        Self {
            stable_payload: partition
                .stable_payload
                .expect("volatile sidecar attached typestate must contain stable payload"),
            volatile_sidecar: partition
                .volatile_sidecar
                .expect("volatile sidecar attached typestate must contain sidecar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn identity() -> ProjectModelCachePartitionIdentity {
        ProjectModelCachePartitionIdentity {
            canonical_project_identity: "repo://example".to_string(),
            manifest_schema_version: 7,
            manifest_hash: "manifest-a".to_string(),
            retrieval_plan_version: "retrieval-v1".to_string(),
            renderer_template_version: "renderer-v1".to_string(),
            agents_project_rules_digest: Some("rules-a".to_string()),
            render_budget: 4096,
            truncation_policy: "metadata-only-on-overflow".to_string(),
            cache_partition_schema_version: CACHE_PARTITION_SCHEMA_VERSION,
        }
    }

    fn source(path: &str, readback: &str) -> ProjectModelCachePartitionSource {
        ProjectModelCachePartitionSource::new(path, format!("node:{path}"), readback)
            .line_range(1, 3)
    }

    fn seal(
        identity: ProjectModelCachePartitionIdentity,
        sources: Vec<ProjectModelCachePartitionSource>,
    ) -> ProjectModelStablePayload {
        ProjectModelCachePartitionInput { identity }
            .manifest_known()
            .unwrap()
            .select_sources(sources)
            .unwrap()
            .verify_readback()
            .unwrap()
            .seal_stable_payload()
            .unwrap()
            .stable_payload()
            .clone()
    }

    #[test]
    fn identical_stable_inputs_produce_identical_bytes_and_identity() {
        let setup = (identity(), vec![source("src/lib.rs", "fn main() {}")]);

        let actual = (
            seal(setup.0.clone(), setup.1.clone()),
            seal(setup.0.clone(), setup.1.clone()),
        );

        let expected = true;
        assert_eq!(
            actual.0.bytes == actual.1.bytes && actual.0.identity == actual.1.identity,
            expected
        );
    }

    #[test]
    fn volatile_sidecar_changes_do_not_change_stable_payload_or_identity() {
        let sealed = ProjectModelCachePartitionInput { identity: identity() }
            .manifest_known()
            .unwrap()
            .select_sources(vec![source("src/lib.rs", "fn main() {}")])
            .unwrap()
            .verify_readback()
            .unwrap()
            .seal_stable_payload()
            .unwrap();
        let stable_before = sealed.stable_payload().clone();

        let actual = sealed
            .attach_volatile_sidecar(ProjectModelVolatileSidecarInput {
                current_time: Some("2026-05-21T00:00:00+03:00".to_string()),
                diagnostics: vec!["semantic_vector_state=ready".to_string()],
                ..Default::default()
            })
            .stable_payload()
            .clone();

        let expected = stable_before;
        assert_eq!(actual, expected);
    }

    #[test]
    fn manifest_hash_change_alters_stable_identity() {
        let mut changed = identity();
        changed.manifest_hash = "manifest-b".to_string();

        let actual = (
            seal(identity(), vec![source("src/lib.rs", "fn main() {}")]).identity,
            seal(changed, vec![source("src/lib.rs", "fn main() {}")]).identity,
        );

        let expected = false;
        assert_eq!(actual.0 == actual.1, expected);
    }

    #[test]
    fn source_order_change_alters_stable_identity() {
        let actual = (
            seal(identity(), vec![source("a.rs", "a"), source("b.rs", "b")]).identity,
            seal(identity(), vec![source("b.rs", "b"), source("a.rs", "a")]).identity,
        );

        let expected = false;
        assert_eq!(actual.0 == actual.1, expected);
    }

    #[test]
    fn render_budget_change_alters_stable_identity() {
        let mut changed = identity();
        changed.render_budget = 8192;

        let actual = (
            seal(identity(), vec![source("src/lib.rs", "fn main() {}")]).identity,
            seal(changed, vec![source("src/lib.rs", "fn main() {}")]).identity,
        );

        let expected = false;
        assert_eq!(actual.0 == actual.1, expected);
    }

    #[test]
    fn renderer_schema_change_alters_stable_identity() {
        let mut changed = identity();
        changed.renderer_template_version = "renderer-v2".to_string();

        let actual = (
            seal(identity(), vec![source("src/lib.rs", "fn main() {}")]).identity,
            seal(changed, vec![source("src/lib.rs", "fn main() {}")]).identity,
        );

        let expected = false;
        assert_eq!(actual.0 == actual.1, expected);
    }

    #[test]
    fn unknown_manifest_cannot_construct_cache_eligible_payload() {
        let mut unknown = identity();
        unknown.manifest_hash.clear();

        let actual = ProjectModelCachePartitionInput { identity: unknown }.manifest_known();

        assert!(actual.is_err());
    }

    #[test]
    fn readback_digest_mismatch_blocks_stable_payload_sealing() {
        let setup = source("src/lib.rs", "fn main() {}").expected_readback_digest("wrong-digest");

        let actual = ProjectModelCachePartitionInput { identity: identity() }
            .manifest_known()
            .unwrap()
            .select_sources(vec![setup])
            .unwrap()
            .verify_readback();

        assert!(matches!(
            actual,
            Err(CachePartitionError::ReadbackDigestMismatch { .. })
        ));
    }
}
