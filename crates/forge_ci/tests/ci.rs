use std::fs;
use std::path::PathBuf;

use forge_ci::workflows as workflow;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn workflow_text(path: &str) -> String {
    fs::read_to_string(repo_root().join(path)).unwrap()
}

fn assert_workflow_is_read_only_dispatch(path: &str) {
    let fixture = workflow_text(path);

    assert!(fixture.contains("workflow_dispatch: {}"));
    assert!(fixture.contains("contents: read"));
    assert!(!fixture.contains("contents: write"));
    assert!(!fixture.contains("issues: write"));
    assert!(!fixture.contains("pull-requests: write"));
    assert!(!fixture.contains("pull_request_target"));
    assert!(!fixture.contains("release-drafter/release-drafter"));
    assert!(!fixture.contains("actions/stale"));
    assert!(!fixture.contains("antinomyhq/"));
    assert!(!fixture.contains("tailcallhq/"));
}

fn assert_generator_source_has_no_disabled_mutators(path: &str) {
    let fixture = workflow_text(path);

    assert!(!fixture.contains("contents(Level::Write)"));
    assert!(!fixture.contains("issues(Level::Write)"));
    assert!(!fixture.contains("pull_requests(Level::Write)"));
    assert!(!fixture.contains("pull_request_target"));
    assert!(!fixture.contains("release-drafter"));
    assert!(!fixture.contains("actions/stale"));
    assert!(!fixture.contains("antinomyhq/"));
    assert!(!fixture.contains("NPM_TOKEN"));
    assert!(!fixture.contains("HOMEBREW_ACCESS"));
}

#[test]
fn generate() {
    workflow::generate_ci_workflow();
}

#[test]
fn test_release_drafter() {
    workflow::generate_release_drafter_workflow();

    assert_workflow_is_read_only_dispatch(".github/workflows/release-drafter.yml");
}

#[test]
fn test_release_workflow() {
    workflow::release_publish();

    let fixture = workflow_text(".github/workflows/release.yml");

    assert_workflow_is_read_only_dispatch(".github/workflows/release.yml");
    assert!(!fixture.contains("release:"));
    assert!(!fixture.contains("npm_release"));
    assert!(!fixture.contains("homebrew_release"));
    assert!(!fixture.contains("NPM_TOKEN"));
    assert!(!fixture.contains("HOMEBREW_ACCESS"));
}

#[test]
fn test_labels_workflow() {
    workflow::generate_labels_workflow();

    assert_workflow_is_read_only_dispatch(".github/workflows/labels.yml");
}

#[test]
fn test_stale_workflow() {
    workflow::generate_stale_workflow();

    assert_workflow_is_read_only_dispatch(".github/workflows/stale.yml");
}

#[test]
fn test_autofix_workflow() {
    workflow::generate_autofix_workflow();
}

#[test]
fn test_bounty_workflow() {
    workflow::generate_bounty_workflow();

    assert_workflow_is_read_only_dispatch(".github/workflows/bounty.yml");
}

#[test]
fn test_disabled_generator_sources_do_not_keep_mutating_job_builders() {
    for path in [
        "crates/forge_ci/src/jobs/bounty_job.rs",
        "crates/forge_ci/src/jobs/build.rs",
        "crates/forge_ci/src/jobs/draft_release_update_job.rs",
        "crates/forge_ci/src/jobs/release_homebrew.rs",
        "crates/forge_ci/src/jobs/release_npm.rs",
    ] {
        assert_generator_source_has_no_disabled_mutators(path);
    }
}
