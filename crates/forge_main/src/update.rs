use std::sync::Arc;

use colored::Colorize;
use forge_api::API;
use forge_config::{Update, UpdateFrequency};
use forge_select::ForgeWidget;
use forge_tracker::VERSION;
use update_informer::{Check, Version, registry};

/// Local fork checkout that owns the active Forge installation.
const LOCAL_FORGE_SOURCE: &str = "/home/stranmor/Documents/project/_mycelium/oven";
const FORGE_REAL_TARGET: &str = "/home/stranmor/.local/lib/forge/forge-real";
const FORGE_WRAPPER_ENTRYPOINT: &str = "/home/stranmor/.local/bin/forge";
const UPDATE_REPOSITORY: &str = "Stranmor/oven";

fn local_fork_update_command() -> String {
    format!(
        "set -eu; repo={repo:?}; target={target:?}; wrapper={wrapper:?}; \
         test -d \"$repo/.git\"; \
         origin_url=\"$(git -C \"$repo\" remote get-url origin)\"; \
         test \"$origin_url\" = \"git@github.com:Stranmor/oven.git\" -o \
              \"$origin_url\" = \"https://github.com/Stranmor/oven.git\"; \
         git -C \"$repo\" fetch origin main:refs/remotes/origin/main; \
         test \"$(git -C \"$repo\" rev-parse --abbrev-ref HEAD)\" = \"main\"; \
         test \"$(git -C \"$repo\" rev-parse HEAD)\" = \
              \"$(git -C \"$repo\" rev-parse refs/remotes/origin/main)\"; \
         test -z \"$(git -C \"$repo\" status --porcelain)\"; \
         test -L \"$wrapper\"; \
         test \"$(readlink -f \"$wrapper\")\" != \"$target\"; \
         cargo build --release --manifest-path \"$repo/Cargo.toml\" -p forge_main; \
         mkdir -p \"$(dirname \"$target\")\"; \
         tmp=\"$(mktemp \"$(dirname \"$target\")/.forge-real.XXXXXX\")\"; \
         cp \"$repo/target/release/forge\" \"$tmp\"; \
         chmod 755 \"$tmp\"; \
         mv -f \"$tmp\" \"$target\"",
        repo = LOCAL_FORGE_SOURCE,
        target = FORGE_REAL_TARGET,
        wrapper = FORGE_WRAPPER_ENTRYPOINT,
    )
}

/// Builds Forge from the local Stranmor/oven fork and atomically replaces the
/// real binary target behind the wrapper, failing silently.
/// When `auto_update` is true, exits immediately after a successful update
/// without prompting the user.
async fn execute_update_command(api: Arc<impl API>, auto_update: bool) {
    let command = local_fork_update_command();
    let output = api.execute_shell_command_raw(&command).await;

    match output {
        Err(err) => {
            // Send an event to the tracker on failure
            // We don't need to handle this result since we're failing silently
            let _ = send_update_failure_event(&format!("Auto update failed {err}")).await;
        }
        Ok(output) => {
            if output.success() {
                let should_exit = if auto_update {
                    true
                } else {
                    let answer = forge_select::ForgeWidget::confirm(
                        "You need to close forge to complete update. Do you want to close it now?",
                    )
                    .with_default(true)
                    .prompt();
                    answer.unwrap_or_default().unwrap_or_default()
                };
                if should_exit {
                    std::process::exit(0);
                }
            } else {
                let exit_output = match output.code() {
                    Some(code) => format!("Process exited with code: {code}"),
                    None => "Process exited without code".to_string(),
                };
                let _ =
                    send_update_failure_event(&format!("Auto update failed, {exit_output}",)).await;
            }
        }
    }
}

async fn confirm_update(version: Version) -> bool {
    let answer = ForgeWidget::confirm(format!(
        "Confirm upgrade from {} -> {} (latest)?",
        VERSION.to_string().bold().white(),
        version.to_string().bold().white()
    ))
    .with_default(true)
    .prompt();

    match answer {
        Ok(Some(result)) => result,
        Ok(None) => false, // User canceled
        Err(_) => false,   // Error occurred
    }
}

fn should_check_for_updates(frequency: &UpdateFrequency) -> bool {
    !matches!(frequency, UpdateFrequency::Never)
}

/// Checks if there is an update available
pub async fn on_update(api: Arc<impl API>, update: Option<&Update>) {
    let update = update.cloned().unwrap_or_default();
    let frequency = update.frequency.unwrap_or_default();

    if !should_check_for_updates(&frequency) {
        return;
    }

    let auto_update = update.auto_update.unwrap_or_default();

    // Check if version is development version, in which case we skip the update
    // check
    if VERSION.contains("dev") || VERSION == "0.1.0" {
        // Skip update for development version 0.1.0
        return;
    }

    let informer = update_informer::new(registry::GitHub, UPDATE_REPOSITORY, VERSION)
        .interval(frequency.into());

    if let Some(version) = informer.check_version().ok().flatten()
        && (auto_update || confirm_update(version).await)
    {
        execute_update_command(api, auto_update).await;
    }
}

/// Sends an event to the tracker when an update fails
async fn send_update_failure_event(error_msg: &str) -> anyhow::Result<()> {
    tracing::error!(error = error_msg, "Update failed");
    // Always return Ok since we want to fail silently
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_should_skip_update_check_when_frequency_is_never() {
        let fixture = UpdateFrequency::Never;

        let actual = should_check_for_updates(&fixture);

        let expected = false;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_uses_fork_source() {
        let fixture = local_fork_update_command();

        let actual = fixture.contains("Stranmor/oven.git")
            && fixture.contains(LOCAL_FORGE_SOURCE)
            && fixture.contains("cargo build --release")
            && fixture.contains("-p forge_main");

        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_never_uses_official_installer() {
        let fixture = local_fork_update_command();
        let official_url_fragments = ["forgecode", "dev/cli"];
        let has_official_url = official_url_fragments
            .iter()
            .any(|fragment| fixture.contains(fragment));

        let actual = has_official_url || fixture.contains("curl ") || fixture.contains("| sh");

        let expected = false;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_preserves_wrapper_boundary() {
        let fixture = local_fork_update_command();

        let actual = fixture.contains(FORGE_REAL_TARGET)
            && fixture.contains(FORGE_WRAPPER_ENTRYPOINT)
            && fixture.contains("mv -f \"$tmp\" \"$target\"")
            && !fixture.contains("mv -f \"$tmp\" \"$wrapper\"")
            && !fixture.contains("cp \"$repo/target/release/forge\" \"$wrapper\"");

        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_update_repository_points_to_local_fork() {
        let actual = UPDATE_REPOSITORY;

        let expected = "Stranmor/oven";
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_never_references_official_distribution_host() {
        let fixture = local_fork_update_command();

        let forbidden_fragments = [
            "forgecode.dev",
            "api.forgecode.dev",
            "install.sh",
            "dev/cli",
            "npm install -g",
            "bun install -g",
        ];

        let actual = forbidden_fragments
            .iter()
            .any(|fragment| fixture.contains(fragment));

        let expected = false;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_never_pipes_network_to_shell() {
        let fixture = local_fork_update_command();

        let downloads_remote_code = ["curl ", "wget "]
            .iter()
            .any(|fragment| fixture.contains(fragment));
        let executes_piped_shell = ["| sh", "| bash", "sh -c", "bash -c"]
            .iter()
            .any(|fragment| fixture.contains(fragment));

        let actual = downloads_remote_code || executes_piped_shell;

        let expected = false;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_forge_real_target_is_distinct_from_wrapper_entrypoint() {
        let actual = FORGE_REAL_TARGET != FORGE_WRAPPER_ENTRYPOINT
            && FORGE_REAL_TARGET.ends_with("/forge-real")
            && FORGE_WRAPPER_ENTRYPOINT.ends_with("/forge");

        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_requires_origin_main_state() {
        let fixture = local_fork_update_command();

        let actual = fixture.contains("fetch origin main:refs/remotes/origin/main")
            && !fixture.contains("fetch origin main;")
            && fixture.contains("rev-parse --abbrev-ref HEAD")
            && fixture.contains("= \"main\"")
            && fixture.contains("rev-parse HEAD")
            && fixture.contains("rev-parse refs/remotes/origin/main")
            && fixture.contains("status --porcelain");

        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_local_fork_update_command_requires_wrapper_entrypoint() {
        let fixture = local_fork_update_command();

        let actual = fixture.contains("test -L \"$wrapper\"")
            && !fixture.contains("test -e \"$wrapper\"")
            && fixture.contains("readlink -f \"$wrapper\"")
            && fixture.contains("!= \"$target\"");

        let expected = true;
        assert_eq!(actual, expected);
    }
}
