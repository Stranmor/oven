use std::process::Command;

fn clean_version(version: &str) -> String {
    match version.strip_prefix('v') {
        Some(stripped) => stripped.to_string(),
        None => version.to_string(),
    }
}

fn git_last_updated() -> Option<String> {
    let output = Command::new("git")
        .args(["log", "-1", "--format=%cI"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn last_updated() -> String {
    std::env::var("APP_LAST_UPDATED")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(git_last_updated)
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    // Priority order:
    // 1. APP_VERSION environment variable (for CI/CD builds)
    // 2. Fallback to dev version

    let version = std::env::var("APP_VERSION")
        .map(|v| clean_version(&v))
        .unwrap_or_else(|_| "0.1.0-dev".to_string());

    // Make version available to the application
    println!("cargo:rustc-env=CARGO_PKG_VERSION={version}");
    println!("cargo:rustc-env=FORGE_LAST_UPDATED={}", last_updated());

    // Make version available to the application
    println!("cargo:rustc-env=CARGO_PKG_NAME=forge");

    // Ensure rebuild when environment changes
    println!("cargo:rerun-if-env-changed=APP_VERSION");
    println!("cargo:rerun-if-env-changed=APP_LAST_UPDATED");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads/main");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");
}
