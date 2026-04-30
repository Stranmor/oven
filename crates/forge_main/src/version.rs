/// Forge package version compiled into the binary.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git commit date of the source used to build this binary.
pub const LAST_UPDATED: &str = env!("FORGE_LAST_UPDATED");

/// Version string shown to operators in CLI and UI surfaces.
pub const VERSION_WITH_LAST_UPDATED: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (updated ",
    env!("FORGE_LAST_UPDATED"),
    ")"
);
