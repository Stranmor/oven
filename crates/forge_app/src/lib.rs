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
mod agent;
mod agent_executor;
mod agent_provider_resolver;
mod app;
mod apply_tunable_parameters;
mod changed_files;
mod command_generator;
mod compact;
mod data_gen;
pub mod dto;
mod error;
mod file_tracking;
mod fmt;
mod git_app;
mod hooks;
mod infra;
mod init_conversation_metrics;
mod mcp_executor;
mod operation;
mod orch;
#[cfg(test)]
mod orch_spec;
mod retry;
mod search_dedup;
mod services;
mod set_conversation_id;
pub mod system_prompt;
mod template_engine;
mod title_generator;
mod tool_executor;
mod tool_registry;
mod tool_resolver;
mod transformers;
mod truncation;
mod user;
pub mod user_prompt;
pub mod utils;
mod walker;
mod workspace_status;

pub use agent::*;
pub use agent_provider_resolver::*;
pub use app::*;
pub use command_generator::*;
pub use data_gen::*;
pub use error::*;
pub use git_app::*;
pub use infra::*;
pub use services::*;
pub use template_engine::*;
pub use tool_resolver::*;
pub use user::*;
pub use utils::{compute_hash, is_binary_content_type};
pub use walker::*;
pub use workspace_status::*;
pub mod domain {
    pub use forge_domain::*;
}
