# Agent Guidelines

This document contains guidelines and best practices for AI agents working with this codebase.

## Error Management

- Use `anyhow::Result` for error handling in services and repositories.
- Create domain errors using `thiserror`.
- Never implement `From` for converting domain errors, manually convert them

## Writing Tests

- All tests should be written in three discrete steps:

  ```rust,ignore
  use pretty_assertions::assert_eq; // Always use pretty assertions

  fn test_foo() {
      let setup = ...; // Instantiate a fixture or setup for the test
      let actual = ...; // Execute the fixture to create an output
      let expected = ...; // Define a hand written expected result
      assert_eq!(actual, expected); // Assert that the actual result matches the expected result
  }
  ```

- Use `pretty_assertions` for better error messages.

- Use fixtures to create test data.

- Use `assert_eq!` for equality checks.

- Use `assert!(...)` for boolean checks.

- Use unwraps in test functions and anyhow::Result in fixtures.

- Keep the boilerplate to a minimum.

- Use words like `fixture`, `actual` and `expected` in test functions.

- Fixtures should be generic and reusable.

- Test should always be written in the same file as the source code.

- Use `new`, Default and derive_setters::Setters to create `actual`, `expected` and specially `fixtures`. For example:

  **Good:**

  ```rust,ignore
  User::default().age(12).is_happy(true).name("John")
  User::new("Job").age(12).is_happy()
  User::test() // Special test constructor
  ```

  **Bad:**

  ```rust,ignore
  User {name: "John".to_string(), is_happy: true, age: 12}
  User::with_name("Job") // Bad name, should stick to User::new() or User::test()
  ```

- Use `unwrap()` unless the error information is useful. Use `expect` instead of `panic!` when error message is useful. For example:

  **Good:**

  ```rust,ignore
  users.first().expect("List should not be empty")
  ```

  **Bad:**

  ```rust,ignore
  if let Some(user) = users.first() {
      // ...
  } else {
      panic!("List should not be empty")
  }
  ```

- Prefer using `assert_eq` on full objects instead of asserting each field:

  **Good:**

  ```rust,ignore
  assert_eq!(actual, expected);
  ```

  **Bad:**

  ```rust,ignore
  assert_eq!(actual.a, expected.a);
  assert_eq!(actual.b, expected.b);
  ```

## Verification

Always verify changes by running tests and linting the codebase

1. Run crate specific tests to ensure they pass.

   ```
   cargo insta test --accept
   ```

2. **Build Guidelines**:
   - **NEVER** run `cargo build --release` unless absolutely necessary (e.g., performance testing, creating binaries for distribution)
   - For verification, use `cargo check` (fastest), `cargo insta test`, or `cargo build` (debug mode)
   - Release builds take significantly longer and are rarely needed for development verification

## Writing Domain Types

- Use `derive_setters` to derive setters and use the `strip_option` and the `into` attributes on the struct types.

## Documentation

- **Always** write Rust docs (`///`) for all public methods, functions, structs, enums, and traits.
- Document parameters with `# Arguments` and errors with `# Errors` sections when applicable.
- **Do not include code examples** - docs are for LLMs, not humans. Focus on clear, concise functionality descriptions.

## Refactoring

- If asked to fix failing tests, always confirm whether to update the implementation or the tests.

## Git Operations

- Safely assume git is pre-installed
- Safely assume github cli (gh) is pre-installed
- Always use `Co-Authored-By: ForgeCode <noreply@forgecode.dev>` for git commits and Github comments

## Local Forge Update Source

The active local `forge` binary is protected infrastructure. Its updater-consumed source MUST be the user's fork `Stranmor/oven` on `origin/main`, because that branch integrates upstream Forge changes with local regression fixes. Upstream input MUST be `tailcallhq/forgecode` `main`; stale `antinomyhq/forgecode` upstream remotes are historical drift and must not drive automation. Directly tracking or consuming upstream `main` for the active local binary is forbidden: it can overwrite local patches and reintroduce fixed regressions.

Correct update flow: merge or port upstream changes into `origin/main` first, verify the integrated fork, then let the updater consume that fork state. The updater MUST install the real executable to `/home/stranmor/.local/lib/forge/forge-real` and preserve wrapper/symlink entrypoints: a freshly built binary must never replace `~/.local/bin/forge` when that path is a wrapper or symlink. If the active PATH chain points through a managed dotfiles/config path, such as `/home/stranmor/.local/bin/forge -> /home/stranmor/configs/bin/forge`, that target is also part of the protected entrypoint boundary. A stale ELF binary or non-delegating wrapper at any hop must be repaired forward by preserving the previous executable as a backup and replacing the hop with a minimal wrapper that `exec`s `$HOME/.local/lib/forge/forge-real`; do not overwrite unknown executable state without preserving it first.

Verification must prove the active command path, not only the installed target state. After any local Forge update or deployment, resolve the PATH entrypoint actually used by `forge`, inspect the full wrapper/symlink chain, and prove that executing `forge --version` reaches the intended `/home/stranmor/.local/lib/forge/forge-real`. When the chain has been repaired or when a stale binary was found, verification must include process-level evidence such as `strace`/`execve` showing the active command delegates to `forge-real`, plus the resulting `forge --version`. A successful updater state, changed `forge-real` timestamp, installed-source revision, or direct execution of `forge-real` is insufficient if the active PATH entrypoint still resolves to an old ELF binary or any wrapper that does not delegate to `forge-real`.

Local/global Cargo configuration is ambient build state, not repository truth. If a user-level Cargo config such as `/home/stranmor/.cargo/config.toml` makes release builds incompatible with Forge dependencies (for example `profile.release.panic = "abort"` breaking `forge_main` through `html2md`), do not edit the repository or global config merely to complete a local install. Use a task-local build override such as `CARGO_PROFILE_RELEASE_PANIC=unwind` together with local `target-dir`/profile overrides, document the override in the deployment proof, and leave durable config changes to a separate explicit configuration task.

Detection: About to point a local Forge auto-updater at upstream/main, install a built binary directly over `~/.local/bin/forge`, bypass the fork integration branch, report a Forge update as verified from `forge-real`/updater state alone, ignore a stale ELF/non-delegating executable inside the active wrapper chain, or patch repo/global Cargo config to work around a local release-build profile mismatch → STOP → update `origin/main` first, preserve and repair wrapper/symlink hops forward, use task-local build overrides for ambient Cargo profile conflicts, trace the active PATH command through every wrapper/symlink hop, and verify the active `forge --version` reaches the intended real binary with process-level evidence when the chain was suspect.

Mnemonic: The fork is the update source; upstream is input, not the installed truth. The active PATH command is the proof, not the payload file; local build quirks stay local unless deliberately promoted.

## Rust TUI Architecture Direction

Forge's interactive terminal UI direction is a Rust-native TUI built on `ratatui` with `crossterm` as the terminal rendering substrate. This is an additive presentation architecture decision, not authorization for a hard rewrite of the existing classic UI path. The active upstream-compatible install and update model remains unchanged: local active Forge comes from `Stranmor/oven` `origin/main`; `tailcallhq/forgecode` `main` is upstream input that must be merged or ported into the fork before local installation, never consumed directly as the installed source of truth.

The TUI must preserve a typed UI boundary. Domain/API/app surfaces emit `ChatResponse` values or equivalent typed domain events, those events are transformed into a shared UI render model such as `forge_ui_model`, and renderers consume that model. The classic stdout/transcript renderer and the `ratatui` renderer are sibling presentation adapters; neither should parse domain objects ad hoc or own business semantics. Markdown, tool-call output, streaming chunks, transcript records, status indicators, and structured assistant/user messages should be represented once in the typed render model and rendered by each adapter, not duplicated as parallel parser/rendering logic.

`ratatui` and `crossterm` belong only in presentation crates/modules. They must stay out of domain, API, agent/application orchestration, provider, tool-call, project-model, and infrastructure crates except for narrow feature-gated adapter wiring whose dependency direction remains presentation-only. `crates/forge_main/src/ui.rs` and other hot upstream files should receive only thin seams, delegation points, or compatibility-preserving adapters. Local rich TUI behavior must live behind additive crates/modules and feature/config gates so upstream sync remains reviewable and low-conflict.

Classic stdout and transcript mode remain first-class fallbacks for shell workflows, CI logs, non-interactive terminals, redirected output, remote automation, and any environment where the TUI is unavailable or disabled. The TUI must not make transcript fidelity, stream consumption, tool output visibility, or shell-safe behavior second-class. Renderer selection should be explicit and safe, with equivalent semantic coverage across renderers.

Drift protection is required for this architecture. Tests or fixtures must protect: upstream-sync compatibility of thin seams in hot files; renderer behavior for the shared UI model; markdown/tool-output semantics across classic and TUI renderers; feature/config-gated availability; and fallback behavior for stdout/transcript workflows. Detection: About to add TUI behavior by rewriting `crates/forge_main/src/ui.rs`, importing `ratatui`/`crossterm` into domain/API/app crates, duplicating markdown/tool-output parsing in each renderer, consuming upstream directly instead of the fork, or weakening classic stdout/transcript behavior → STOP → route the change through the typed UI model boundary, keep dependencies presentation-only, gate rich TUI features additively, and add drift tests.

Mnemonic: TUI is a renderer, not the product core. Events become a typed UI model; stdout and `ratatui` render the same semantics; the fork stays the installed truth.

## Service Implementation Guidelines

Services should follow clean architecture principles and maintain clear separation of concerns:

### Core Principles

- **No service-to-service dependencies**: Services should never depend on other services directly
- **Infrastructure dependency**: Services should depend only on infrastructure abstractions when needed
- **Single type parameter**: Services should take at most one generic type parameter for infrastructure
- **No trait objects**: Avoid `Box<dyn ...>` - use concrete types and generics instead
- **Constructor pattern**: Implement `new()` without type bounds - apply bounds only on methods that need them
- **Compose dependencies**: Use the `+` operator to combine multiple infrastructure traits into a single bound
- **Arc<T> for infrastructure**: Store infrastructure as `Arc<T>` for cheap cloning and shared ownership
- **Tuple struct pattern**: For simple services with single dependency, use tuple structs `struct Service<T>(Arc<T>)`

### Examples

#### Simple Service (No Infrastructure)

```rust,ignore
pub struct UserValidationService;

impl UserValidationService {
    pub fn new() -> Self { ... }

    pub fn validate_email(&self, email: &str) -> Result<()> {
        // Validation logic here
        ...
    }

    pub fn validate_age(&self, age: u32) -> Result<()> {
        // Age validation logic here
        ...
    }
}
```

#### Service with Infrastructure Dependency

```rust,ignore
// Infrastructure trait (defined in infrastructure layer)
pub trait UserRepository {
    fn find_by_email(&self, email: &str) -> Result<Option<User>>;
    fn save(&self, user: &User) -> Result<()>;
}

// Service with single generic parameter using Arc
pub struct UserService<R> {
    repository: Arc<R>,
}

impl<R> UserService<R> {
    // Constructor without type bounds, takes Arc<R>
    pub fn new(repository: Arc<R>) -> Self { ... }
}

impl<R: UserRepository> UserService<R> {
    // Business logic methods have type bounds where needed
    pub fn create_user(&self, email: &str, name: &str) -> Result<User> { ... }
    pub fn find_user(&self, email: &str) -> Result<Option<User>> { ... }
}
```

#### Tuple Struct Pattern for Simple Services

```rust,ignore
// Infrastructure traits
pub trait FileReader {
    async fn read_file(&self, path: &Path) -> Result<String>;
}

pub trait Environment {
    fn max_file_size(&self) -> u64;
}

// Tuple struct for simple single dependency service
pub struct FileService<F>(Arc<F>);

impl<F> FileService<F> {
    // Constructor without bounds
    pub fn new(infra: Arc<F>) -> Self { ... }
}

impl<F: FileReader + Environment> FileService<F> {
    // Business logic methods with composed trait bounds
    pub async fn read_with_validation(&self, path: &Path) -> Result<String> { ... }
}
```

### Anti-patterns to Avoid

```rust,ignore
// BAD: Service depending on another service
pub struct BadUserService<R, E> {
    repository: R,
    email_service: E, // Don't do this!
}

// BAD: Using trait objects
pub struct BadUserService {
    repository: Box<dyn UserRepository>, // Avoid Box<dyn>
}

// BAD: Multiple infrastructure dependencies with separate type parameters
pub struct BadUserService<R, C, L> {
    repository: R,
    cache: C,
    logger: L, // Too many generic parameters - hard to use and test
}

impl<R: UserRepository, C: Cache, L: Logger> BadUserService<R, C, L> {
    // BAD: Constructor with type bounds makes it hard to use
    pub fn new(repository: R, cache: C, logger: L) -> Self { ... }
}

// BAD: Usage becomes cumbersome
let service = BadUserService::<PostgresRepo, RedisCache, FileLogger>::new(...);
```

## TARGET GOAL

Implement a separated Project Representation / Model Context layer as the active architecture goal. The preferred crate/folder boundary is `crates/forge_project_model`; if empirical integration constraints require a different name or placement, preserve the same separation and document the concrete reason in code review/commit context. This layer must own project/model-context representation rather than scattering it across providers, UI, tool-call code, or prompt assembly.

The layer should implement current SOTA project/model-context representation methods as typed Rust surfaces, including:

- repository manifest and workspace metadata;
- AST, LSP, and symbol indexes;
- dependency graph and call graph;
- knowledge graph linking files, symbols, tasks, decisions, and retrieved evidence;
- hybrid retrieval across lexical, semantic, structural, and graph signals;
- episodic memory and tool-use traces for agent workflows;
- shard manifests for context packaging and incremental loading;
- freshness, provenance, invalidation, and source-of-truth metadata;
- evals/regression fixtures proving retrieval quality, freshness behavior, and context-pack construction.

Architectural boundary: project/model-context construction, indexing, retrieval, provenance, freshness, and context-pack assembly belong behind this dedicated crate/folder boundary. Existing crates may consume the layer through typed APIs, but must not reimplement parallel context models, ad-hoc prompt-context builders, untyped JSON blobs, duplicated indexes, or provider-specific project-memory logic. Detection: About to add model/project context logic outside the dedicated boundary → STOP → either move it into `crates/forge_project_model` or add only a thin typed adapter that depends on that boundary.

Current status: project-model injection correctness, the retrieval quality baseline, durable context-pack artifact persistence, and redaction-safe search episode capture are implemented in `e47ad48e25b202eea222dc3c8f341e701f68dcb0`, `887cdf49e0fadc77b4d134216cb1863b3d510e42`, `9138efb5a4fcd37ecb656b76aac915dae669608e`, and `3f7650b004eacd1248d7aefb88e636e16e138218`. Verified with focused project-model/context/cache/artifact/tool-episode test suites, full `cargo test -p forge_project_model`, focused `forge_services` query-workspace/context-pack suites, `cargo check -p forge_project_model -p forge_services`, `cargo check -p forge_project_model -p forge_services -p forge_app`, `cargo test -p forge_app learning_capture` (4 tests), `git diff --check`, and adversarial critic PASS. Full `cargo insta test` remains pending local `cargo-nextest` availability with no snapshot changes in this milestone.

KV/cache discipline: context and system-information changes must preserve existing KV caching as an architectural constraint, not an incidental optimization. Separate frequently changing facts from stable cacheable payloads before changing prompts, context packs, system-information renderers, provider payloads, or project-model outputs. Volatile facts such as current time, live clock state, request/session identifiers, transient process status, freshness probes, and other per-run observations must be modeled as small late-bound fields or separate invalidation inputs instead of being embedded into large stable cached payloads. Stable facts such as repository manifests, workspace metadata, tool descriptions, project guidelines, source-derived indexes, and reusable context shards should remain cacheable across turns and runs whenever their source-of-truth content has not changed. Avoid unnecessary KV rewrites, preserve cache hit efficiency, and make freshness/invalidation explicit at the smallest safe granularity. Detection: About to add or modify system-information, context assembly, provider request construction, project-model manifests, or cached prompt/context payloads with data that changes every run → STOP → split volatile data from stable cached data, define the freshness/invalidation boundary, and prove the change does not force avoidable KV churn.

Mnemonic: Project context is a product model, not prompt glue; one typed layer owns it. Cache stable structure; late-bind volatile facts.
