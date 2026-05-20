# Contributing to Stranmor/oven

Stranmor/oven is a maintained fork of upstream ForgeCode. Contributions in this repository should target fork-specific integration, regression fixes, and presentation/packaging work unless a change is clearly better sent upstream first.

## Upstream attribution

The project descends from upstream ForgeCode and preserves upstream license history. Keep attribution and copyright notices intact when moving or editing inherited files.

## License

This repository is licensed under Apache-2.0, matching the root `LICENSE` file. Do not describe this repository as proprietary.

## Local setup

```bash
cargo build -p forge_main --bin forge
cargo test -p forge_ci
```

For broader Rust changes, run the relevant crate checks and tests before opening a pull request.
