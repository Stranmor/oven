//! Static Cargo workspace, package, target, feature, and dependency declarations.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::types::{
    CargoDependencyDeclaration, CargoDependencyKind, CargoFeatureMetadata, CargoPackageDependency,
    CargoPackageMetadata, CargoTargetDeclaration, CargoTargetKind, CargoTargetMetadata,
    CargoWorkspaceMetadata,
};
use crate::util::{normalize_path, provenance};

#[derive(Clone, Debug)]
pub(crate) struct CargoManifestInput {
    pub(crate) path: String,
    pub(crate) content: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StaticCargoMetadata {
    pub(crate) workspace: Option<CargoWorkspaceMetadata>,
    pub(crate) packages: Vec<CargoPackageMetadata>,
    pub(crate) dependencies: Vec<CargoPackageDependency>,
}

#[derive(Clone, Debug)]
struct ParsedManifest {
    path: String,
    root: String,
    document: DocumentMut,
}

#[derive(Clone, Debug)]
struct PackageIdentity {
    manifest_path: String,
    package_root: String,
    name: String,
}

struct DependencyExtractionContext<'a> {
    manifest: &'a ParsedManifest,
    declaring_package: Option<&'a str>,
    workspace_dependencies: &'a BTreeMap<String, CargoPackageDependency>,
    packages: &'a [PackageIdentity],
    kind: CargoDependencyKind,
    target: Option<&'a str>,
    source: &'a str,
}

/// Extracts static Cargo declaration metadata from already discovered Cargo manifests.
///
/// # Arguments
///
/// * `manifests` - UTF-8 Cargo.toml inputs discovered by the project indexer.
/// * `known_files` - Relative project files used only for static target/path confirmation.
///
/// # Errors
///
/// Returns an error when any Cargo.toml cannot be parsed as TOML.
pub(crate) fn extract_static_cargo_metadata(
    manifests: &[CargoManifestInput],
    known_files: &BTreeSet<String>,
) -> Result<StaticCargoMetadata> {
    let mut parsed = parse_manifests(manifests)?;
    parsed.sort_by(|left, right| left.path.cmp(&right.path));
    let package_identities = package_identities(&parsed);
    let workspace = extract_workspace(&parsed, &package_identities);
    let workspace_dependencies = workspace_dependencies(&parsed);
    let mut packages = Vec::new();
    let mut dependencies = Vec::new();
    for manifest in &parsed {
        if let Some(package) = extract_package(manifest, known_files) {
            dependencies.extend(extract_package_dependencies(
                manifest,
                &package.name,
                &workspace_dependencies,
                &package_identities,
            ));
            packages.push(package);
        }
    }
    dependencies.extend(workspace_dependencies.values().cloned());
    packages.sort_by(|left, right| {
        left.manifest_path
            .cmp(&right.manifest_path)
            .then_with(|| left.name.cmp(&right.name))
    });
    dependencies.sort_by(compare_dependencies);
    Ok(StaticCargoMetadata { workspace, packages, dependencies })
}

fn parse_manifests(manifests: &[CargoManifestInput]) -> Result<Vec<ParsedManifest>> {
    let mut parsed = Vec::new();
    for manifest in manifests {
        let document = manifest
            .content
            .parse::<DocumentMut>()
            .with_context(|| format!("parse Cargo manifest {}", manifest.path))?;
        parsed.push(ParsedManifest {
            path: manifest.path.clone(),
            root: manifest_root(&manifest.path),
            document,
        });
    }
    Ok(parsed)
}

fn package_identities(manifests: &[ParsedManifest]) -> Vec<PackageIdentity> {
    let mut identities = manifests
        .iter()
        .filter_map(|manifest| {
            let name = manifest
                .document
                .get("package")?
                .as_table()?
                .get("name")?
                .as_str()?
                .to_string();
            Some(PackageIdentity {
                manifest_path: manifest.path.clone(),
                package_root: manifest.root.clone(),
                name,
            })
        })
        .collect::<Vec<_>>();
    identities.sort_by(|left, right| left.manifest_path.cmp(&right.manifest_path));
    identities
}

fn extract_workspace(
    manifests: &[ParsedManifest],
    package_identities: &[PackageIdentity],
) -> Option<CargoWorkspaceMetadata> {
    let manifest = manifests
        .iter()
        .filter(|manifest| manifest.document.get("workspace").is_some())
        .min_by(|left, right| left.path.cmp(&right.path))?;
    let workspace_table = manifest.document.get("workspace")?.as_table()?;
    let mut members = string_array(workspace_table.get("members"));
    members.sort();
    let package_manifest_paths =
        workspace_member_package_paths(manifest, &members, package_identities);
    Some(CargoWorkspaceMetadata {
        manifest_path: manifest.path.clone(),
        root_path: manifest.root.clone(),
        members,
        package_manifest_paths,
        provenance: provenance(
            &manifest.path,
            None,
            None,
            "cargo-workspace-static",
            &format!("workspace:{}", manifest.path),
        ),
    })
}

fn workspace_member_package_paths(
    workspace_manifest: &ParsedManifest,
    members: &[String],
    package_identities: &[PackageIdentity],
) -> Vec<String> {
    let mut paths = BTreeSet::new();
    for member in members {
        if let Some(prefix) = member.strip_suffix("/*") {
            let root = join_relative(&workspace_manifest.root, prefix);
            for package in package_identities {
                if parent_path(&package.package_root) == root {
                    paths.insert(package.manifest_path.clone());
                }
            }
        } else {
            let root = join_relative(&workspace_manifest.root, member);
            let manifest_path = join_relative(&root, "Cargo.toml");
            if package_identities
                .iter()
                .any(|package| package.manifest_path == manifest_path)
            {
                paths.insert(manifest_path);
            }
        }
    }
    paths.into_iter().collect()
}

fn extract_package(
    manifest: &ParsedManifest,
    known_files: &BTreeSet<String>,
) -> Option<CargoPackageMetadata> {
    let package_table = manifest.document.get("package")?.as_table()?;
    let name = package_table.get("name")?.as_str()?.to_string();
    let version = package_table
        .get("version")
        .and_then(Item::as_str)
        .map(str::to_string);
    let edition = package_table
        .get("edition")
        .and_then(Item::as_str)
        .map(str::to_string);
    let mut targets = extract_targets(manifest, &name, known_files);
    targets.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.path.cmp(&right.path))
    });
    let mut features = extract_features(manifest);
    features.sort_by(|left, right| left.name.cmp(&right.name));
    Some(CargoPackageMetadata {
        manifest_path: manifest.path.clone(),
        package_root: manifest.root.clone(),
        name: name.clone(),
        version,
        edition,
        targets,
        features,
        provenance: provenance(
            &manifest.path,
            None,
            None,
            "cargo-package-static",
            &format!("package:{}:{name}", manifest.path),
        ),
    })
}

fn extract_targets(
    manifest: &ParsedManifest,
    package_name: &str,
    known_files: &BTreeSet<String>,
) -> Vec<CargoTargetMetadata> {
    let mut targets = Vec::new();
    if let Some(path) = explicit_lib_path(manifest) {
        targets.push(target(
            manifest,
            package_name,
            CargoTargetKind::Lib,
            &path,
            CargoTargetDeclaration::Declared,
            "cargo-target-static",
        ));
    } else if let Some(path) =
        first_existing(known_files, &[join_relative(&manifest.root, "src/lib.rs")])
    {
        targets.push(target(
            manifest,
            package_name,
            CargoTargetKind::Lib,
            &path,
            CargoTargetDeclaration::ConventionInferred,
            "cargo-target-static-inferred",
        ));
    }
    for (name, path, declaration) in bin_targets(manifest, known_files) {
        targets.push(target(
            manifest,
            &name,
            CargoTargetKind::Bin,
            &path,
            declaration,
            "cargo-target-static",
        ));
    }
    targets
}

fn explicit_lib_path(manifest: &ParsedManifest) -> Option<String> {
    let table = manifest.document.get("lib")?.as_table()?;
    let path = table
        .get("path")
        .and_then(Item::as_str)
        .map(|path| join_relative(&manifest.root, path))
        .unwrap_or_else(|| join_relative(&manifest.root, "src/lib.rs"));
    Some(path)
}

fn bin_targets(
    manifest: &ParsedManifest,
    known_files: &BTreeSet<String>,
) -> Vec<(String, String, CargoTargetDeclaration)> {
    let mut targets = Vec::new();
    if let Some(array) = manifest
        .document
        .get("bin")
        .and_then(Item::as_array_of_tables)
    {
        for table in array {
            let name = table
                .get("name")
                .and_then(Item::as_str)
                .unwrap_or("bin")
                .to_string();
            let path = table
                .get("path")
                .and_then(Item::as_str)
                .map(|path| join_relative(&manifest.root, path))
                .unwrap_or_else(|| join_relative(&manifest.root, &format!("src/bin/{name}.rs")));
            targets.push((name, path, CargoTargetDeclaration::Declared));
        }
    }
    let main_path = join_relative(&manifest.root, "src/main.rs");
    if known_files.contains(&main_path) && !targets.iter().any(|(_, path, _)| path == &main_path) {
        targets.push((
            package_binary_name(manifest),
            main_path,
            CargoTargetDeclaration::ConventionInferred,
        ));
    }
    let bin_prefix = join_relative(&manifest.root, "src/bin/");
    for file in known_files {
        if let Some(rest) = file.strip_prefix(&bin_prefix)
            && let Some(name) = rest.strip_suffix(".rs")
            && !name.contains('/')
            && !targets.iter().any(|(_, path, _)| path == file)
        {
            targets.push((
                name.to_string(),
                file.clone(),
                CargoTargetDeclaration::ConventionInferred,
            ));
        }
    }
    targets
}

fn target(
    manifest: &ParsedManifest,
    name: &str,
    kind: CargoTargetKind,
    path: &str,
    declaration: CargoTargetDeclaration,
    source: &str,
) -> CargoTargetMetadata {
    CargoTargetMetadata {
        name: name.to_string(),
        kind,
        path: path.to_string(),
        declaration,
        provenance: provenance(
            &manifest.path,
            None,
            None,
            source,
            &format!("target:{}:{name}:{path}", manifest.path),
        ),
    }
}

fn package_binary_name(manifest: &ParsedManifest) -> String {
    manifest
        .document
        .get("package")
        .and_then(Item::as_table)
        .and_then(|table| table.get("name"))
        .and_then(Item::as_str)
        .unwrap_or("bin")
        .to_string()
}

fn extract_features(manifest: &ParsedManifest) -> Vec<CargoFeatureMetadata> {
    let mut features = Vec::new();
    if let Some(table) = manifest.document.get("features").and_then(Item::as_table) {
        for (name, item) in table.iter() {
            let mut members = string_array(Some(item));
            members.sort();
            features.push(CargoFeatureMetadata {
                name: name.to_string(),
                members,
                provenance: provenance(
                    &manifest.path,
                    None,
                    None,
                    "cargo-feature-static",
                    &format!("feature:{}:{name}", manifest.path),
                ),
            });
        }
    }
    features
}

fn workspace_dependencies(
    manifests: &[ParsedManifest],
) -> BTreeMap<String, CargoPackageDependency> {
    let mut dependencies = BTreeMap::new();
    for manifest in manifests {
        let Some(table) = manifest
            .document
            .get("workspace")
            .and_then(Item::as_table)
            .and_then(|workspace| workspace.get("dependencies"))
            .and_then(Item::as_table)
        else {
            continue;
        };
        for (name, item) in table.iter() {
            let mut dependency = dependency_from_item(
                manifest,
                None,
                name,
                item,
                CargoDependencyKind::Normal,
                None,
                "workspace.dependencies",
            );
            if dependency.declaration == CargoDependencyDeclaration::DeclaredWorkspaceInherited {
                dependency.declaration = CargoDependencyDeclaration::UnresolvedStatic;
            }
            dependencies.insert(name.to_string(), dependency);
        }
    }
    dependencies
}

fn extract_package_dependencies(
    manifest: &ParsedManifest,
    package_name: &str,
    workspace_dependencies: &BTreeMap<String, CargoPackageDependency>,
    packages: &[PackageIdentity],
) -> Vec<CargoPackageDependency> {
    let mut dependencies = Vec::new();
    dependencies.extend(section_dependencies(
        manifest,
        Some(package_name),
        workspace_dependencies,
        packages,
        "dependencies",
        CargoDependencyKind::Normal,
        None,
    ));
    dependencies.extend(section_dependencies(
        manifest,
        Some(package_name),
        workspace_dependencies,
        packages,
        "dev-dependencies",
        CargoDependencyKind::Dev,
        None,
    ));
    dependencies.extend(section_dependencies(
        manifest,
        Some(package_name),
        workspace_dependencies,
        packages,
        "build-dependencies",
        CargoDependencyKind::Build,
        None,
    ));
    if let Some(targets) = manifest.document.get("target").and_then(Item::as_table) {
        for (target_name, target_item) in targets.iter() {
            if let Some(target_table) = target_item.as_table() {
                dependencies.extend(section_dependencies_from_table(
                    DependencyExtractionContext {
                        manifest,
                        declaring_package: Some(package_name),
                        workspace_dependencies,
                        packages,
                        kind: CargoDependencyKind::Normal,
                        target: Some(target_name),
                        source: "dependencies",
                    },
                    target_table,
                    "dependencies",
                ));
                dependencies.extend(section_dependencies_from_table(
                    DependencyExtractionContext {
                        manifest,
                        declaring_package: Some(package_name),
                        workspace_dependencies,
                        packages,
                        kind: CargoDependencyKind::Dev,
                        target: Some(target_name),
                        source: "dev-dependencies",
                    },
                    target_table,
                    "dev-dependencies",
                ));
                dependencies.extend(section_dependencies_from_table(
                    DependencyExtractionContext {
                        manifest,
                        declaring_package: Some(package_name),
                        workspace_dependencies,
                        packages,
                        kind: CargoDependencyKind::Build,
                        target: Some(target_name),
                        source: "build-dependencies",
                    },
                    target_table,
                    "build-dependencies",
                ));
            }
        }
    }
    dependencies
}

fn section_dependencies(
    manifest: &ParsedManifest,
    declaring_package: Option<&str>,
    workspace_dependencies: &BTreeMap<String, CargoPackageDependency>,
    packages: &[PackageIdentity],
    section: &str,
    kind: CargoDependencyKind,
    target: Option<&str>,
) -> Vec<CargoPackageDependency> {
    let Some(table) = manifest.document.get(section).and_then(Item::as_table) else {
        return Vec::new();
    };
    let context = DependencyExtractionContext {
        manifest,
        declaring_package,
        workspace_dependencies,
        packages,
        kind,
        target,
        source: section,
    };
    dependencies_from_table(&context, table)
}

fn section_dependencies_from_table(
    context: DependencyExtractionContext<'_>,
    target_table: &Table,
    section: &str,
) -> Vec<CargoPackageDependency> {
    let Some(table) = target_table.get(section).and_then(Item::as_table) else {
        return Vec::new();
    };
    dependencies_from_table(&context, table)
}

fn dependencies_from_table(
    context: &DependencyExtractionContext<'_>,
    table: &Table,
) -> Vec<CargoPackageDependency> {
    let mut dependencies = Vec::new();
    for (name, item) in table.iter() {
        let mut dependency = dependency_from_item(
            context.manifest,
            context.declaring_package,
            name,
            item,
            context.kind.clone(),
            context.target,
            context.source,
        );
        if dependency.declaration == CargoDependencyDeclaration::DeclaredWorkspaceInherited
            && let Some(workspace_dependency) = context.workspace_dependencies.get(name)
        {
            dependency.package_name = workspace_dependency.package_name.clone();
        }
        dependency.linked_package_manifest_path =
            linked_path_dependency(&dependency, context.manifest, context.packages);
        dependencies.push(dependency);
    }
    dependencies
}

fn dependency_from_item(
    manifest: &ParsedManifest,
    declaring_package: Option<&str>,
    name: &str,
    item: &Item,
    kind: CargoDependencyKind,
    target: Option<&str>,
    source: &str,
) -> CargoPackageDependency {
    let mut package_name = name.to_string();
    let mut version = item.as_str().map(str::to_string);
    let mut path = None;
    let mut optional = false;
    let mut features = Vec::new();
    let mut declaration = CargoDependencyDeclaration::DeclaredExternal;
    if let Some(table) = item.as_inline_table() {
        package_name = table
            .get("package")
            .and_then(value_as_str)
            .unwrap_or(name)
            .to_string();
        version = table
            .get("version")
            .and_then(value_as_str)
            .map(str::to_string);
        path = table.get("path").and_then(value_as_str).map(str::to_string);
        optional = table
            .get("optional")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        features = value_string_array(table.get("features"));
        if table
            .get("workspace")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            declaration = CargoDependencyDeclaration::DeclaredWorkspaceInherited;
        } else if path.is_some() {
            declaration = CargoDependencyDeclaration::DeclaredPath;
        }
    } else if let Some(table) = item.as_table() {
        package_name = table
            .get("package")
            .and_then(Item::as_str)
            .unwrap_or(name)
            .to_string();
        version = table
            .get("version")
            .and_then(Item::as_str)
            .map(str::to_string);
        path = table.get("path").and_then(Item::as_str).map(str::to_string);
        optional = table
            .get("optional")
            .and_then(Item::as_bool)
            .unwrap_or(false);
        features = string_array(table.get("features"));
        if table
            .get("workspace")
            .and_then(Item::as_bool)
            .unwrap_or(false)
        {
            declaration = CargoDependencyDeclaration::DeclaredWorkspaceInherited;
        } else if path.is_some() {
            declaration = CargoDependencyDeclaration::DeclaredPath;
        }
    } else if item.is_none() || item.as_value().is_none() {
        declaration = CargoDependencyDeclaration::UnresolvedStatic;
    }
    features.sort();
    CargoPackageDependency {
        manifest_path: manifest.path.clone(),
        declaring_package: declaring_package.map(str::to_string),
        dependency_key: name.to_string(),
        package_name,
        kind,
        target: target.map(str::to_string),
        version,
        path,
        optional,
        features,
        declaration,
        linked_package_manifest_path: None,
        provenance: provenance(
            &manifest.path,
            None,
            None,
            source,
            &format!("dependency:{}:{source}:{name}:{target:?}", manifest.path),
        ),
    }
}

fn linked_path_dependency(
    dependency: &CargoPackageDependency,
    manifest: &ParsedManifest,
    packages: &[PackageIdentity],
) -> Option<String> {
    if dependency.declaration != CargoDependencyDeclaration::DeclaredPath {
        return None;
    }
    let dependency_path = dependency.path.as_ref()?;
    let package_root = join_relative(&manifest.root, dependency_path);
    packages
        .iter()
        .find(|package| {
            package.package_root == package_root && package.name == dependency.package_name
        })
        .map(|package| package.manifest_path.clone())
}

fn value_as_str(value: &Value) -> Option<&str> {
    value.as_str()
}

fn string_array(item: Option<&Item>) -> Vec<String> {
    let Some(item) = item else {
        return Vec::new();
    };
    item.as_array()
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn first_existing(known_files: &BTreeSet<String>, paths: &[String]) -> Option<String> {
    paths
        .iter()
        .find(|path| known_files.contains(path.as_str()))
        .cloned()
}

fn manifest_root(path: &str) -> String {
    parent_path(path)
}

fn parent_path(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(normalize_path)
        .unwrap_or_default()
}

fn join_relative(root: &str, path: &str) -> String {
    normalize_lexical(Path::new(root).join(path))
}

fn normalize_lexical(path: PathBuf) -> String {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(value) => normalized.push(value),
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    normalize_path(&normalized)
}

fn compare_dependencies(
    left: &CargoPackageDependency,
    right: &CargoPackageDependency,
) -> std::cmp::Ordering {
    left.manifest_path
        .cmp(&right.manifest_path)
        .then_with(|| left.declaring_package.cmp(&right.declaring_package))
        .then_with(|| left.target.cmp(&right.target))
        .then_with(|| left.kind.cmp(&right.kind))
        .then_with(|| left.dependency_key.cmp(&right.dependency_key))
        .then_with(|| left.package_name.cmp(&right.package_name))
}
