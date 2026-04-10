import re

with open("crates/forge_app/src/system_prompt.rs", "r") as f:
    content = f.read()

old_fetch = """    /// Fetches file extension statistics by running git ls-files command.
    async fn fetch_extensions(&self, max_extensions: usize) -> Option<Extension> {
        let output = self
            .services
            .execute(
                "git ls-files".into(),
                self.environment.cwd.clone(),
                false,
                true,
                None,
                None,
            )
            .await
            .ok()?;

        // If git command fails (e.g., not in a git repo), return None
        if output.output.exit_code != Some(0) {
            return None;
        }

        parse_extensions(&output.output.stdout, max_extensions)
    }"""

new_fetch = """    /// Fetches file extension statistics using the domain's FileDiscoveryService.
    async fn fetch_extensions(&self, max_extensions: usize) -> Option<Extension> {
        let files = self
            .services
            .collect_files(crate::Walker::unlimited().cwd(self.environment.cwd.clone()))
            .await
            .ok()?;

        if files.is_empty() {
            return None;
        }

        parse_extensions(&files, max_extensions)
    }"""

content = content.replace(old_fetch, new_fetch)

old_parse = """/// Parses the newline-separated output of `git ls-files` into an [`Extension`]
/// summary.
fn parse_extensions(extensions: &str, max_extensions: usize) -> Option<Extension> {
    let all_files: Vec<&str> = extensions
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    let total_files = all_files.len();
    if total_files == 0 {
        return None;
    }

    // Count files by extension; files without extensions are tracked as "(no ext)"
    let mut counts = HashMap::<&str, usize>::new();
    all_files
        .iter()
        .map(|line| {
            let file_name = line.rsplit_once(['/', '\\']).map_or(*line, |(_, name)| name);
            file_name
                .rsplit_once('.')
                .filter(|(prefix, _)| !prefix.is_empty())
                .map_or("(no ext)", |(_, ext)| ext)
        })
        .for_each(|ext| *counts.entry(ext).or_default() += 1);"""

new_parse = """/// Parses a list of files into an [`Extension`] summary.
fn parse_extensions(files: &[File], max_extensions: usize) -> Option<Extension> {
    let all_files: Vec<&File> = files.iter().filter(|f| !f.is_dir).collect();
    let total_files = all_files.len();
    if total_files == 0 {
        return None;
    }

    // Count files by extension; files without extensions are tracked as "(no ext)"
    let mut counts = HashMap::<&str, usize>::new();
    all_files
        .iter()
        .map(|f| {
            let file_name = f.path.rsplit_once(['/', '\\']).map_or(f.path.as_str(), |(_, name)| name);
            file_name
                .rsplit_once('.')
                .filter(|(prefix, _)| !prefix.is_empty())
                .map_or("(no ext)", |(_, ext)| ext)
        })
        .for_each(|ext| *counts.entry(ext).or_default() += 1);"""

content = content.replace(old_parse, new_parse)

old_test_sorts = """    #[test]
    fn test_parse_extensions_sorts_git_output() {
        let fixture = include_str!("fixtures/git_ls_files_mixed.txt");
        let actual = parse_extensions(fixture, MAX_EXTENSIONS).unwrap();"""

new_test_sorts = """    fn lines_to_files(lines: &str) -> Vec<File> {
        lines
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| File {
                path: line.to_string(),
                is_dir: false,
            })
            .collect()
    }

    #[test]
    fn test_parse_extensions_sorts_git_output() {
        let fixture = include_str!("fixtures/git_ls_files_mixed.txt");
        let files = lines_to_files(fixture);
        let actual = parse_extensions(&files, MAX_EXTENSIONS).unwrap();"""

content = content.replace(old_test_sorts, new_test_sorts)

old_test_truncates = """    #[test]
    fn test_parse_extensions_truncates_to_max() {
        // Real `git ls-files` output from this repo: 822 files, 19 distinct extensions.
        // Top 15 are shown; the remaining 4 (html, jsonl, lock, proto — 1 each) are
        // rolled up.
        let fixture = include_str!("fixtures/git_ls_files_many_extensions.txt");
        let actual = parse_extensions(fixture, MAX_EXTENSIONS).unwrap();"""

new_test_truncates = """    #[test]
    fn test_parse_extensions_truncates_to_max() {
        // Real `git ls-files` output from this repo: 822 files, 19 distinct extensions.
        // Top 15 are shown; the remaining 4 (html, jsonl, lock, proto — 1 each) are
        // rolled up.
        let fixture = include_str!("fixtures/git_ls_files_many_extensions.txt");
        let files = lines_to_files(fixture);
        let actual = parse_extensions(&files, MAX_EXTENSIONS).unwrap();"""

content = content.replace(old_test_truncates, new_test_truncates)

old_test_none = """    #[test]
    fn test_parse_extensions_returns_none_for_empty_output() {
        assert_eq!(parse_extensions("", MAX_EXTENSIONS), None);
        assert_eq!(parse_extensions("   \\n  \\n", MAX_EXTENSIONS), None);
    }"""

new_test_none = """    #[test]
    fn test_parse_extensions_returns_none_for_empty_output() {
        assert_eq!(parse_extensions(&[], MAX_EXTENSIONS), None);
    }"""

content = content.replace(old_test_none, new_test_none)

with open("crates/forge_app/src/system_prompt.rs", "w") as f:
    f.write(content)
