use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use bstr::ByteSlice;
use forge_app::{
    FileInfoInfra, FileReaderInfra, FsSearchService, Match, MatchResult, SearchResult, Walker,
    WalkerInfra,
};

use forge_domain::{FSSearch, OutputMode};
use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::{Searcher, SearcherBuilder, Sink, SinkContext, SinkContextKind, SinkMatch};

const DEFAULT_AGENT_SEARCH_VISITED_FILE_LIMIT: usize = 10_000;
const DEFAULT_AGENT_SEARCH_MAX_FILE_SIZE: u64 = 32 * 1024 * 1024;

/// A powerful search tool built on grep-matcher and grep-searcher crates.
/// Supports regex patterns, file type filtering, output modes, context lines,
/// and multiline matching.
pub struct ForgeFsSearch<W> {
    infra: Arc<W>,
}

impl<W> ForgeFsSearch<W> {
    /// Creates a new filesystem search service.
    pub fn new(infra: Arc<W>) -> Self {
        Self { infra }
    }
}

#[async_trait::async_trait]
impl<W: WalkerInfra + FileReaderInfra + FileInfoInfra> FsSearchService for ForgeFsSearch<W> {
    async fn search(&self, params: FSSearch) -> anyhow::Result<Option<SearchResult>> {
        // Determine search path (default to current directory)
        let search_path = match &params.path {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => std::env::current_dir()
                .with_context(|| "Failed to get current working directory")?,
        };

        // Validate path exists
        if !self.infra.exists(&search_path).await? {
            return Err(anyhow!("Path does not exist: {}", search_path.display()));
        }

        // Build regex matcher
        let matcher = self.build_matcher(&params)?;

        // Determine output mode (default to FilesWithMatches)
        let output_mode = params
            .output_mode
            .as_ref()
            .unwrap_or(&OutputMode::FilesWithMatches);

        // Execute search lazily after infra traversal so head_limit can stop file reads.
        let matches = self
            .search_matching_files(&search_path, &matcher, &params, output_mode)
            .await?;

        if matches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(SearchResult { matches }))
        }
    }
}

impl<W: WalkerInfra + FileReaderInfra + FileInfoInfra> ForgeFsSearch<W> {
    /// Builds a regex matcher from search parameters.
    fn build_matcher(&self, params: &FSSearch) -> anyhow::Result<grep_regex::RegexMatcher> {
        let mut builder = RegexMatcherBuilder::new();

        if params.case_insensitive.unwrap_or(false) {
            builder.case_insensitive(true);
        }

        if params.multiline.unwrap_or(false) {
            builder.multi_line(true);
            builder.dot_matches_new_line(true);
        }

        builder
            .build(&params.pattern)
            .with_context(|| format!("Invalid regex pattern: {}", params.pattern))
    }

    fn walker_file_limit(_params: &FSSearch, _output_mode: &OutputMode) -> usize {
        DEFAULT_AGENT_SEARCH_VISITED_FILE_LIMIT
    }

    fn search_budget(&self, params: &FSSearch, _output_mode: &OutputMode) -> SearchBudget {
        let offset = params.offset.unwrap_or(0) as usize;
        let limit = params.head_limit.map(|limit| limit as usize);
        SearchBudget::new(offset, limit)
    }

    async fn search_matching_files(
        &self,
        search_path: &Path,
        matcher: &grep_regex::RegexMatcher,
        params: &FSSearch,
        output_mode: &OutputMode,
    ) -> anyhow::Result<Vec<Match>> {
        let types_matcher = Self::build_types_matcher(params)?;
        let glob_patterns = Self::build_glob_patterns(params)?;
        let mut budget = self.search_budget(params, output_mode);
        let mut matches = Vec::new();

        if budget.is_satisfied() {
            return Ok(matches);
        }

        if self.infra.is_file(search_path).await? {
            if Self::matches_file_filters_sync(
                search_path,
                search_path.parent().unwrap_or(search_path),
                &glob_patterns,
                types_matcher.as_ref(),
            )? {
                self.search_one_file(
                    search_path,
                    matcher,
                    params,
                    output_mode,
                    &mut budget,
                    &mut matches,
                )
                .await?;
            }
            return Ok(matches);
        }

        let mut visited_files = 0usize;
        let mut traversal_limited = false;
        let walker_file_limit = Self::walker_file_limit(params, output_mode);
        let walked_files = self
            .infra
            .walk(
                Walker::unlimited()
                    .cwd(search_path.to_path_buf())
                    .max_files(walker_file_limit),
            )
            .await
            .with_context(|| format!("Failed to walk directory '{}'", search_path.display()))?;
        for walked_file in walked_files {
            if budget.is_satisfied() {
                break;
            }
            if visited_files >= DEFAULT_AGENT_SEARCH_VISITED_FILE_LIMIT {
                traversal_limited = true;
                break;
            }
            if walked_file.is_dir() {
                continue;
            }

            let path = search_path.join(&walked_file.path);
            if !self.infra.is_file(&path).await? {
                continue;
            }

            visited_files = visited_files.saturating_add(1);
            if !Self::matches_file_filters_sync(
                &path,
                search_path,
                &glob_patterns,
                types_matcher.as_ref(),
            )? {
                continue;
            }

            self.search_one_file(
                &path,
                matcher,
                params,
                output_mode,
                &mut budget,
                &mut matches,
            )
            .await?;
        }

        if traversal_limited {
            matches.push(Match {
                path: search_path.to_string_lossy().to_string(),
                result: Some(MatchResult::Error(format!(
                    "Search stopped after visiting {DEFAULT_AGENT_SEARCH_VISITED_FILE_LIMIT} files; refine path, glob, type, or head_limit"
                ))),
            });
        }

        Ok(matches)
    }

    fn build_types_matcher(params: &FSSearch) -> anyhow::Result<Option<ignore::types::Types>> {
        if let Some(file_type) = params.file_type.as_deref().filter(|s| !s.is_empty()) {
            use ignore::types::TypesBuilder;

            let mut builder = TypesBuilder::new();
            builder.add_defaults();
            builder.select(file_type);

            Ok(Some(builder.build().with_context(|| {
                format!("Failed to build type matcher for: {file_type}")
            })?))
        } else {
            Ok(None)
        }
    }

    fn build_glob_patterns(params: &FSSearch) -> anyhow::Result<Option<Vec<glob::Pattern>>> {
        let Some(glob_pattern) = params.glob.as_deref().filter(|pattern| !pattern.is_empty())
        else {
            return Ok(None);
        };

        let expanded = Self::expand_brace_glob(glob_pattern);
        let mut patterns = Vec::with_capacity(expanded.len());
        for pattern in expanded {
            patterns.push(
                glob::Pattern::new(&pattern)
                    .with_context(|| format!("Invalid glob pattern: {glob_pattern}"))?,
            );
        }
        Ok(Some(patterns))
    }

    fn expand_brace_glob(pattern: &str) -> Vec<String> {
        let Some(open) = pattern.find('{') else {
            return vec![pattern.to_string()];
        };
        let search_start = open.saturating_add(1);
        let Some(close_offset) = pattern[search_start..].find('}') else {
            return vec![pattern.to_string()];
        };
        let close = search_start.saturating_add(close_offset);
        let prefix = &pattern[..open];
        let suffix_start = close.saturating_add(1);
        let suffix = &pattern[suffix_start..];
        let body = &pattern[search_start..close];

        body.split(',')
            .filter(|part| !part.is_empty())
            .map(|part| format!("{prefix}{part}{suffix}"))
            .collect()
    }

    fn matches_file_filters_sync(
        path: &Path,
        search_root: &Path,
        glob_patterns: &Option<Vec<glob::Pattern>>,
        types_matcher: Option<&ignore::types::Types>,
    ) -> anyhow::Result<bool> {
        if let Some(patterns) = glob_patterns {
            let matches = patterns.iter().any(|pattern| {
                if pattern.as_str().contains('/') {
                    path.strip_prefix(search_root)
                        .map(|relative| pattern.matches_path(relative))
                        .unwrap_or(false)
                } else {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|file_name| pattern.matches(file_name))
                        .unwrap_or(false)
                }
            });

            if !matches {
                return Ok(false);
            }
        }

        if let Some(types) = types_matcher
            && !types.matched(path, false).is_whitelist()
        {
            return Ok(false);
        }

        Ok(true)
    }

    async fn search_one_file(
        &self,
        path: &Path,
        matcher: &grep_regex::RegexMatcher,
        params: &FSSearch,
        output_mode: &OutputMode,
        budget: &mut SearchBudget,
        matches: &mut Vec<Match>,
    ) -> anyhow::Result<()> {
        if budget.is_satisfied()
            || self.infra.file_size(path).await? > DEFAULT_AGENT_SEARCH_MAX_FILE_SIZE
            || self.infra.is_binary(path).await?
        {
            return Ok(());
        }

        match output_mode {
            OutputMode::FilesWithMatches => {
                if self
                    .file_has_match(path, matcher, params.multiline.unwrap_or(false))
                    .await?
                    && budget.accept_match()
                {
                    matches.push(Match {
                        path: path.to_string_lossy().to_string(),
                        result: Some(MatchResult::FileMatch),
                    });
                }
            }
            OutputMode::Content => {
                self.search_content_file(path, matcher, params, budget, matches)
                    .await?;
            }
            OutputMode::Count => {
                let count = self
                    .search_file_count(path, matcher, params.multiline.unwrap_or(false))
                    .await?;
                if count > 0 && budget.accept_match() {
                    matches.push(Match {
                        path: path.to_string_lossy().to_string(),
                        result: Some(MatchResult::Count { count }),
                    });
                }
            }
        }

        Ok(())
    }

    async fn file_has_match(
        &self,
        path: &Path,
        matcher: &grep_regex::RegexMatcher,
        multiline: bool,
    ) -> anyhow::Result<bool> {
        let content = self.infra.read(path).await?;
        let mut has_match = false;
        SearcherBuilder::new()
            .multi_line(multiline)
            .build()
            .search_slice(
                matcher,
                &content,
                UTF8(|_, _| {
                    has_match = true;
                    Ok(false)
                }),
            )?;
        Ok(has_match)
    }

    async fn search_file_count(
        &self,
        path: &Path,
        matcher: &grep_regex::RegexMatcher,
        multiline: bool,
    ) -> anyhow::Result<usize> {
        let content = self.infra.read(path).await?;
        let mut count = 0usize;
        SearcherBuilder::new()
            .multi_line(multiline)
            .build()
            .search_slice(
                matcher,
                &content,
                UTF8(|_, _| {
                    count = count.saturating_add(1);
                    Ok(true)
                }),
            )?;
        Ok(count)
    }

    async fn search_content_file(
        &self,
        path: &Path,
        matcher: &grep_regex::RegexMatcher,
        params: &FSSearch,
        budget: &mut SearchBudget,
        all_matches: &mut Vec<Match>,
    ) -> anyhow::Result<()> {
        let show_line_numbers = params.show_line_numbers.unwrap_or(true);
        let has_context = params.context.is_some()
            || params.before_context.is_some()
            || params.after_context.is_some();

        let mut searcher_builder = SearcherBuilder::new();
        searcher_builder.line_number(true);
        searcher_builder.multi_line(params.multiline.unwrap_or(false));

        let (before_context_limit, after_context_limit) = if let Some(context) = params.context {
            let context = context as usize;
            searcher_builder.before_context(context);
            searcher_builder.after_context(context);
            (context, context)
        } else {
            let before = params.before_context.unwrap_or(0) as usize;
            if before > 0 {
                searcher_builder.before_context(before);
            }
            let after = params.after_context.unwrap_or(0) as usize;
            if after > 0 {
                searcher_builder.after_context(after);
            }
            (before, after)
        };

        let mut searcher = searcher_builder.build();
        let content = self.infra.read(path).await?;
        let path_string = path.to_string_lossy().to_string();

        if has_context {
            let collection_limit = budget.collection_limit();
            let mut sink = ContextSink::new(
                path_string,
                show_line_numbers,
                collection_limit,
                before_context_limit,
                after_context_limit,
            );
            searcher.search_slice(matcher, &content, &mut sink)?;
            for search_match in sink.into_matches(true) {
                if budget.accept_match() {
                    all_matches.push(search_match);
                }
                if budget.is_satisfied() {
                    break;
                }
            }
        } else {
            searcher.search_slice(
                matcher,
                &content,
                UTF8(|line_num, line| {
                    if budget.accept_match() {
                        all_matches.push(Match {
                            path: path_string.clone(),
                            result: Some(MatchResult::Found {
                                line_number: if show_line_numbers {
                                    Some(usize::try_from(line_num).unwrap_or(usize::MAX))
                                } else {
                                    None
                                },
                                line: line.trim_end().to_string(),
                            }),
                        });
                    }
                    Ok(!budget.is_satisfied())
                }),
            )?;
        }

        Ok(())
    }
}

struct SearchBudget {
    skipped_matches: usize,
    offset: usize,
    remaining: Option<usize>,
}

impl SearchBudget {
    fn new(offset: usize, limit: Option<usize>) -> Self {
        Self { skipped_matches: 0, offset, remaining: limit }
    }

    fn accept_match(&mut self) -> bool {
        if self.skipped_matches < self.offset {
            self.skipped_matches = self.skipped_matches.saturating_add(1);
            return false;
        }

        match &mut self.remaining {
            Some(0) => false,
            Some(remaining) => {
                *remaining = remaining.saturating_sub(1);
                true
            }
            None => true,
        }
    }

    fn is_satisfied(&self) -> bool {
        self.remaining == Some(0)
    }

    fn collection_limit(&self) -> usize {
        match self.remaining {
            Some(remaining) => {
                remaining.saturating_add(self.offset.saturating_sub(self.skipped_matches))
            }
            None => usize::MAX,
        }
    }
}

/// Custom sink for capturing matches with context lines.
///
/// This sink implements the full `Sink` trait from grep-searcher to capture
/// both matches and their surrounding context lines. The grep-searcher's
/// convenience sinks (UTF8, Lossy, Bytes) only report matches and ignore
/// context, so we need a custom implementation.
struct ContextSink {
    path: String,
    show_line_numbers: bool,
    matches: Vec<Match>,
    before_context: Vec<String>,
    current_match: Option<(usize, String)>,
    current_before_context: Vec<String>,
    current_after_context: Vec<String>,
    match_limit: usize,
    after_context_limit: usize,
    before_context_limit: usize,
}

impl ContextSink {
    fn new(
        path: String,
        show_line_numbers: bool,
        match_limit: usize,
        before_context_limit: usize,
        after_context_limit: usize,
    ) -> Self {
        Self {
            path,
            show_line_numbers,
            matches: Vec::new(),
            before_context: Vec::new(),
            current_match: None,
            current_before_context: Vec::new(),
            current_after_context: Vec::new(),
            match_limit,
            after_context_limit,
            before_context_limit,
        }
    }

    fn flush_current(&mut self, has_context: bool) {
        if let Some((line_num, line)) = self.current_match.take() {
            if has_context {
                self.matches.push(Match {
                    path: self.path.clone(),
                    result: Some(MatchResult::ContextMatch {
                        line_number: if self.show_line_numbers {
                            Some(line_num)
                        } else {
                            None
                        },
                        line,
                        before_context: self.current_before_context.clone(),
                        after_context: self.current_after_context.clone(),
                    }),
                });
            } else {
                self.matches.push(Match {
                    path: self.path.clone(),
                    result: Some(MatchResult::Found {
                        line_number: if self.show_line_numbers {
                            Some(line_num)
                        } else {
                            None
                        },
                        line,
                    }),
                });
            }
        }
    }

    fn into_matches(mut self, has_context: bool) -> Vec<Match> {
        self.flush_current(has_context);
        self.matches
    }
}

impl Sink for ContextSink {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        let had_pending_match = self.current_match.is_some();
        let previous_after_context = if had_pending_match {
            self.current_after_context.clone()
        } else {
            Vec::new()
        };
        self.flush_current(true);
        if had_pending_match {
            self.current_after_context.clear();
        }

        if self.matches.len() >= self.match_limit {
            return Ok(false);
        }

        let line_num = usize::try_from(mat.line_number().unwrap_or(0)).unwrap_or(usize::MAX);
        let line = mat.bytes().to_str_lossy().trim_end().to_string();
        self.current_match = Some((line_num, line));
        let mut before_context = std::mem::take(&mut self.before_context);
        if before_context.is_empty() && !previous_after_context.is_empty() {
            before_context = previous_after_context
                .into_iter()
                .rev()
                .take(self.before_context_limit)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
        }
        self.current_before_context = before_context;
        if self.after_context_limit == 0 && self.matches.len().saturating_add(1) >= self.match_limit
        {
            self.flush_current(true);
            return Ok(false);
        }

        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        ctx: &SinkContext<'_>,
    ) -> Result<bool, Self::Error> {
        let line = ctx.bytes().to_str_lossy().trim_end().to_string();
        match ctx.kind() {
            SinkContextKind::Before => {
                self.before_context.push(line);
            }
            SinkContextKind::After => {
                self.current_after_context.push(line);
                if self.current_match.is_some()
                    && self.current_after_context.len() >= self.after_context_limit
                    && self.matches.len().saturating_add(1) >= self.match_limit
                {
                    self.flush_current(true);
                    return Ok(false);
                }
            }
            _ => {}
        }

        Ok(true)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use forge_app::{FileReaderInfra, WalkedFile, Walker, WalkerInfra};
    use forge_domain::{FSSearch, OutputMode};
    use pretty_assertions::assert_eq;
    use tokio::fs;

    use super::*;
    use crate::utils::TempDir;

    // Mock infrastructure for testing
    struct MockInfra {
        binary_exts: HashSet<String>,
    }

    impl Default for MockInfra {
        fn default() -> Self {
            let binary_exts = [
                "exe", "dll", "so", "dylib", "bin", "obj", "o", "class", "pyc", "jar", "war",
                "ear", "zip", "tar", "gz", "rar", "7z", "iso", "img", "pdf", "doc", "docx", "xls",
                "xlsx", "ppt", "pptx", "bmp", "ico", "mp3", "mp4", "avi", "mov", "sqlite", "db",
            ];
            Self {
                binary_exts: HashSet::from_iter(binary_exts.into_iter().map(|ext| ext.to_string())),
            }
        }
    }

    #[async_trait::async_trait]
    impl FileReaderInfra for MockInfra {
        async fn read_utf8(&self, _path: &Path) -> anyhow::Result<String> {
            unimplemented!()
        }

        fn read_batch_utf8(
            &self,
            _batch_size: usize,
            _paths: Vec<PathBuf>,
        ) -> impl futures::Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
            futures::stream::empty()
        }

        async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
            fs::read(path)
                .await
                .with_context(|| format!("Failed to read file '{}'", path.display()))
        }

        async fn range_read_utf8(
            &self,
            _path: &Path,
            _start_line: u64,
            _end_line: u64,
        ) -> anyhow::Result<(String, forge_domain::FileInfo)> {
            unimplemented!()
        }
    }

    #[async_trait::async_trait]
    impl FileInfoInfra for MockInfra {
        async fn is_file(&self, path: &Path) -> anyhow::Result<bool> {
            let metadata = tokio::fs::metadata(path).await;
            match metadata {
                Ok(meta) => Ok(meta.is_file()),
                Err(_) => Ok(false),
            }
        }

        async fn is_binary(&self, path: &Path) -> anyhow::Result<bool> {
            let ext = path.extension().and_then(|s| s.to_str());
            Ok(self.binary_exts.contains(ext.unwrap_or("")))
        }

        async fn exists(&self, path: &Path) -> anyhow::Result<bool> {
            Ok(tokio::fs::metadata(path).await.is_ok())
        }

        async fn file_size(&self, path: &Path) -> anyhow::Result<u64> {
            Ok(tokio::fs::metadata(path).await?.len())
        }
    }

    async fn collect_walked_files(root: &Path) -> anyhow::Result<Vec<WalkedFile>> {
        let mut files = Vec::new();
        let mut pending = vec![root.to_path_buf()];

        while let Some(directory) = pending.pop() {
            let mut entries = fs::read_dir(&directory).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                let metadata = fs::symlink_metadata(&path).await?;
                if metadata.file_type().is_symlink() {
                    continue;
                }

                let relative_path = path.strip_prefix(root)?.to_string_lossy().to_string();
                let file_name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string());
                if metadata.is_dir() {
                    pending.push(path);
                    files.push(WalkedFile {
                        path: format!("{relative_path}/"),
                        file_name,
                        size: metadata.len(),
                    });
                    continue;
                }

                if metadata.is_file() {
                    files.push(WalkedFile { path: relative_path, file_name, size: metadata.len() });
                }
            }
        }

        Ok(files)
    }

    #[async_trait::async_trait]
    impl WalkerInfra for MockInfra {
        async fn walk(&self, config: Walker) -> anyhow::Result<Vec<WalkedFile>> {
            collect_walked_files(&config.cwd).await
        }
    }

    #[derive(Clone, Default)]
    struct CountingFileInfoInfra {
        file_size_calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl FileInfoInfra for CountingFileInfoInfra {
        async fn is_file(&self, path: &Path) -> anyhow::Result<bool> {
            Ok(tokio::fs::metadata(path)
                .await
                .is_ok_and(|metadata| metadata.is_file()))
        }

        async fn is_binary(&self, _path: &Path) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn exists(&self, path: &Path) -> anyhow::Result<bool> {
            Ok(tokio::fs::metadata(path).await.is_ok())
        }

        async fn file_size(&self, path: &Path) -> anyhow::Result<u64> {
            self.file_size_calls.fetch_add(1, Ordering::SeqCst);
            Ok(tokio::fs::metadata(path).await?.len())
        }
    }

    #[async_trait::async_trait]
    impl FileReaderInfra for CountingFileInfoInfra {
        async fn read_utf8(&self, _path: &Path) -> anyhow::Result<String> {
            unimplemented!()
        }

        fn read_batch_utf8(
            &self,
            _batch_size: usize,
            _paths: Vec<PathBuf>,
        ) -> impl futures::Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
            futures::stream::empty()
        }

        async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
            fs::read(path)
                .await
                .with_context(|| format!("Failed to read file '{}'", path.display()))
        }

        async fn range_read_utf8(
            &self,
            _path: &Path,
            _start_line: u64,
            _end_line: u64,
        ) -> anyhow::Result<(String, forge_domain::FileInfo)> {
            unimplemented!()
        }
    }

    #[async_trait::async_trait]
    impl WalkerInfra for CountingFileInfoInfra {
        async fn walk(&self, config: Walker) -> anyhow::Result<Vec<WalkedFile>> {
            collect_walked_files(&config.cwd).await
        }
    }

    async fn create_test_directory() -> anyhow::Result<TempDir> {
        let temp_dir = TempDir::new()?;

        fs::write(
            temp_dir.path().join("test.txt"),
            "hello world\ntest line\nfoo bar",
        )
        .await?;
        fs::write(temp_dir.path().join("other.txt"), "no match here").await?;
        fs::write(
            temp_dir.path().join("code.rs"),
            "fn test() {}\nfn main() {}",
        )
        .await?;
        fs::write(temp_dir.path().join("app.js"), "function test() {}").await?;

        Ok(temp_dir)
    }

    #[tokio::test]
    async fn test_head_limit_stops_after_enough_file_matches() {
        let fixture = TempDir::new().unwrap();
        for index in 0..25 {
            fs::write(
                fixture.path().join(format!("match_{index:02}.txt")),
                "needle\n",
            )
            .await
            .unwrap();
        }
        let infra = CountingFileInfoInfra::default();
        let file_size_calls = Arc::clone(&infra.file_size_calls);
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(infra))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1;

        assert_eq!(actual.matches.len(), expected);
        assert_eq!(file_size_calls.load(Ordering::SeqCst), expected);
    }

    #[tokio::test]
    async fn test_head_limit_stops_content_scan_inside_large_file() {
        let fixture = TempDir::new().unwrap();
        let mut content = String::new();
        for index in 0..100 {
            content.push_str(&format!("needle {index}\n"));
        }
        fs::write(fixture.path().join("large.txt"), content)
            .await
            .unwrap();
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            head_limit: Some(3),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 3;

        assert_eq!(actual.matches.len(), expected);
    }

    #[tokio::test]
    async fn test_content_search_without_head_limit_returns_all_matches() {
        let fixture = TempDir::new().unwrap();
        let mut content = String::new();
        for index in 0..1100 {
            content.push_str(&format!("needle {index}\n"));
        }
        fs::write(fixture.path().join("many.txt"), content)
            .await
            .unwrap();
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1100;

        assert_eq!(actual.matches.len(), expected);
    }

    #[tokio::test]
    async fn test_head_limit_zero_returns_before_directory_walk() {
        #[derive(Default)]
        struct WalkCountingInfra {
            walk_calls: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl FileInfoInfra for WalkCountingInfra {
            async fn is_file(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(false)
            }

            async fn is_binary(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(false)
            }

            async fn exists(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(true)
            }

            async fn file_size(&self, _path: &Path) -> anyhow::Result<u64> {
                Ok(0)
            }
        }

        #[async_trait::async_trait]
        impl FileReaderInfra for WalkCountingInfra {
            async fn read_utf8(&self, _path: &Path) -> anyhow::Result<String> {
                unimplemented!()
            }

            fn read_batch_utf8(
                &self,
                _batch_size: usize,
                _paths: Vec<PathBuf>,
            ) -> impl futures::Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
                futures::stream::empty()
            }

            async fn read(&self, _path: &Path) -> anyhow::Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn range_read_utf8(
                &self,
                _path: &Path,
                _start_line: u64,
                _end_line: u64,
            ) -> anyhow::Result<(String, forge_domain::FileInfo)> {
                unimplemented!()
            }
        }

        #[async_trait::async_trait]
        impl WalkerInfra for WalkCountingInfra {
            async fn walk(&self, _config: Walker) -> anyhow::Result<Vec<WalkedFile>> {
                self.walk_calls.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            }
        }

        let fixture = WalkCountingInfra::default();
        let walk_calls = Arc::clone(&fixture.walk_calls);
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some("/tmp".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(0),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(fixture))
            .search(params)
            .await
            .unwrap();
        let expected = 0;

        assert!(actual.is_none());
        assert_eq!(walk_calls.load(Ordering::SeqCst), expected);
    }

    #[tokio::test]
    async fn test_multiline_search_matches_across_lines() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("multi.txt"), "alpha\nbeta\n")
            .await
            .unwrap();
        let params = FSSearch {
            pattern: "alpha\\nbeta".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            multiline: Some(true),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1;

        assert_eq!(actual.matches.len(), expected);
    }

    #[tokio::test]
    async fn test_glob_filter_runs_before_file_io() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("skip.txt"), "needle\n")
            .await
            .unwrap();
        fs::write(fixture.path().join("keep.rs"), "needle\n")
            .await
            .unwrap();
        let infra = CountingFileInfoInfra::default();
        let file_size_calls = Arc::clone(&infra.file_size_calls);
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("*.rs".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(infra))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1;

        assert_eq!(actual.matches.len(), expected);
        assert_eq!(file_size_calls.load(Ordering::SeqCst), expected);
    }

    #[tokio::test]
    async fn test_glob_star_with_head_limit_does_not_touch_entire_deep_tree() {
        let fixture = TempDir::new().unwrap();
        for index in 0..50 {
            let dir_path = fixture
                .path()
                .join(format!("forest/branch_{index:02}/leaf"));
            fs::create_dir_all(&dir_path).await.unwrap();
            fs::write(dir_path.join(format!("file_{index:02}.txt")), "needle\n")
                .await
                .unwrap();
        }
        let infra = CountingFileInfoInfra::default();
        let file_size_calls = Arc::clone(&infra.file_size_calls);
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("*".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(infra))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1;

        assert_eq!(actual.matches.len(), expected);
        assert_eq!(file_size_calls.load(Ordering::SeqCst), expected);
    }

    #[tokio::test]
    async fn test_safe_traversal_cap_is_forwarded_to_walker_budget() {
        #[derive(Default)]
        struct ConfigRecordingInfra {
            observed_max_files: Arc<std::sync::Mutex<Vec<Option<usize>>>>,
        }

        #[async_trait::async_trait]
        impl FileInfoInfra for ConfigRecordingInfra {
            async fn is_file(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(false)
            }

            async fn is_binary(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(false)
            }

            async fn exists(&self, _path: &Path) -> anyhow::Result<bool> {
                Ok(true)
            }

            async fn file_size(&self, _path: &Path) -> anyhow::Result<u64> {
                Ok(0)
            }
        }

        #[async_trait::async_trait]
        impl FileReaderInfra for ConfigRecordingInfra {
            async fn read_utf8(&self, _path: &Path) -> anyhow::Result<String> {
                unimplemented!()
            }

            fn read_batch_utf8(
                &self,
                _batch_size: usize,
                _paths: Vec<PathBuf>,
            ) -> impl futures::Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
                futures::stream::empty()
            }

            async fn read(&self, _path: &Path) -> anyhow::Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn range_read_utf8(
                &self,
                _path: &Path,
                _start_line: u64,
                _end_line: u64,
            ) -> anyhow::Result<(String, forge_domain::FileInfo)> {
                unimplemented!()
            }
        }

        #[async_trait::async_trait]
        impl WalkerInfra for ConfigRecordingInfra {
            async fn walk(&self, config: Walker) -> anyhow::Result<Vec<WalkedFile>> {
                self.observed_max_files
                    .lock()
                    .unwrap()
                    .push(config.max_files);
                Ok(Vec::new())
            }
        }

        let fixture = ConfigRecordingInfra::default();
        let observed_max_files = Arc::clone(&fixture.observed_max_files);
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some("/tmp".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(1),
            ..Default::default()
        };

        let _actual = ForgeFsSearch::new(Arc::new(fixture))
            .search(params)
            .await
            .unwrap();
        let expected = vec![Some(DEFAULT_AGENT_SEARCH_VISITED_FILE_LIMIT)];
        let actual = observed_max_files.lock().unwrap().clone();

        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_head_limit_skips_nonmatching_files_before_first_match() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("aaa_nonmatch.txt"), "haystack\n")
            .await
            .unwrap();
        fs::write(fixture.path().join("zzz_match.txt"), "needle\n")
            .await
            .unwrap();
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(CountingFileInfoInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = "zzz_match.txt";

        assert_eq!(actual.matches.len(), 1);
        assert!(actual.matches[0].path.ends_with(expected));
    }

    #[tokio::test]
    async fn test_symlinked_files_are_not_followed_during_directory_search() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("real.txt"), "plain\n")
            .await
            .unwrap();
        std::os::unix::fs::symlink(
            fixture.path().join("real.txt"),
            fixture.path().join("link.txt"),
        )
        .unwrap();
        let params = FSSearch {
            pattern: "plain".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("link.txt".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();
        let expected = None;

        assert_eq!(actual.map(|result| result.matches.len()), expected);
    }

    #[tokio::test]
    async fn test_basic_content_search() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        assert!(!result.matches.is_empty());
        // Should find matches in test.txt, code.rs, and app.js
        assert!(result.matches.len() >= 3);
    }

    #[tokio::test]
    async fn test_files_with_matches_mode() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should return file paths only, no content
        assert!(
            result
                .matches
                .iter()
                .all(|m| m.result.is_some() && matches!(m.result, Some(MatchResult::FileMatch)))
        );
    }

    #[tokio::test]
    async fn test_count_mode() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Count),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should return counts
        assert!(
            result
                .matches
                .iter()
                .all(|m| matches!(m.result, Some(MatchResult::Count { count: _ })))
        );
    }

    #[tokio::test]
    async fn test_empty_file_type_treated_as_none() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            file_type: Some("".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        // Should not error - empty file_type should be treated as None
        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should match files across all types (not filtered)
        assert!(result.matches.len() >= 3);
    }

    #[tokio::test]
    async fn test_glob_filter() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("*.rs".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should only match .rs files
        assert!(result.matches.iter().all(|m| m.path.ends_with(".rs")));
    }

    #[tokio::test]
    async fn test_glob_and_file_type_filters_are_composed() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("code.rs"), "needle")
            .await
            .unwrap();
        fs::write(fixture.path().join("script.js"), "needle")
            .await
            .unwrap();
        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("*.*".to_string()),
            file_type: Some("rust".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = 1;

        assert_eq!(actual.matches.len(), expected);
        assert!(actual.matches[0].path.ends_with("code.rs"));
    }

    #[tokio::test]
    async fn test_case_insensitive() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "HELLO".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            case_insensitive: Some(true),
            output_mode: Some(OutputMode::Content),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
    }

    #[tokio::test]
    async fn test_case_sensitive_default() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "HELLO".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        // Should not match because it's case-sensitive by default
        assert!(actual.is_none());
    }

    #[tokio::test]
    async fn test_line_numbers_enabled() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            show_line_numbers: Some(true),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should have line numbers
        assert!(
            result
                .matches
                .iter()
                .filter_map(|m| m.result.as_ref())
                .all(|r| matches!(r, MatchResult::Found { line_number: Some(_), .. }))
        );
    }

    #[tokio::test]
    async fn test_line_numbers_disabled() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "test".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            show_line_numbers: Some(false),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should not have line numbers
        assert!(
            result
                .matches
                .iter()
                .filter_map(|m| m.result.as_ref())
                .all(|r| matches!(r, MatchResult::Found { line_number: None, .. }))
        );
    }

    #[tokio::test]
    async fn test_head_limit_stops_after_requested_file_matches() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("first.txt"), "needle")
            .await
            .unwrap();
        fs::write(fixture.path().join("second.txt"), "needle")
            .await
            .unwrap();

        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();

        let expected = 1;
        assert_eq!(actual.matches.len(), expected);
    }

    #[tokio::test]
    async fn test_content_head_limit_stops_within_file() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("many.txt"),
            "needle 1\nneedle 2\nneedle 3\nneedle 4",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "needle".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            head_limit: Some(2),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();

        let expected = 2;
        assert_eq!(actual.matches.len(), expected);
    }

    #[tokio::test]
    async fn test_recursive_directory_search_still_finds_nested_file() {
        let fixture = TempDir::new().unwrap();
        fs::create_dir_all(fixture.path().join("src/deep"))
            .await
            .unwrap();
        fs::write(
            fixture.path().join("src/deep/lib.rs"),
            "fn nested_target() {}",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "nested_target".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            glob: Some("**/*.rs".to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();

        let expected = 1;
        assert_eq!(actual.matches.len(), expected);
        assert!(actual.matches[0].path.ends_with("src/deep/lib.rs"));
    }

    #[tokio::test]
    async fn test_path_defaults_to_cwd() {
        let params = FSSearch {
            pattern: "test".to_string(),
            path: None,
            ..Default::default()
        };

        // This should use current directory
        let result = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await;

        // Should not error (even if no matches found)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_no_matches_returns_none() {
        let fixture = create_test_directory().await.unwrap();
        let params = FSSearch {
            pattern: "nonexistent_pattern_xyz".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_none());
    }

    #[tokio::test]
    async fn test_skip_binary_files() {
        let fixture = TempDir::new().unwrap();
        fs::write(fixture.path().join("text.txt"), "hello world")
            .await
            .unwrap();
        fs::write(fixture.path().join("binary.exe"), "hello world")
            .await
            .unwrap();

        let params = FSSearch {
            pattern: "hello".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::FilesWithMatches),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        // Should only find text.txt, not binary.exe
        assert_eq!(result.matches.len(), 1);
        assert!(result.matches[0].path.ends_with("text.txt"));
    }

    #[tokio::test]
    async fn test_context_offset_applies_before_head_limit() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "line 1\nMATCH first\nline 3\nMATCH second\nline 5\nMATCH third\nline 7",
        )
        .await
        .unwrap();
        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            context: Some(1),
            offset: Some(1),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = "MATCH second";

        assert_eq!(actual.matches.len(), 1);
        match &actual.matches[0].result {
            Some(MatchResult::ContextMatch {
                line,
                before_context,
                after_context,
                line_number,
            }) => {
                assert_eq!(line, expected);
                assert_eq!(line_number, &Some(4));
                assert_eq!(before_context, &vec!["line 3".to_string()]);
                assert_eq!(after_context, &vec!["line 5".to_string()]);
            }
            _ => panic!("Expected ContextMatch, got {:?}", actual.matches[0].result),
        }
    }

    #[tokio::test]
    async fn test_context_lines_both() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "line 1\nline 2\nline 3\nMATCH HERE\nline 5\nline 6\nline 7",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            context: Some(2), // 2 lines before and after
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        assert_eq!(result.matches.len(), 1);

        // Verify it's a ContextMatch with before and after context
        match &result.matches[0].result {
            Some(MatchResult::ContextMatch {
                line,
                before_context,
                after_context,
                line_number,
            }) => {
                assert_eq!(line, "MATCH HERE");
                assert_eq!(line_number, &Some(4));
                assert_eq!(before_context.len(), 2);
                assert_eq!(before_context[0], "line 2");
                assert_eq!(before_context[1], "line 3");
                assert_eq!(after_context.len(), 2);
                assert_eq!(after_context[0], "line 5");
                assert_eq!(after_context[1], "line 6");
            }
            _ => panic!("Expected ContextMatch, got {:?}", result.matches[0].result),
        }
    }

    #[tokio::test]
    async fn test_before_context_only() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "line 1\nline 2\nline 3\nMATCH HERE\nline 5\nline 6",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            before_context: Some(2), // 2 lines before only
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        assert_eq!(result.matches.len(), 1);

        match &result.matches[0].result {
            Some(MatchResult::ContextMatch { line, before_context, after_context, .. }) => {
                assert_eq!(line, "MATCH HERE");
                assert_eq!(before_context.len(), 2);
                assert_eq!(before_context[0], "line 2");
                assert_eq!(before_context[1], "line 3");
                assert_eq!(after_context.len(), 0); // No after context
            }
            _ => panic!("Expected ContextMatch"),
        }
    }

    #[tokio::test]
    async fn test_after_context_only() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "line 1\nline 2\nMATCH HERE\nline 4\nline 5\nline 6",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            after_context: Some(2), // 2 lines after only
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        assert_eq!(result.matches.len(), 1);

        match &result.matches[0].result {
            Some(MatchResult::ContextMatch { line, before_context, after_context, .. }) => {
                assert_eq!(line, "MATCH HERE");
                assert_eq!(before_context.len(), 0); // No before context
                assert_eq!(after_context.len(), 2);
                assert_eq!(after_context[0], "line 4");
                assert_eq!(after_context[1], "line 5");
            }
            _ => panic!("Expected ContextMatch"),
        }
    }

    #[tokio::test]
    async fn test_after_context_is_preserved_when_head_limit_stops() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "MATCH HERE\nafter 1\nafter 2\nafter 3\nMATCH AGAIN",
        )
        .await
        .unwrap();
        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            after_context: Some(2),
            head_limit: Some(1),
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap()
            .unwrap();
        let expected = vec!["after 1".to_string(), "after 2".to_string()];

        match &actual.matches[0].result {
            Some(MatchResult::ContextMatch { after_context, .. }) => {
                assert_eq!(after_context, &expected);
            }
            _ => panic!("Expected ContextMatch"),
        }
    }

    #[tokio::test]
    async fn test_no_context_returns_found() {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("test.txt"),
            "line 1\nMATCH HERE\nline 3",
        )
        .await
        .unwrap();

        let params = FSSearch {
            pattern: "MATCH".to_string(),
            path: Some(fixture.path().to_string_lossy().to_string()),
            output_mode: Some(OutputMode::Content),
            // No context specified
            ..Default::default()
        };

        let actual = ForgeFsSearch::new(Arc::new(MockInfra::default()))
            .search(params)
            .await
            .unwrap();

        assert!(actual.is_some());
        let result = actual.unwrap();
        assert_eq!(result.matches.len(), 1);

        // Should be Found, not ContextMatch when no context is requested
        match &result.matches[0].result {
            Some(MatchResult::Found { line, .. }) => {
                assert_eq!(line, "MATCH HERE");
            }
            _ => panic!("Expected Found, got {:?}", result.matches[0].result),
        }
    }
}
