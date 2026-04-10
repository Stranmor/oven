import os
test_code = """
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::collections::{BTreeMap, HashMap};
    use futures::stream::Stream;
    use std::pin::Pin;

    use anyhow::Result;
    use async_trait::async_trait;
    use forge_app::{CommandInfra, EnvironmentInfra, FileReaderInfra, WalkerInfra, WalkedFile, Walker, WorkspaceService};
    use forge_domain::{
        AnyProvider, AuthCredential, AuthDetails, CodeSearchQuery, CommandOutput,
        ConfigOperation, Environment, FileHash, FileUpload, FileUploadInfo, MigrationResult, Node,
        ProviderId, ProviderTemplate, WorkspaceAuth, WorkspaceId, WorkspaceInfo,
        ApiKey, UserId, SearchParams, WorkspaceFiles, FileDeletion, FileInfo
    };
    use forge_config::ForgeConfig;
    use fake::{Fake, Faker};

    use crate::fd::FileDiscovery;
    use super::ForgeWorkspaceService;
    use pretty_assertions::assert_eq;

    #[derive(Clone)]
    struct MockInfra {
        pub workspaces: Vec<WorkspaceInfo>,
        pub search_limit: Arc<std::sync::Mutex<Option<usize>>>,
        pub max_sem_search_results: usize,
    }

    impl MockInfra {
        fn new(max_sem_search_results: usize) -> Self {
            Self {
                workspaces: vec![],
                search_limit: Arc::new(std::sync::Mutex::new(None)),
                max_sem_search_results,
            }
        }
    }

    #[async_trait]
    impl forge_domain::ProviderRepository for MockInfra {
        async fn get_all_providers(&self) -> Result<Vec<AnyProvider>> { Ok(vec![]) }
        async fn get_provider(&self, _id: ProviderId) -> Result<ProviderTemplate> { Err(anyhow::anyhow!("unimplemented")) }
        async fn upsert_credential(&self, _c: AuthCredential) -> Result<()> { Ok(()) }
        async fn get_credential(&self, id: &ProviderId) -> Result<Option<AuthCredential>> {
            if id == &ProviderId::FORGE_SERVICES {
                let mut url_params = HashMap::new();
                url_params.insert("user_id".to_string().into(), "test-user".to_string().into());
                Ok(Some(AuthCredential {
                    id: ProviderId::FORGE_SERVICES,
                    auth_details: AuthDetails::ApiKey(ApiKey::from("test-token".to_string())),
                    url_params,
                }))
            } else {
                Ok(None)
            }
        }
        async fn remove_credential(&self, _id: &ProviderId) -> Result<()> { Ok(()) }
        async fn migrate_env_credentials(&self) -> Result<Option<MigrationResult>> { Ok(None) }
    }

    #[async_trait]
    impl forge_domain::WorkspaceIndexRepository for MockInfra {
        async fn authenticate(&self) -> Result<WorkspaceAuth> { Err(anyhow::anyhow!("unimplemented")) }
        async fn create_workspace(&self, _dir: &Path, _token: &ApiKey) -> Result<WorkspaceId> { Err(anyhow::anyhow!("unimplemented")) }
        async fn upload_files(&self, _u: &FileUpload, _t: &ApiKey) -> Result<FileUploadInfo> { Ok(FileUploadInfo::new(0, 0)) }
        async fn search(&self, q: &CodeSearchQuery<'_>, _t: &ApiKey) -> Result<Vec<Node>> {
            *self.search_limit.lock().unwrap() = q.data.limit;
            Ok(vec![])
        }
        async fn list_workspaces(&self, _t: &ApiKey) -> Result<Vec<WorkspaceInfo>> {
            Ok(self.workspaces.clone())
        }
        async fn get_workspace(&self, _id: &WorkspaceId, _t: &ApiKey) -> Result<Option<WorkspaceInfo>> { Err(anyhow::anyhow!("unimplemented")) }
        async fn list_workspace_files(&self, _id: &WorkspaceFiles, _t: &ApiKey) -> Result<Vec<FileHash>> { Err(anyhow::anyhow!("unimplemented")) }
        async fn delete_files(&self, _d: &FileDeletion, _t: &ApiKey) -> Result<()> { Ok(()) }
        async fn delete_workspace(&self, _id: &WorkspaceId, _t: &ApiKey) -> Result<()> { Ok(()) }
    }

    #[async_trait]
    impl FileReaderInfra for MockInfra {
        async fn read(&self, _p: &Path) -> Result<Vec<u8>> { Ok(vec![]) }
        async fn read_utf8(&self, _p: &Path) -> Result<String> { Ok("".into()) }
        fn read_batch_utf8(&self, _limit: usize, _paths: Vec<PathBuf>) -> impl Stream<Item = (PathBuf, Result<String>)> + Send {
            futures::stream::empty()
        }
        async fn range_read_utf8(&self, _p: &Path, _s: u64, _e: u64) -> Result<(String, FileInfo)> { Err(anyhow::anyhow!("unimplemented")) }
    }

    impl EnvironmentInfra for MockInfra {
        type Config = ForgeConfig;
        fn get_environment(&self) -> Environment { Faker.fake() }
        fn get_config(&self) -> Result<ForgeConfig> {
            let mut cfg = ForgeConfig::default();
            cfg.max_sem_search_results = self.max_sem_search_results;
            Ok(cfg)
        }
        async fn update_environment(&self, _ops: Vec<ConfigOperation>) -> Result<()> { Ok(()) }
        fn get_env_var(&self, _k: &str) -> Option<String> { None }
        fn get_env_vars(&self) -> BTreeMap<String, String> { BTreeMap::new() }
    }

    #[async_trait]
    impl CommandInfra for MockInfra {
        async fn execute_command(
            &self,
            command: String,
            _working_dir: PathBuf,
            _keep_ansi: bool,
            _silent: bool,
            _env_vars: Option<Vec<String>>,
        ) -> Result<CommandOutput> { Ok(CommandOutput { command, exit_code: Some(0), stdout: "".into(), stderr: "".into() }) }
        async fn execute_command_raw(
            &self,
            _command: &str,
            _working_dir: PathBuf,
            _env_vars: Option<Vec<String>>,
        ) -> Result<std::process::ExitStatus> { Err(anyhow::anyhow!("unimplemented")) }
    }

    #[async_trait]
    impl WalkerInfra for MockInfra {
        async fn walk(&self, _c: Walker) -> Result<Vec<WalkedFile>> { Ok(vec![]) }
    }

    struct MockDiscovery;
    #[async_trait]
    impl FileDiscovery for MockDiscovery {
        async fn discover(&self, _p: &Path) -> Result<Vec<PathBuf>> { Ok(vec![]) }
    }

    #[tokio::test]
    async fn test_is_indexed_not_found() {
        let infra = Arc::new(MockInfra::new(10));
        let discovery = Arc::new(MockDiscovery);
        let service = ForgeWorkspaceService::new(infra, discovery);

        let result = service.is_indexed(Path::new("/definitely/does/not/exist/ever/12345")).await.unwrap();
        assert_eq!(result, false);
    }

    #[tokio::test]
    async fn test_query_workspace_enforces_limit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let canonical_path = temp_dir.path().canonicalize().unwrap();

        let mut infra_val = MockInfra::new(15);
        infra_val.workspaces.push(WorkspaceInfo {
            workspace_id: WorkspaceId::generate(),
            working_dir: canonical_path.to_string_lossy().to_string(),
            node_count: None,
            relation_count: None,
            last_updated: None,
            created_at: chrono::Utc::now(),
        });

        let infra = Arc::new(infra_val);
        let discovery = Arc::new(MockDiscovery);
        let service = ForgeWorkspaceService::new(infra.clone(), discovery);

        // Limit > config max (20 > 15) -> clamped to 15
        let mut params = SearchParams::new("query", "use_case");
        params.limit = Some(20);
        let _ = service.query_workspace(temp_dir.path().to_path_buf(), params.clone()).await.unwrap();
        assert_eq!(*infra.search_limit.lock().unwrap(), Some(15));

        // Limit < config max (10 < 15) -> kept at 10
        params.limit = Some(10);
        let _ = service.query_workspace(temp_dir.path().to_path_buf(), params.clone()).await.unwrap();
        assert_eq!(*infra.search_limit.lock().unwrap(), Some(10));

        // No limit -> uses config max (15)
        params.limit = None;
        let _ = service.query_workspace(temp_dir.path().to_path_buf(), params.clone()).await.unwrap();
        assert_eq!(*infra.search_limit.lock().unwrap(), Some(15));
    }

    #[tokio::test]
    async fn test_workspace_credentials_extraction() {
        let infra = Arc::new(MockInfra::new(10));
        let discovery = Arc::new(MockDiscovery);
        let service = ForgeWorkspaceService::new(infra, discovery);

        let (token, user_id) = service.get_workspace_credentials().await.unwrap();
        assert_eq!(token, forge_domain::ApiKey::from("test-token".to_string()));
        assert_eq!(user_id.to_string(), "test-user");
    }

    #[tokio::test]
    async fn test_sync_workspace_emits_starting() {
        let temp_dir = tempfile::tempdir().unwrap();
        let canonical_path = temp_dir.path().canonicalize().unwrap();

        let mut infra_val = MockInfra::new(15);
        infra_val.workspaces.push(WorkspaceInfo {
            workspace_id: WorkspaceId::generate(),
            working_dir: canonical_path.to_string_lossy().to_string(),
            node_count: None,
            relation_count: None,
            last_updated: None,
            created_at: chrono::Utc::now(),
        });

        let infra = Arc::new(infra_val);
        let discovery = Arc::new(MockDiscovery);
        let service = ForgeWorkspaceService::new(infra.clone(), discovery);

        let mut stream = service.sync_workspace(temp_dir.path().to_path_buf()).await.unwrap();
        
        use futures::stream::StreamExt;
        if let Some(Ok(forge_domain::SyncProgress::Starting)) = stream.next().await {
            // Expected
        } else {
            panic!("Expected SyncProgress::Starting");
        }
    }
}
"""
with open("crates/forge_services/src/context_engine.rs", "a") as f:
    f.write(test_code)
