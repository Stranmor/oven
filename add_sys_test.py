import os
test_code = """
    #[derive(Clone)]
    struct MockServices {
        files_to_return: Vec<File>,
    }

    #[async_trait::async_trait]
    impl crate::FileDiscoveryService for MockServices {
        async fn collect_files(&self, _config: crate::Walker) -> anyhow::Result<Vec<File>> {
            Ok(self.files_to_return.clone())
        }
        async fn list_current_directory(&self) -> anyhow::Result<Vec<File>> {
            Ok(vec![])
        }
    }

    #[async_trait::async_trait]
    impl crate::SkillFetchService for MockServices {
        async fn fetch_skill(&self, _name: &str) -> anyhow::Result<forge_domain::Skill> {
            Err(anyhow::anyhow!("not implemented"))
        }
        async fn list_skills(&self) -> anyhow::Result<Vec<forge_domain::Skill>> {
            Ok(vec![])
        }
    }

    #[async_trait::async_trait]
    impl crate::ShellService for MockServices {
        async fn execute(
            &self,
            _command: String,
            _cwd: std::path::PathBuf,
            _keep_ansi: bool,
            _silent: bool,
            _env_vars: Option<Vec<String>>,
            _description: Option<String>,
        ) -> anyhow::Result<forge_domain::ShellOutput> {
            Ok(forge_domain::ShellOutput { output: forge_domain::CommandOutput { exit_code: Some(0), stdout: "".into(), stderr: "".into() }, command: "".into() })
        }
    }

    #[tokio::test]
    async fn test_system_prompt_extension_integration() {
        let files = vec![
            File { path: "src/main.rs".into(), is_dir: false },
            File { path: "src/lib.rs".into(), is_dir: false },
            File { path: "README.md".into(), is_dir: false },
        ];
        
        let services = Arc::new(MockServices { files_to_return: files });
        let env = Environment::default();
        let agent = Agent::default();
        
        let system_prompt = SystemPrompt::new(services, env, agent).max_extensions(15);
        
        let conversation = Conversation::default();
        let result = system_prompt.add_system_message(conversation).await.unwrap();
        
        let ctx = result.context.unwrap();
        let context_text = ctx.to_string();
        
        // Assert extensions were fetched and integrated into context
        assert!(context_text.contains("<workspace_extensions"));
        assert!(context_text.contains(".rs"));
        assert!(context_text.contains(".md"));
    }
"""
with open("crates/forge_app/src/system_prompt.rs", "a") as f:
    f.write(test_code)
