use std::sync::Arc;

use forge_app::domain::{
    ChatCompletionMessage, Context, ContextWindowBudget, Model, ModelId, ProviderResponse,
    ResultStream,
};
use forge_app::{EnvironmentInfra, HttpInfra};
use forge_domain::{ChatRepository, ModelSource, Provider, ProviderId};
use forge_infra::CacacheStorage;
use tokio::task::AbortHandle;
use url::Url;

use crate::provider::anthropic::AnthropicResponseRepository;
use crate::provider::bedrock::BedrockResponseRepository;
use crate::provider::google::GoogleResponseRepository;
use crate::provider::openai::OpenAIResponseRepository;
use crate::provider::openai_responses::OpenAIResponsesResponseRepository;
use crate::provider::opencode::OpenCodeZenResponseRepository;

/// Repository responsible for routing chat requests to the appropriate provider
/// implementation based on the provider's response type.
pub struct ForgeChatRepository<F> {
    router: Arc<ProviderRouter<F>>,
    model_cache: Arc<CacacheStorage>,
    bg_refresh: BgRefresh,
}

impl<F: EnvironmentInfra<Config = forge_config::ForgeConfig> + HttpInfra> ForgeChatRepository<F> {
    /// Creates a new ForgeChatRepository with the given infrastructure.
    ///
    /// # Arguments
    ///
    /// * `infra` - Infrastructure providing environment and HTTP capabilities
    pub fn new(infra: Arc<F>) -> Self {
        let env = infra.get_environment();
        let config = infra.get_config().unwrap_or_default();
        let model_cache_ttl_secs = config.model_cache_ttl_secs;

        let openai_repo = OpenAIResponseRepository::new(infra.clone());
        let codex_repo = OpenAIResponsesResponseRepository::new(infra.clone());
        let anthropic_repo = AnthropicResponseRepository::new(infra.clone());
        let bedrock_repo =
            BedrockResponseRepository::new(Arc::new(config.retry.unwrap_or_default()));
        let google_repo = GoogleResponseRepository::new(infra.clone());
        let opencode_zen_repo = OpenCodeZenResponseRepository::new(infra.clone());

        let model_cache = Arc::new(CacacheStorage::new(
            env.cache_dir().join("model_cache"),
            Some(std::time::Duration::from_secs(model_cache_ttl_secs)),
        ));

        Self {
            router: Arc::new(ProviderRouter {
                openai_repo,
                codex_repo,
                anthropic_repo,
                bedrock_repo,
                google_repo,
                opencode_zen_repo,
            }),
            model_cache,
            bg_refresh: BgRefresh::default(),
        }
    }
}

#[async_trait::async_trait]
impl<F: EnvironmentInfra<Config = forge_config::ForgeConfig> + HttpInfra + Sync> ChatRepository
    for ForgeChatRepository<F>
{
    async fn chat(
        &self,
        model_id: &ModelId,
        context: Context,
        provider: Provider<Url>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
        self.router.chat(model_id, context, provider).await
    }

    async fn models(&self, provider: Provider<Url>) -> anyhow::Result<Vec<Model>> {
        use forge_app::KVStore;

        let cache_key = format!("models:{}", provider.id);

        if let Ok(Some(cached)) = self
            .model_cache
            .cache_get::<_, Vec<Model>>(&cache_key)
            .await
        {
            tracing::debug!(provider_id = %provider.id, "returning cached models; refreshing in background");

            // Spawn a background task to refresh the disk cache. The abort
            // handle is stored so the task is cancelled if the service is dropped.
            let cache = self.model_cache.clone();
            let router = self.router.clone();
            let key = cache_key;
            let handle = tokio::spawn(async move {
                match router.models(provider).await {
                    Ok(models) => {
                        if let Err(err) = cache.cache_set(&key, &models).await {
                            tracing::warn!(error = %err, "background refresh: failed to cache model list");
                        }
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "background refresh: failed to fetch models");
                    }
                }
            });
            self.bg_refresh.register(handle.abort_handle());

            return Ok(cached);
        }

        let models = self.router.models(provider).await?;

        if let Err(err) = self.model_cache.cache_set(&cache_key, &models).await {
            tracing::warn!(error = %err, "failed to cache model list");
        }

        Ok(models)
    }
}

/// Routes chat and model requests to the correct provider backend.
struct ProviderRouter<F> {
    openai_repo: OpenAIResponseRepository<F>,
    codex_repo: OpenAIResponsesResponseRepository<F>,
    anthropic_repo: AnthropicResponseRepository<F>,
    bedrock_repo: BedrockResponseRepository,
    google_repo: GoogleResponseRepository<F>,
    opencode_zen_repo: OpenCodeZenResponseRepository<F>,
}

impl<F: HttpInfra + EnvironmentInfra<Config = forge_config::ForgeConfig> + Sync> ProviderRouter<F> {
    async fn chat(
        &self,
        model_id: &ModelId,
        context: Context,
        provider: Provider<Url>,
    ) -> ResultStream<ChatCompletionMessage, anyhow::Error> {
        let context = self
            .validate_context_window_before_dispatch(model_id, context, &provider)
            .await?;

        match provider.response {
            Some(ProviderResponse::OpenAI) => {
                // Check if model is a Codex model
                if model_id.as_str().contains("gpt-5")
                    && (provider.id == ProviderId::OPENAI
                        || provider.id == ProviderId::GITHUB_COPILOT
                        || provider.id == ProviderId::CODEX)
                {
                    self.codex_repo.chat(model_id, context, provider).await
                } else if provider.id == ProviderId::CODEX {
                    // All Codex provider models use the Responses API
                    self.codex_repo.chat(model_id, context, provider).await
                } else {
                    self.openai_repo.chat(model_id, context, provider).await
                }
            }
            Some(ProviderResponse::OpenAIResponses) => {
                self.codex_repo.chat(model_id, context, provider).await
            }
            Some(ProviderResponse::Anthropic) => {
                self.anthropic_repo.chat(model_id, context, provider).await
            }
            Some(ProviderResponse::Bedrock) => {
                self.bedrock_repo.chat(model_id, context, provider).await
            }
            Some(ProviderResponse::Google) => {
                self.google_repo.chat(model_id, context, provider).await
            }
            Some(ProviderResponse::OpenCode) => {
                self.opencode_zen_repo
                    .chat(model_id, context, provider)
                    .await
            }
            None => Err(anyhow::anyhow!(
                "Provider response type not configured for provider: {}",
                provider.id
            )),
        }
    }

    async fn validate_context_window_before_dispatch(
        &self,
        model_id: &ModelId,
        context: Context,
        provider: &Provider<Url>,
    ) -> anyhow::Result<Context> {
        let context_window = match context.model_context_length {
            Some(context_window) => context_window,
            None => self
                .resolve_model_context_length(model_id, provider)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Provider dispatch context-window guard cannot prove safety for model '{}' on provider '{}' because context_length metadata is missing. Add context_length to the model metadata or select a model with known context window.",
                        model_id,
                        provider.id
                    )
                })?,
        };

        let context_window = usize::try_from(context_window).map_err(|_| {
            anyhow::anyhow!(
                "Provider dispatch context-window guard cannot represent context_length {} for model '{}' on provider '{}'.",
                context_window,
                model_id,
                provider.id
            )
        })?;
        let output_reservation = context
            .max_tokens
            .unwrap_or(ContextWindowBudget::DEFAULT_OUTPUT_TOKEN_RESERVATION);
        let context_budget = ContextWindowBudget::new(context_window, output_reservation);
        let input_budget = context_budget.effective_input_budget().ok_or_else(|| {
            anyhow::anyhow!(
                "Provider dispatch context-window guard blocked request for model '{}' on provider '{}'. Context window is {} tokens, reserved output is {} tokens, and safety margin is {} tokens, leaving no safe prompt budget. Lower max_tokens or select a larger-context model.",
                model_id,
                provider.id,
                context_budget.context_window(),
                context_budget.output_reservation(),
                context_budget.safety_margin()
            )
        })?;
        let estimated_input = Self::estimated_context_input_tokens(&context);

        if estimated_input > input_budget {
            anyhow::bail!(
                "Provider dispatch context-window guard blocked an oversized request before provider backend dispatch. Model '{}' on provider '{}' has context window {} tokens; reserved output is {} tokens; safety margin is {} tokens; effective input budget is {} tokens; conservative request estimate is {} tokens. Reduce context, lower max_tokens, or select a larger-context model.",
                model_id,
                provider.id,
                context_budget.context_window(),
                context_budget.output_reservation(),
                context_budget.safety_margin(),
                input_budget,
                estimated_input
            );
        }

        Ok(context.model_context_length(u64::try_from(context_window).map_err(|_| {
            anyhow::anyhow!(
                "Provider dispatch context-window guard cannot reattach context_length {} for model '{}' on provider '{}'.",
                context_window,
                model_id,
                provider.id
            )
        })?))
    }

    async fn resolve_model_context_length(
        &self,
        model_id: &ModelId,
        provider: &Provider<Url>,
    ) -> anyhow::Result<Option<u64>> {
        if let Some(ModelSource::Hardcoded(models)) = provider.models.as_ref() {
            return Ok(models
                .iter()
                .find(|model| &model.id == model_id && model.provider_id == provider.id)
                .and_then(|model| model.context_length));
        }

        Ok(self
            .models(provider.clone())
            .await?
            .into_iter()
            .find(|model| &model.id == model_id && model.provider_id == provider.id)
            .and_then(|model| model.context_length))
    }

    fn estimated_context_input_tokens(context: &Context) -> usize {
        serde_json::to_vec(context)
            .map(|payload| payload.len())
            .unwrap_or_else(|_| context.token_count_approx())
    }

    async fn models(&self, provider: Provider<Url>) -> anyhow::Result<Vec<Model>> {
        match provider.response {
            Some(ProviderResponse::OpenAI) => self.openai_repo.models(provider).await,
            Some(ProviderResponse::OpenAIResponses) => self.codex_repo.models(provider).await,
            Some(ProviderResponse::Anthropic) => self.anthropic_repo.models(provider).await,
            Some(ProviderResponse::Bedrock) => self.bedrock_repo.models(provider).await,
            Some(ProviderResponse::Google) => self.google_repo.models(provider).await,
            Some(ProviderResponse::OpenCode) => self.opencode_zen_repo.models(provider).await,
            None => Err(anyhow::anyhow!(
                "Provider response type not configured for provider: {}",
                provider.id
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use forge_app::domain::{
        AuthMethod, ContextMessage, Environment, MessageEntry, ModelSource, Role, TextMessage,
    };
    use forge_eventsource::EventSource;
    use pretty_assertions::assert_eq;
    use reqwest::Response;
    use reqwest::header::HeaderMap;

    use super::*;

    #[derive(Default)]
    struct MockInfra {
        http_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl HttpInfra for MockInfra {
        async fn http_get(
            &self,
            _url: &Url,
            _headers: Option<HeaderMap>,
        ) -> anyhow::Result<Response> {
            self.http_calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("provider backend must not be reached")
        }

        async fn http_post(
            &self,
            _url: &Url,
            _headers: Option<HeaderMap>,
            _body: Bytes,
        ) -> anyhow::Result<Response> {
            self.http_calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("provider backend must not be reached")
        }

        async fn http_delete(&self, _url: &Url) -> anyhow::Result<Response> {
            self.http_calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("provider backend must not be reached")
        }

        async fn http_eventsource(
            &self,
            _url: &Url,
            _headers: Option<HeaderMap>,
            _body: Bytes,
        ) -> anyhow::Result<EventSource> {
            self.http_calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("provider backend must not be reached")
        }
    }

    impl EnvironmentInfra for MockInfra {
        type Config = forge_config::ForgeConfig;

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }

        fn get_environment(&self) -> Environment {
            fake::Fake::fake(&fake::Faker)
        }

        fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(
            &self,
            _ops: Vec<forge_app::domain::ConfigOperation>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn router_fixture(infra: Arc<MockInfra>) -> ProviderRouter<MockInfra> {
        ProviderRouter {
            openai_repo: OpenAIResponseRepository::new(infra.clone()),
            codex_repo: OpenAIResponsesResponseRepository::new(infra.clone()),
            anthropic_repo: AnthropicResponseRepository::new(infra.clone()),
            bedrock_repo: BedrockResponseRepository::new(Arc::new(
                forge_config::RetryConfig::default(),
            )),
            google_repo: GoogleResponseRepository::new(infra.clone()),
            opencode_zen_repo: OpenCodeZenResponseRepository::new(infra),
        }
    }

    fn provider_fixture(response: ProviderResponse, context_length: Option<u64>) -> Provider<Url> {
        let provider_id = ProviderId::from("test_provider".to_string());
        Provider {
            id: provider_id.clone(),
            provider_type: Default::default(),
            response: Some(response),
            url: Url::parse("https://example.com/v1/chat/completions").unwrap(),
            models: Some(ModelSource::Hardcoded(vec![Model {
                id: ModelId::new("test-model"),
                provider_id: provider_id.clone(),
                name: None,
                description: None,
                context_length,
                tools_supported: None,
                supports_parallel_tool_calls: None,
                supports_reasoning: None,
                input_modalities: vec![],
            }])),
            auth_methods: vec![AuthMethod::ApiKey],
            url_params: vec![],
            credential: None,
            custom_headers: None,
        }
    }

    fn context_fixture(content: String) -> Context {
        Context::default().messages(vec![MessageEntry::from(ContextMessage::Text(
            TextMessage::new(Role::User, content),
        ))])
    }

    #[tokio::test]
    async fn test_provider_dispatch_unknown_context_window_blocks_all_backends_before_http() {
        let responses = vec![
            ProviderResponse::OpenAI,
            ProviderResponse::OpenAIResponses,
            ProviderResponse::Anthropic,
            ProviderResponse::Bedrock,
            ProviderResponse::Google,
            ProviderResponse::OpenCode,
        ];

        for response in responses {
            let infra = Arc::new(MockInfra::default());
            let fixture = router_fixture(infra.clone());
            let provider = provider_fixture(response, None);

            let actual = match fixture
                .chat(
                    &ModelId::new("test-model"),
                    context_fixture("short".to_string()),
                    provider,
                )
                .await
            {
                Ok(_) => panic!("provider dispatch should fail before backend"),
                Err(error) => error.to_string(),
            };
            let expected = true;

            assert_eq!(
                actual.contains("context_length metadata is missing"),
                expected
            );
            assert_eq!(infra.http_calls.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn test_provider_dispatch_oversized_context_blocks_before_http() {
        let infra = Arc::new(MockInfra::default());
        let fixture = router_fixture(infra.clone());
        let provider = provider_fixture(ProviderResponse::Anthropic, Some(12_000));

        let actual = match fixture
            .chat(
                &ModelId::new("test-model"),
                context_fixture("x ".repeat(9_000)),
                provider,
            )
            .await
        {
            Ok(_) => panic!("provider dispatch should fail before backend"),
            Err(error) => error.to_string(),
        };
        let expected = true;

        assert_eq!(actual.contains("oversized request"), expected);
        assert_eq!(infra.http_calls.load(Ordering::SeqCst), 0);
    }
}

#[derive(Default)]
struct BgRefresh(std::sync::Mutex<Vec<AbortHandle>>);

impl BgRefresh {
    /// Registers an abort handle to be cancelled when this guard is dropped.
    fn register(&self, handle: AbortHandle) {
        if let Ok(mut handles) = self.0.lock() {
            handles.push(handle);
        }
    }
}

impl Drop for BgRefresh {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.0.lock() {
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
    }
}
