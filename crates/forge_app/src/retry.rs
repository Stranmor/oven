use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use forge_config::RetryConfig;
use forge_domain::Error;

use crate::dto::openai::{Error as OpenAiError, ErrorCode, ErrorResponse};

pub async fn retry_with_config<F, Fut, T, C>(
    config: &RetryConfig,
    operation: F,
    notify: Option<C>,
) -> anyhow::Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
    C: Fn(&anyhow::Error, Duration) + Send + Sync + 'static,
{
    let strategy = ExponentialBuilder::default()
        .with_min_delay(Duration::from_millis(config.min_delay_ms))
        .with_factor(config.backoff_factor as f32)
        .with_max_times(config.max_attempts)
        .with_jitter();

    let retryable = operation.retry(&strategy).when(should_retry);

    match notify {
        Some(callback) => retryable.notify(callback).await,
        None => retryable.await,
    }
}

/// Determines if an error should trigger a retry attempt.
///
/// This function checks if the error is a retryable domain error.
/// Currently, only `Error::Retryable` errors will trigger retries.
pub(crate) fn is_provider_context_window_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<OpenAiError>()
            .is_some_and(|error| match error {
                OpenAiError::Response(response) => {
                    error_response_has_context_window_signal(response)
                }
                OpenAiError::InvalidStatusCode(_) => false,
            })
    })
}

fn should_retry(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<Error>()
        .is_some_and(|error| match error {
            Error::Retryable(source) => !is_provider_context_window_error(source),
            _ => false,
        })
}

fn error_response_has_context_window_signal(error: &ErrorResponse) -> bool {
    let code_matches = error
        .code
        .as_ref()
        .and_then(ErrorCode::as_str)
        .is_some_and(text_has_context_window_signal);
    let message_matches = error
        .message
        .as_deref()
        .is_some_and(text_has_context_window_signal);
    let nested_matches = error
        .error
        .as_deref()
        .is_some_and(error_response_has_context_window_signal);

    code_matches || message_matches || nested_matches
}

fn text_has_context_window_signal(text: &str) -> bool {
    let normalized = text.to_lowercase();
    normalized.contains("context_length_exceeded")
        || normalized.contains("maximum context length")
        || normalized.contains("context window")
        || normalized.contains("context length") && normalized.contains("exceed")
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_should_retry_rejects_retryable_provider_context_window_error() {
        let fixture = anyhow!(Error::Retryable(anyhow!(OpenAiError::Response(
            ErrorResponse::default()
                .code(ErrorCode::String("context_length_exceeded".to_string()))
                .message("This model's maximum context length was exceeded".to_string()),
        ))));

        let actual = should_retry(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_should_retry_allows_non_context_retryable_error() {
        let fixture = anyhow!(Error::Retryable(anyhow!("transient hallucination")));

        let actual = should_retry(&fixture);
        let expected = true;

        assert_eq!(actual, expected);
    }
}
