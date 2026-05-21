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

/// Determines whether a provider error chain represents an exhausted context
/// window condition.
///
/// The detector accepts typed OpenAI context errors and typed HTTP status
/// errors whose attached provider JSON body strictly reports
/// `context_length_exceeded`. Generic 400 errors and quota-related messages are
/// rejected so callers can retry normal transient failures without looping on
/// unrecoverable oversized prompts.
pub fn is_provider_context_window_error(error: &anyhow::Error) -> bool {
    let has_typed_context_signal = error.chain().any(|cause| {
        cause
            .downcast_ref::<OpenAiError>()
            .is_some_and(|error| match error {
                OpenAiError::Response(response) => {
                    error_response_has_strict_context_length_exceeded(response)
                }
                OpenAiError::InvalidStatusCode(_) => false,
            })
    });
    if has_typed_context_signal {
        return true;
    }

    error.chain().any(|cause| {
        cause
            .downcast_ref::<OpenAiError>()
            .is_some_and(|error| matches!(error, OpenAiError::InvalidStatusCode(400)))
    }) && error_chain_has_context_window_error_body(error)
}

fn should_retry(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<Error>()
        .is_some_and(|error| match error {
            Error::Retryable(source) => !is_provider_context_window_error(source),
            _ => false,
        })
}

fn error_chain_has_context_window_error_body(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let text = cause.to_string();
        parse_error_response_from_text(&text)
            .is_some_and(|response| error_response_has_strict_context_length_exceeded(&response))
    })
}

fn parse_error_response_from_text(text: &str) -> Option<ErrorResponse> {
    serde_json::from_str::<ErrorResponse>(text)
        .ok()
        .or_else(|| json_object_slice(text).and_then(|json| serde_json::from_str(json).ok()))
}

fn json_object_slice(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?.checked_add(1)?;
    if start < end {
        text.get(start..end)
    } else {
        None
    }
}

fn error_response_has_strict_context_length_exceeded(error: &ErrorResponse) -> bool {
    let code_matches = error
        .code
        .as_ref()
        .and_then(ErrorCode::as_str)
        .is_some_and(|text| text.eq_ignore_ascii_case("context_length_exceeded"));
    let message_matches = error
        .message
        .as_deref()
        .is_some_and(text_has_context_length_exceeded_signal);
    let nested_matches = error
        .error
        .as_deref()
        .is_some_and(error_response_has_strict_context_length_exceeded);

    code_matches || message_matches || nested_matches
}

fn text_has_context_length_exceeded_signal(text: &str) -> bool {
    let normalized = text.to_lowercase();
    if text_has_quota_plan_or_tier_signal(&normalized) {
        return false;
    }

    normalized.contains("context_length_exceeded")
        || normalized.contains("maximum context length") && normalized.contains("exceed")
        || normalized.contains("context length") && normalized.contains("exceed")
}

fn text_has_quota_plan_or_tier_signal(normalized: &str) -> bool {
    normalized.contains("quota")
        || normalized.contains("plan")
        || normalized.contains("tier")
        || normalized.contains("rate limit")
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_provider_context_window_error_detects_invalid_status_code_with_openai_body() {
        let body = serde_json::json!({
            "error": {
                "message": "This model's maximum context length was exceeded.",
                "code": "context_length_exceeded"
            }
        })
        .to_string();
        let fixture = anyhow::Error::from(OpenAiError::InvalidStatusCode(400)).context(body);

        let actual = is_provider_context_window_error(&fixture);
        let expected = true;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_rejects_generic_invalid_status_code() {
        let body = serde_json::json!({
            "error": {
                "message": "Generic invalid request",
                "code": 400
            }
        })
        .to_string();
        let fixture = anyhow::Error::from(OpenAiError::InvalidStatusCode(400)).context(body);

        let actual = is_provider_context_window_error(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_rejects_non_json_invalid_status_context() {
        let fixture = anyhow::Error::from(OpenAiError::InvalidStatusCode(400))
            .context("requested context window tier quota exceeded for current plan");

        let actual = is_provider_context_window_error(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_rejects_context_window_quota_json_body() {
        let body = serde_json::json!({
            "error": {
                "message": "requested context window tier quota exceeded for current plan",
                "code": "quota_exceeded"
            }
        })
        .to_string();
        let fixture = anyhow::Error::from(OpenAiError::InvalidStatusCode(400)).context(body);

        let actual = is_provider_context_window_error(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_rejects_typed_quota_response_with_context_window_text() {
        let fixture = anyhow!(OpenAiError::Response(
            ErrorResponse::default()
                .code(ErrorCode::String("quota_exceeded".to_string()))
                .message(
                    "requested context window tier quota exceeded for current plan".to_string()
                ),
        ));

        let actual = is_provider_context_window_error(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_rejects_typed_quota_response_with_context_length_text() {
        let fixture = anyhow!(OpenAiError::Response(
            ErrorResponse::default()
                .code(ErrorCode::String("quota_exceeded".to_string()))
                .message(
                    "requested maximum context length tier quota exceeded for current plan"
                        .to_string()
                ),
        ));

        let actual = is_provider_context_window_error(&fixture);
        let expected = false;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_context_window_error_detects_typed_context_length_code_with_quota_like_message()
     {
        let fixture = anyhow!(OpenAiError::Response(
            ErrorResponse::default()
                .code(ErrorCode::String("context_length_exceeded".to_string()))
                .message(
                    "requested maximum context length tier quota exceeded for current plan"
                        .to_string()
                ),
        ));

        let actual = is_provider_context_window_error(&fixture);
        let expected = true;

        assert_eq!(actual, expected);
    }

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
