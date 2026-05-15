use serde::{Deserialize, Serialize};

/// Provider-level request size metrics used for compaction reporting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRequestEstimate {
    /// Estimated input tokens for the serialized provider request.
    pub estimated_tokens: usize,
    /// Effective provider input budget after output reservation and safety margin.
    pub input_budget: Option<usize>,
    /// Tokens/bytes by which the estimate exceeds the effective input budget.
    pub excess_tokens: Option<usize>,
}

impl ProviderRequestEstimate {
    /// Creates a provider request estimate with the matching budget state.
    ///
    /// # Arguments
    /// * `estimated_tokens` - Estimated input tokens for the provider request.
    /// * `input_budget` - Effective input budget available for provider input.
    pub fn new(estimated_tokens: usize, input_budget: Option<usize>) -> Self {
        Self {
            estimated_tokens,
            input_budget,
            excess_tokens: input_budget.map(|budget| estimated_tokens.saturating_sub(budget)),
        }
    }
}

/// Direction and magnitude of provider request estimate change after compaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderRequestEstimateChange {
    /// Provider estimate did not change.
    Unchanged,
    /// Provider estimate shrank by the given percentage.
    Reduction { percentage: f64 },
    /// Provider estimate grew by the given percentage.
    Growth { percentage: f64 },
    /// Provider estimate grew from a zero-token baseline.
    GrowthFromZero,
}

impl ProviderRequestEstimateChange {
    /// Returns the signed percentage used by legacy reduction callers.
    pub fn signed_reduction_percentage(&self) -> f64 {
        match self {
            Self::Unchanged => 0.0,
            Self::Reduction { percentage } => *percentage,
            Self::Growth { percentage } => -*percentage,
            Self::GrowthFromZero => -100.0,
        }
    }
}

/// Contains metrics related to context compaction
/// This struct provides information about the compaction operation
/// such as the original and compacted token counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Number of tokens in the original context
    pub original_tokens: usize,
    /// Number of tokens after compaction
    pub compacted_tokens: usize,
    /// Number of messages in the original context
    pub original_messages: usize,
    /// Number of messages after compaction
    pub compacted_messages: usize,
    /// Provider request estimate before compaction when active provider/model data is available.
    pub original_provider_request: Option<ProviderRequestEstimate>,
    /// Provider request estimate after compaction when active provider/model data is available.
    pub compacted_provider_request: Option<ProviderRequestEstimate>,
}

impl CompactionResult {
    /// Create a new CompactionResult with the specified metrics
    pub fn new(
        original_tokens: usize,
        compacted_tokens: usize,
        original_messages: usize,
        compacted_messages: usize,
    ) -> Self {
        Self {
            original_tokens,
            compacted_tokens,
            original_messages,
            compacted_messages,
            original_provider_request: None,
            compacted_provider_request: None,
        }
    }

    /// Attaches active-provider request estimates to the compaction result.
    ///
    /// # Arguments
    /// * `original_provider_request` - Provider estimate before compaction.
    /// * `compacted_provider_request` - Provider estimate after compaction.
    pub fn provider_request_estimates(
        mut self,
        original_provider_request: ProviderRequestEstimate,
        compacted_provider_request: ProviderRequestEstimate,
    ) -> Self {
        self.original_provider_request = Some(original_provider_request);
        self.compacted_provider_request = Some(compacted_provider_request);
        self
    }

    /// Calculate the provider request estimate change after compaction.
    pub fn provider_request_estimate_change(&self) -> Option<ProviderRequestEstimateChange> {
        let original = self.original_provider_request.as_ref()?.estimated_tokens;
        let compacted = self.compacted_provider_request.as_ref()?.estimated_tokens;
        match compacted.cmp(&original) {
            std::cmp::Ordering::Equal => Some(ProviderRequestEstimateChange::Unchanged),
            std::cmp::Ordering::Less if original == 0 => {
                Some(ProviderRequestEstimateChange::Unchanged)
            }
            std::cmp::Ordering::Less => Some(ProviderRequestEstimateChange::Reduction {
                percentage: ((original - compacted) as f64 / original as f64) * 100.0,
            }),
            std::cmp::Ordering::Greater if original == 0 => {
                Some(ProviderRequestEstimateChange::GrowthFromZero)
            }
            std::cmp::Ordering::Greater => Some(ProviderRequestEstimateChange::Growth {
                percentage: ((compacted - original) as f64 / original as f64) * 100.0,
            }),
        }
    }

    /// Calculate the signed percentage reduction in provider request estimate.
    pub fn provider_request_reduction_percentage(&self) -> Option<f64> {
        Some(
            self.provider_request_estimate_change()?
                .signed_reduction_percentage(),
        )
    }

    /// Returns whether the compacted provider request estimate fits its input budget.
    pub fn compacted_provider_request_fits(&self) -> Option<bool> {
        let estimate = self.compacted_provider_request.as_ref()?;
        estimate
            .input_budget
            .map(|budget| estimate.estimated_tokens <= budget)
    }

    /// Calculate the percentage reduction in tokens
    pub fn token_reduction_percentage(&self) -> f64 {
        if self.original_tokens == 0 || self.compacted_tokens == 0 {
            return 0.0;
        }
        ((self.original_tokens.saturating_sub(self.compacted_tokens)) as f64
            / self.original_tokens as f64)
            * 100.0
    }

    /// Calculate the percentage reduction in messages
    pub fn message_reduction_percentage(&self) -> f64 {
        if self.original_messages == 0 || self.compacted_messages == 0 {
            return 0.0;
        }
        ((self
            .original_messages
            .saturating_sub(self.compacted_messages)) as f64
            / self.original_messages as f64)
            * 100.0
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_token_reduction_percentage() {
        let result = CompactionResult::new(1000, 500, 20, 10);
        assert_eq!(result.token_reduction_percentage(), 50.0);

        // Edge case: no original tokens
        let result = CompactionResult::new(0, 0, 20, 10);
        assert_eq!(result.token_reduction_percentage(), 0.0);

        // Edge case: no compacted tokens
        let result = CompactionResult::new(1000, 0, 20, 0);
        assert_eq!(result.token_reduction_percentage(), 0.0);
    }

    #[test]
    fn test_message_reduction_percentage() {
        let result = CompactionResult::new(1000, 500, 20, 10);
        assert_eq!(result.message_reduction_percentage(), 50.0);

        // Edge case: no original messages
        let result = CompactionResult::new(1000, 500, 0, 0);
        assert_eq!(result.message_reduction_percentage(), 0.0);

        // Edge case: no compacted messages
        let result = CompactionResult::new(1000, 0, 20, 0);
        assert_eq!(result.message_reduction_percentage(), 0.0);
    }

    #[test]
    fn test_provider_request_reduction_percentage_and_fit_status() {
        let fixture = CompactionResult::new(1000, 500, 20, 10).provider_request_estimates(
            ProviderRequestEstimate::new(2000, Some(1800)),
            ProviderRequestEstimate::new(1200, Some(1800)),
        );
        let actual = (
            fixture.provider_request_reduction_percentage(),
            fixture.compacted_provider_request_fits(),
        );
        let expected = (Some(40.0), Some(true));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_request_reduction_percentage_reports_growth_as_negative() {
        let fixture = CompactionResult::new(1000, 1100, 20, 22).provider_request_estimates(
            ProviderRequestEstimate::new(1200, Some(1800)),
            ProviderRequestEstimate::new(1500, Some(1800)),
        );
        let actual = (
            fixture.provider_request_estimate_change(),
            fixture.provider_request_reduction_percentage(),
        );
        let expected = (
            Some(ProviderRequestEstimateChange::Growth { percentage: 25.0 }),
            Some(-25.0),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_request_estimate_change_reports_growth_from_zero() {
        let fixture = CompactionResult::new(0, 100, 0, 1).provider_request_estimates(
            ProviderRequestEstimate::new(0, Some(1800)),
            ProviderRequestEstimate::new(1500, Some(1800)),
        );
        let actual = (
            fixture.provider_request_estimate_change(),
            fixture.provider_request_reduction_percentage(),
        );
        let expected = (
            Some(ProviderRequestEstimateChange::GrowthFromZero),
            Some(-100.0),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_provider_request_excess_reports_over_budget_amount() {
        let result = CompactionResult::new(1000, 500, 20, 10).provider_request_estimates(
            ProviderRequestEstimate::new(2000, Some(1800)),
            ProviderRequestEstimate::new(1900, Some(1800)),
        );
        let actual = result
            .compacted_provider_request
            .as_ref()
            .and_then(|estimate| estimate.excess_tokens);
        let expected = Some(100);

        assert_eq!(actual, expected);
        assert_eq!(result.compacted_provider_request_fits(), Some(false));
    }
}
