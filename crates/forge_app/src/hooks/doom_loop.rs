use async_trait::async_trait;
use forge_domain::{
    ContextMessage, Conversation, EventData, EventHandle, RequestPayload, Role, TextMessage,
};
use forge_template::Element;
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::TemplateEngine;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
struct DoomLoopThreshold(usize);

impl DoomLoopThreshold {
    const MINIMUM: usize = 2;

    fn new(value: usize) -> Self {
        Self(value.max(Self::MINIMUM))
    }

    fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ToolActionSignature {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct DoomLoopDetection {
    repetition_count: usize,
    tool_call_count: usize,
    pattern_start_idx: usize,
    pattern_length: usize,
    pattern: Vec<ToolActionSignature>,
}

impl DoomLoopDetection {
    fn suppression_key(&self) -> String {
        let mut hasher = Sha256::new();
        Self::update_hash(&mut hasher, &self.tool_call_count.to_string());
        Self::update_hash(&mut hasher, &self.pattern_start_idx.to_string());
        Self::update_hash(&mut hasher, &self.pattern_length.to_string());
        Self::update_hash(&mut hasher, &self.repetition_count.to_string());

        for signature in &self.pattern {
            Self::update_hash(&mut hasher, &signature.name);
            Self::update_hash(&mut hasher, &signature.arguments);
        }

        hex::encode(hasher.finalize())
    }

    fn update_hash(hasher: &mut Sha256, value: &str) {
        hasher.update(value.len().to_string().as_bytes());
        hasher.update(b":");
        hasher.update(value.as_bytes());
        hasher.update(b";");
    }
}

/// Detector for repeated tool-call patterns that indicate an agent is stuck.
#[derive(Debug, Clone)]
pub struct DoomLoopDetector {
    threshold: DoomLoopThreshold,
}

impl Default for DoomLoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DoomLoopDetector {
    const DEFAULT_THRESHOLD: usize = 3;

    /// Creates a detector with the production threshold.
    pub fn new() -> Self {
        Self { threshold: DoomLoopThreshold::new(Self::DEFAULT_THRESHOLD) }
    }

    #[cfg(test)]
    fn threshold(mut self, threshold: usize) -> Self {
        self.threshold = DoomLoopThreshold::new(threshold);
        self
    }

    #[cfg(test)]
    fn detect_from_conversation(&self, conversation: &Conversation) -> Option<usize> {
        self.detect_loop(conversation)
            .map(|detection| detection.repetition_count)
    }

    fn detect_loop(&self, conversation: &Conversation) -> Option<DoomLoopDetection> {
        let all_signatures = self.extract_tool_signatures(conversation);
        let (pattern_start_idx, repetition_count) =
            self.check_repeating_pattern(&all_signatures)?;
        let pattern_length = all_signatures
            .len()
            .checked_sub(pattern_start_idx)?
            .checked_div(repetition_count)?;
        let pattern_end_idx = pattern_start_idx.checked_add(pattern_length)?;
        let pattern = all_signatures
            .get(pattern_start_idx..pattern_end_idx)?
            .to_vec();

        Some(DoomLoopDetection {
            repetition_count,
            tool_call_count: all_signatures.len(),
            pattern_start_idx,
            pattern_length,
            pattern,
        })
    }

    fn extract_tool_signatures(&self, conversation: &Conversation) -> Vec<ToolActionSignature> {
        let assistant_messages = conversation
            .context
            .as_ref()
            .map(|ctx| {
                Self::extract_assistant_messages(ctx.messages.iter().map(|entry| &entry.message))
            })
            .unwrap_or_default();

        assistant_messages
            .iter()
            .filter_map(|msg| msg.tool_calls.as_ref())
            .flat_map(|calls| calls.iter())
            .map(|call| ToolActionSignature {
                name: call.name.as_str().to_string(),
                arguments: call.arguments.clone().into_string(),
            })
            .collect()
    }

    fn check_repeating_pattern<T>(&self, sequence: &[T]) -> Option<(usize, usize)>
    where
        T: Eq,
    {
        if sequence.is_empty() || sequence.len() < self.threshold.get() {
            return None;
        }

        for pattern_length in 1..sequence.len() {
            let complete_repetitions =
                self.count_recent_pattern_repetitions(sequence, pattern_length);

            if complete_repetitions >= self.threshold.get() {
                let pattern_offset = complete_repetitions.checked_mul(pattern_length)?;
                let pattern_start_idx = sequence.len().checked_sub(pattern_offset)?;

                if sequence.get(pattern_start_idx).is_some() {
                    return Some((pattern_start_idx, complete_repetitions));
                }
            }
        }

        None
    }

    fn count_recent_pattern_repetitions<T>(&self, sequence: &[T], pattern_length: usize) -> usize
    where
        T: Eq,
    {
        if pattern_length == 0 || sequence.len() < pattern_length {
            return 0;
        }

        let total_len = sequence.len();
        let mut repetitions = 0;
        let pattern_start = total_len - pattern_length;
        let Some(pattern) = sequence.get(pattern_start..total_len) else {
            return repetitions;
        };
        repetitions += 1;

        let mut pos = pattern_start;
        while pos >= pattern_length {
            pos -= pattern_length;
            let Some(chunk) = sequence.get(pos..pos + pattern_length) else {
                break;
            };

            if chunk == pattern {
                repetitions += 1;
            } else {
                break;
            }
        }

        repetitions
    }

    /// Extracts assistant messages from conversation context messages.
    pub fn extract_assistant_messages<'a>(
        messages: impl Iterator<Item = &'a ContextMessage> + 'a,
    ) -> Vec<&'a TextMessage> {
        messages
            .filter_map(|msg| {
                if let ContextMessage::Text(text_msg) = msg
                    && text_msg.role == Role::Assistant
                {
                    return Some(text_msg);
                }
                None
            })
            .collect()
    }

    fn has_reminder_for_detection(
        messages: &[forge_domain::MessageEntry],
        detection: &DoomLoopDetection,
    ) -> bool {
        let marker = format!("doom_loop_id=\"{}\"", detection.suppression_key());
        messages.iter().any(|entry| {
            entry.message.has_role(Role::System)
                && entry
                    .message
                    .content()
                    .is_some_and(|content| content.contains(&marker))
        })
    }

    fn render_reminder(&self, detection: &DoomLoopDetection) -> anyhow::Result<String> {
        let reminder = TemplateEngine::default().render(
            "forge-doom-loop-reminder.md",
            &serde_json::json!({"consecutive_calls": detection.repetition_count}),
        )?;
        Ok(Element::new("system_reminder")
            .attr("doom_loop_id", detection.suppression_key())
            .attr("tool_call_count", detection.tool_call_count)
            .attr("pattern_length", detection.pattern_length)
            .cdata(reminder)
            .render())
    }
}

#[async_trait]
impl EventHandle<EventData<RequestPayload>> for DoomLoopDetector {
    async fn handle(
        &self,
        event: &EventData<RequestPayload>,
        conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if let Some(detection) = self.detect_loop(conversation) {
            warn!(
                agent_id = %event.agent.id,
                request_count = event.payload.request_count,
                consecutive_calls = detection.repetition_count,
                tool_call_count = detection.tool_call_count,
                pattern_length = detection.pattern_length,
                doom_loop_id = detection.suppression_key(),
                "Doom loop detected from conversation context before next request"
            );

            if let Some(context) = conversation.context.as_mut()
                && !Self::has_reminder_for_detection(&context.messages, &detection)
            {
                let content = self.render_reminder(&detection)?;
                context
                    .messages
                    .push(ContextMessage::system(content).into());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod detector_basics;
#[cfg(test)]
mod detector_patterns;
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod handler;
#[cfg(test)]
mod pattern;
#[cfg(test)]
mod regression_sequences;
