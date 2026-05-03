use async_trait::async_trait;
use forge_domain::{
    ContextMessage, Conversation, EventData, EventHandle, RequestPayload, Role, ToolResult,
    ToolValue,
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

#[derive(Debug, Clone)]
struct ToolActionSignature {
    name: String,
    arguments: String,
    intent: ToolIntent,
}

impl PartialEq for ToolActionSignature {
    fn eq(&self, other: &Self) -> bool {
        match (&self.intent.family, &other.intent.family) {
            (ToolFamily::Read, ToolFamily::Read) | (ToolFamily::Search, ToolFamily::Search) => {
                self.intent == other.intent
            }
            _ => self.name == other.name && self.arguments == other.arguments,
        }
    }
}

impl Eq for ToolActionSignature {}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ToolIntent {
    family: ToolFamily,
    target: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum ToolFamily {
    Read,
    Search,
    Shell,
    ProcessRead,
    Other(String),
}

impl ToolFamily {
    fn as_str(&self) -> &str {
        match self {
            ToolFamily::Read => "read",
            ToolFamily::Search => "search",
            ToolFamily::Shell => "shell",
            ToolFamily::ProcessRead => "process_read",
            ToolFamily::Other(name) => name,
        }
    }
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
    fn pattern_family(&self) -> &str {
        self.pattern
            .first()
            .map(|signature| signature.intent.family.as_str())
            .unwrap_or("unknown")
    }

    fn pattern_intent(&self) -> &str {
        self.pattern
            .first()
            .map(|signature| signature.intent.target.as_str())
            .unwrap_or("unknown")
    }

    fn suppression_key(&self) -> String {
        let mut hasher = Sha256::new();
        Self::update_hash(&mut hasher, &self.tool_call_count.to_string());
        Self::update_hash(&mut hasher, &self.pattern_start_idx.to_string());
        Self::update_hash(&mut hasher, &self.pattern_length.to_string());
        Self::update_hash(&mut hasher, &self.repetition_count.to_string());

        for signature in &self.pattern {
            Self::update_hash(&mut hasher, &signature.name);
            Self::update_hash(&mut hasher, &signature.arguments);
            Self::update_hash(&mut hasher, signature.intent.family.as_str());
            Self::update_hash(&mut hasher, &signature.intent.target);
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
        let Some(context) = conversation.context.as_ref() else {
            return Vec::new();
        };

        let mut current_segment = Vec::new();
        for entry in &context.messages {
            match &entry.message {
                ContextMessage::Text(message) if message.role == Role::Assistant => {
                    if let Some(calls) = &message.tool_calls {
                        let mut message_signatures: Vec<ToolActionSignature> = calls
                            .iter()
                            .map(|call| {
                                let name = call.name.as_str().trim().to_lowercase();
                                let arguments = call.arguments.clone().into_string();
                                ToolActionSignature {
                                    intent: Self::classify_intent(&name, &arguments),
                                    name,
                                    arguments,
                                }
                            })
                            .collect();
                        message_signatures.dedup();
                        current_segment.extend(message_signatures);
                    }
                }
                ContextMessage::Tool(result) => {
                    if let Some(progress_intent) = Self::progress_intent(result) {
                        current_segment.retain(|signature| {
                            !Self::signature_matches_progress(signature, &progress_intent)
                        });
                    }
                }
                _ => {}
            }
        }
        current_segment
    }

    fn classify_intent(name: &str, arguments: &str) -> ToolIntent {
        let parsed = serde_json::from_str::<serde_json::Value>(arguments).ok();
        match name {
            "read" => ToolIntent {
                family: ToolFamily::Read,
                target: Self::json_string(&parsed, "file_path")
                    .or_else(|| Self::json_string(&parsed, "path"))
                    .map(Self::normalize_target)
                    .unwrap_or_else(|| arguments.to_string()),
            },
            "fs_search" | "sem_search" => ToolIntent {
                family: ToolFamily::Search,
                target: Self::search_target(&parsed).unwrap_or_else(|| arguments.to_string()),
            },
            "shell" => ToolIntent { family: ToolFamily::Shell, target: arguments.to_string() },
            "process_read" => ToolIntent {
                family: ToolFamily::ProcessRead,
                target: format!(
                    "process_id={};cursor={}",
                    Self::json_string(&parsed, "process_id").unwrap_or_default(),
                    Self::json_u64(&parsed, "cursor").unwrap_or_default()
                ),
            },
            other => ToolIntent {
                family: ToolFamily::Other(other.to_string()),
                target: arguments.to_string(),
            },
        }
    }

    fn json_string(parsed: &Option<serde_json::Value>, field: &str) -> Option<String> {
        parsed.as_ref()?.get(field)?.as_str().map(ToOwned::to_owned)
    }

    fn json_u64(parsed: &Option<serde_json::Value>, field: &str) -> Option<u64> {
        parsed.as_ref()?.get(field)?.as_u64()
    }

    fn normalize_target(path: String) -> String {
        path.trim().trim_start_matches("./").to_string()
    }

    fn search_target(parsed: &Option<serde_json::Value>) -> Option<String> {
        let path = Self::json_string(parsed, "path")
            .map(Self::normalize_target)
            .unwrap_or_default();
        let pattern = Self::json_string(parsed, "pattern").unwrap_or_default();
        let glob = Self::json_string(parsed, "glob").unwrap_or_default();
        let kind = Self::json_string(parsed, "type").unwrap_or_default();
        if path.is_empty() && pattern.is_empty() && glob.is_empty() && kind.is_empty() {
            return None;
        }
        Some(format!(
            "path={path};pattern={pattern};glob={glob};type={kind}"
        ))
    }

    fn output_file_path(result: &ToolResult) -> Option<String> {
        let text = result.output.as_str()?;
        let marker = "path=\"";
        let start = text.find(marker)? + marker.len();
        let end = text[start..].find('"')?;
        Some(text[start..start + end].to_string())
    }

    fn process_output_attr(result: &ToolResult, name: &str) -> Option<String> {
        let text = result.output.as_str()?;
        let marker = format!("{name}=\"");
        let start = text.find(&marker)? + marker.len();
        let end = text[start..].find('"')?;
        Some(text[start..start + end].to_string())
    }

    fn process_output_id(result: &ToolResult) -> Option<String> {
        Self::process_output_attr(result, "process_id")
    }

    fn process_output_next_cursor(result: &ToolResult) -> Option<u64> {
        Self::process_output_attr(result, "next_cursor")?
            .parse()
            .ok()
    }

    fn process_output_has_entries(result: &ToolResult) -> bool {
        result
            .output
            .as_str()
            .and_then(|text| {
                let start_marker = "<![CDATA[";
                let start = text.find(start_marker)? + start_marker.len();
                let end = text[start..].find("]]>")?;
                Some(text[start..start + end].trim())
            })
            .is_some_and(|body| !body.is_empty() && body != "[]")
    }

    fn progress_intent(result: &ToolResult) -> Option<ToolIntent> {
        if result.output.is_error
            || !result.output.values.iter().any(|value| match value {
                ToolValue::Text(text) => {
                    let trimmed = text.trim();
                    trimmed.contains("<file ")
                        || trimmed.contains("<http_response")
                        || trimmed.contains("<process_")
                        || trimmed.contains("<shell_output")
                }
                ToolValue::Image(_) | ToolValue::AI { .. } => true,
                ToolValue::Empty => false,
            })
        {
            return None;
        }

        if result.name.as_str() == "process_read" && !Self::process_output_has_entries(result) {
            return None;
        }

        Some(match result.name.as_str() {
            "read" => ToolIntent {
                family: ToolFamily::Read,
                target: Self::output_file_path(result)
                    .map(Self::normalize_target)
                    .unwrap_or_default(),
            },
            "fs_search" | "sem_search" => ToolIntent {
                family: ToolFamily::Search,
                target: result.output.as_str().unwrap_or_default().to_string(),
            },
            "shell" => ToolIntent { family: ToolFamily::Shell, target: String::new() },
            "process_read" => ToolIntent {
                family: ToolFamily::ProcessRead,
                target: format!(
                    "process_id={};next_cursor={}",
                    Self::process_output_id(result).unwrap_or_default(),
                    Self::process_output_next_cursor(result).unwrap_or_default()
                ),
            },
            other => ToolIntent {
                family: ToolFamily::Other(other.to_string()),
                target: String::new(),
            },
        })
    }

    fn target_field(target: &str, field: &str) -> Option<String> {
        target.split(';').find_map(|part| {
            let (key, value) = part.split_once('=')?;
            (key == field).then(|| value.to_string())
        })
    }

    fn process_signature_progressed(signature: &ToolIntent, progress: &ToolIntent) -> bool {
        let Some(signature_process_id) = Self::target_field(&signature.target, "process_id") else {
            return false;
        };
        let Some(progress_process_id) = Self::target_field(&progress.target, "process_id") else {
            return false;
        };
        if signature_process_id != progress_process_id {
            return false;
        }
        let cursor = Self::target_field(&signature.target, "cursor")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_default();
        let next_cursor = Self::target_field(&progress.target, "next_cursor")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_default();
        cursor < next_cursor
    }

    fn signature_matches_progress(signature: &ToolActionSignature, progress: &ToolIntent) -> bool {
        if signature.intent.family == ToolFamily::ProcessRead
            && progress.family == ToolFamily::ProcessRead
        {
            return Self::process_signature_progressed(&signature.intent, progress);
        }
        signature.intent == *progress
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

    #[cfg(test)]
    pub fn extract_assistant_messages<'a>(
        messages: impl Iterator<Item = &'a ContextMessage> + 'a,
    ) -> Vec<&'a forge_domain::TextMessage> {
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
            .attr(
                "suppression_count",
                detection.repetition_count.saturating_sub(1),
            )
            .attr("tool_family", detection.pattern_family())
            .attr("tool_intent", detection.pattern_intent())
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
                suppression_count = detection.repetition_count.saturating_sub(1),
                tool_family = detection.pattern_family(),
                tool_intent = detection.pattern_intent(),
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
