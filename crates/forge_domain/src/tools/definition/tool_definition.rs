use derive_setters::Setters;
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::ToolName;

///
/// Refer to the specification over here:
/// https://glama.ai/blog/2024-11-25-model-context-protocol-quickstart#server
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Setters)]
#[setters(into, strip_option)]
pub struct ToolDefinition {
    pub name: ToolName,
    pub description: String,
    pub input_schema: Schema,
}

impl ToolDefinition {
    /// Create a new ToolDefinition
    pub fn new<N: ToString>(name: N) -> Self {
        ToolDefinition {
            name: ToolName::new(name),
            description: String::new(),
            input_schema: empty_object_schema(),
        }
    }
}

fn empty_object_schema() -> Schema {
    serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": {}
    }))
    .expect("empty object tool schema must be valid JSON Schema")
}

pub trait ToolDescription {
    fn description(&self) -> String;
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_new_defaults_to_empty_object_schema() {
        let fixture = ToolDefinition::new("empty_tool");

        let actual = serde_json::to_value(&fixture.input_schema).unwrap();
        let expected = json!({
            "type": "object",
            "properties": {}
        });

        assert_eq!(actual, expected);
    }
}
