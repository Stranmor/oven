Terminalizes the current conversation goal as satisfied or blocked using the active ToolCallContext conversation_id.

Use when the model has proven the conversation goal is complete or has reached a closed blocker. The schema intentionally has no conversation_id argument; the runtime conversation context is authoritative. Satisfied requires summary and evidence and forbids blocker_kind. Blocked requires summary, evidence, and blocker_kind. Unknown arguments are rejected.
