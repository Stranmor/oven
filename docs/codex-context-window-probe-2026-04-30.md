# Codex Context Window Probe - 2026-04-30

## Scope

Active Forge route was tested through the Antigravity OpenAI-compatible
`/v1/chat/completions` endpoint with:

- `provider_id`: `openai_compatible`
- `model`: `gpt-5.5`
- payload shape: one short system message plus one repeated `x ` user message
- generation cap: `max_tokens = 1`

No API key or credential material is recorded here.

## Results

| Probe | Observed input tokens | Result |
| --- | ---: | --- |
| `240_000` repetitions | `240_016` prompt tokens | HTTP 200, returned `OK` |
| `260_000` repetitions | `260_016` prompt tokens | HTTP 200, returned `OK` |
| `400_000` repetitions | not returned | client read timeout after 240s on first run |
| `400_000` repetitions | not returned | HTTP 503 after 216.42s: all accounts exhausted |

## Conclusion

The active Codex-compatible route is not capped at 256k input tokens:
`260_016` prompt tokens completed successfully.

The 400k probe did not produce a definitive end-to-end success. It also did not
return `context_length_exceeded`; the observable failure was account exhaustion
after upstream processing time. Treat 400k as operationally unproven on the
current account pool, not as proven unsupported by context length.

Forge should not assume that a configured 400k context window is safely usable
for every request. Subagent prompts still need compact context because requests
above 260k enter a slow/high-risk operational zone even when they are not
rejected by a hard 256k limit.
