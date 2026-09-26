# OpenAI proxy reasoning parity (TASK-931)

## Ownership and contract

Claude Code owns transcript order, persistence, resume and compaction. The proxy
translates the history it receives; it neither reconstructs omitted history nor
runs a second retention/compaction policy.

OpenAI reasoning summaries are display text, not replayable internal state.
`store:false` does **not** mean discard reasoning. Native Codex requests
`reasoning.encrypted_content` and carries native Reasoning items in request input.

The ChatGPT adapter now opts into a request-local `ReasoningCodec` after resolving
credentials. Completed native reasoning items are carried inside an SDK thinking
signature as a versioned `bawt-responses-reasoning-v1:` envelope. This wrapper is
base64-encoded JSON, **not additional encryption or an Anthropic signature**. The
provider's encrypted content remains byte-for-byte intact. The envelope preserves
native ID, summary structure, optional content/status and encrypted content;
readable thinking deltas remain independent display output.

On replay, compatible envelopes become native Responses reasoning items at the
same position. Encrypted-only items survive even when no summary is displayed.
No in-memory or external side table is needed. When the SDK removes a block via
compaction, the proxy does not resurrect it.

## Compatibility

Replay currently requires the same adapter, upstream model, account, and base URL.
The account/URL scope is hashed; tokens are never stored in the envelope. Token
refresh within an account preserves scope. A change of scope during an auth retry
fails visibly rather than sending an already-built body to a different account.

Exact-model matching is a conservative limit, **not a claim that OpenAI forbids
all cross-model replay**. Relax it only against verified provider capabilities.
Different scopes and legacy/unmarked signatures are omitted from the outbound
OpenAI request, not removed from the saved transcript. Malformed envelopes bearing
our version marker fail visibly. Display-only placeholders are never promoted to
native reasoning. Existing raw signatures cannot be reliably attributed; there
is no retrospective migration based on signature-format guesses.

Direct Anthropic traffic bypasses this proxy. This change does not implement a
native-Anthropic outbound sanitizer for sessions switching back from OpenAI.
That existing provider-switch compatibility boundary needs separate verification
before claiming bidirectional opaque-state interoperability. These envelopes
must never be forwarded as genuine Anthropic signed reasoning.

## Parity matrix

Reference: local Codex checkout `e72da2b5` (2026-09-26),
`codex-rs/core/src/client.rs`, `client_common.rs`, `context_manager/history.rs`.

| Concern | Native Codex | Proxy after TASK-931 |
|---|---|---|
| Replay native reasoning with store:false | Yes | Yes, compatible tagged items |
| Standard Responses reasoning.context | Omitted; source documents current_turn default | Omitted |
| Responses Lite reasoning.context | all_turns | all_turns, including Lite HTTP fallback |
| Lite parallel tool calls | Disabled | Disabled |
| Session resume | Native history | SDK-carried envelope, no side table |
| Compaction | Harness owned | Claude Code owned |
| Incremental WS previous_response_id | Supported by native client | Still full-history; unchanged intentionally |
| Legacy raw thinking signatures | Native history has native item provenance | Conservatively not replayed |
| Cross-model opaque state | Provider/client-specific | Exact-model only until verified |

Server use of old reasoning is separate from client replay. Do not infer actual
server retention or tokenizer accounting from a saved transcript or cache rate.
The proxy does not translate Anthropic context-management edits into purportedly
equivalent OpenAI fields; the APIs are not identical.

## Retry and cache behavior

Display-only partial thinking may still be spliced on retry under the existing
policy. Once a replayable native reasoning signature is forwarded, retry must
fail upward like visible text: it cannot retract that state or mix two attempts.
Tool-call no-replay protection remains unchanged.

Replayed items are deterministic and opaque bytes are preserved. New tool hops
append to stable prefixes. Turning this on can change the prefix on the first
request containing newly preserved reasoning; a high cache hit rate alone does
not prove correct reasoning continuity.

## Verification

- `tests/test_proxy_reasoning.py`: real Anthropic SSE parser, native-item ordering,
  multiple items, encrypted-only output, legacy/foreign handling, JSONL resume,
  compaction absence, adapter final payloads, standard/Lite settings and prefix
  stability.
- `tests/test_proxy_reasoning_sdk.py`: opt-in **installed Claude Agent SDK/CLI**
  two-process resume against an isolated loopback fixture. No provider inference,
  tools, real credentials or bot sessions. Run with `RUN_PROXY_SDK_PROBE=1`.
- Existing proxy, retry, transport, concurrency, supervisor and Kimi regressions.

No live OpenAI acceptance claim is made by these hermetic tests. Deployment needs
an explicitly authorized Claude bridge restart; none is implied by source edits.
