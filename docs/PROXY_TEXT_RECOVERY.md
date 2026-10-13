# Verified text recovery (TASK-1064)

## Why not just append the partial answer?

The 2026-10-13 controlled `gpt-6-astra` probe accepted native reasoning + partial
assistant text, but repeated the WHOLE answer. Concatenating it would duplicate
visible content. A successful HTTP request is not proof of safe continuation.

The implementation therefore requires **exact-prefix validation**, not a hopeful
"continue" prompt, fuzzy matching, or restarting the whole agent turn.

## Contract

- Default adapters still fail upward after visible output. Standard ChatGPT SSE
  explicitly opts in; Lite/WS stays unchanged pending separate verification.
- The translator checkpoints only text and native reasoning actually forwarded
  to the SDK, in order. It never substitutes the upstream completed message,
  which may contain text the user never saw.
- On transient network errors, upstream 5xx or progress stalls, build a fresh
  body from the ORIGINAL translated request + cumulative committed output + a
  transport instruction asking for exact visible-prefix replay then continuation.
  Original tools, settings, prompt-cache key, account and routing headers remain
  unchanged. Do not append prior synthetic recovery instructions recursively.
- Buffer all new events until the candidate's visible text matches the entire
  original prefix exactly. Strip only that identical prefix. Then release the
  candidate's native reasoning and new suffix through the normal translator.
- Prefix mismatch, refusal, early terminal/tool, or buffer overflow fails closed
  as `api_error`. No candidate tool_use can escape before validation.
- **Any tool_use already forwarded forbids retry**, including a tool emitted by
  a successful continuation that later stalls. Cancellation propagates, no retry.
- Reasoning-only commitment, refusal, unhandled output items, missing checkpoint,
  permanent errors, auth failures and rate limits retain conservative barriers.
- Existing budgets apply: maximum 3 attempts for transient failures, 2 for
  supervised progress stalls. SDK message_start occurs once; block indexes remain
  monotonic. The failed upstream stream is closed before the next request.

## Bounds and caveats

Checkpoint and validation buffers are limited to 1,000,000 characters each;
validation also has a 2,048-event cap. Beyond those bounds recovery fails rather
than retaining unlimited output. This is request-local state, never a second
conversation store. No new config/environment knobs or schema migrations.

Recovery costs another inference request and prompt tokens for the checkpoint and
exact-prefix instruction. It can fail if the model does not reproduce that prefix
or the augmented request no longer fits the context. It does not guarantee that
all provider stalls become recoverable. Existing usage accounting reports the
terminal successful attempt; it is not a full bill for abandoned attempts.

The 90-second productive-idle deadline is unchanged. Stall logs now include the
last event type and bounded counts of nonproductive event types since the last
progress. They do not log payload text, encrypted reasoning or tokens. These
counts distinguish keepalive-only traffic from absence of any received events;
they cannot prove whether the upstream is secretly reasoning.

## Verification and activation

- `tests/test_proxy_continuation.py`: exact/mismatched and chunked prefix replay,
  native reasoning order and real Anthropic parser, open/network/stall retries,
  cumulative checkpoints, all bucket/tool barriers, cancellation and memory bounds.
- `RUN_PROXY_SDK_PROBE=1 ... pytest tests/test_proxy_reasoning_sdk.py`: isolated
  installed Claude CLI two-process resume persists reasoning from BOTH attempts
  and replays visible text exactly once. No credentials, tools or bot sessions.
- `scripts/benchmarks/task1064_continuation_probe.py --model gpt-6-astra`: opt-in
  real provider traffic, app-owned token broker. Injects a disconnect after 25+
  visible characters through the actual adapter. Real Anthropic SDK parses the
  recovered wire, and its assistant blocks are sent back upstream on a follow-up.
  Run in the test environment with `anthropic` installed; no provider credentials
  are printed. The probe has no tools and touches no live bot session.
- Live 2026-10-13 result: 2 opens for interrupted+recovered response, squares 1–30
  exactly once, native reasoning retained; third open accepts SDK-shaped history
  and returns `900` on the follow-up. Earlier raw mixed-reasoning A/text A/B/text B
  acceptance also succeeded, but without prefix validation duplicated text.

Activation verified 2026-10-13: Nick explicitly approved one Claude bridge
restart; managed job `4a56f71dbcee4e3789e20dab0e18b495` succeeded (exit 0) at
13:15:36Z. Container start advanced to 13:15:36.134455217Z, Docker health was
healthy, bridge `/health` returned `redis:true`, and app `/health` returned
`status:ok`. The approval-continuation conversation resumed through that bridge.
No app, Redis or other bridge restart was issued. At activation, source was uncommitted.

Ordinary turns do not prove fault recovery; retain the isolated injected-failure
probe receipt above as that evidence. Future source changes require a newly
approved restart; implementation/probes never authorize a restart themselves.
Rollback: revert TASK-1064 source changes (preserving unrelated work), then request
another specifically authorized bridge restart. Do not restart Redis or the stack.
