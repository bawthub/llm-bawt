# Temporary one-shot relay argument trace (TASK-1940)

**Source-only diagnostic, default off. No whitespace repair.** Publication does
not activate imported bridge code, arm capture, or authorize a relay/restart.
Activation belongs to the existing service authority, after actual-delta review.
Use only one approved, controlled, secret-free diagnostic relay from a selected
sender. Do not expose unrelated conversation content or use a protected workload.

## Narrow activation contract — not an instruction to activate now

There is no API, DB seed setting, environment toggle, prompt change or general
logging facility. When separately approved, the operator can prepare the fixed
`.llm-bawt-relay-trace` directory immediately below the **running bridge user's
home**. It must be owned by that user and mode `0700`. No directories are created
by the translator. Place an owner-only, regular, single-link `arm.json` there
(mode `0600`, maximum 1024 bytes), with exactly these operational fields:

```json
{"bot_id":"selected-sender-id","armed_at":0,"expires_at":0}
```

Both timestamps are Unix seconds. Replace zeros only during authorized arming:
`armed_at <= now < expires_at <= armed_at + 300`. Zeros above are intentionally
inert. The lease is at most five minutes, including the active capture, not five
minutes per event. A missing/unknown bot context never matches. Use a quiet,
selected sender so its **next** exact `mcp__bawthub__bots_send_message` tool call
is the approved diagnostic call; unrelated tools do not consume the arm. No
recipient is inferred or contacted by the trace. The code sends nothing.

The first matching call atomically creates `capture.jsonl` with exclusive create
and mode `0600`, then unlinks the arm. Existing artifacts are never overwritten,
followed through symlinks, or reopened for append. The artifact itself is the
cross-request/process/restart spent latch: recreating `arm.json` cannot rearm it.
Any deliberate subsequent run requires a separately approved archival/removal
of the spent artifact and a new short lease; there is no automatic retry.

Expired arms are inert even if left on disk. Active captures auto-close on their
expiry timer (including idle streams), byte/event limit, stream closure, or after
one emission. Removing an unclaimed arm cancels it; removal after claim does not
revoke the already-open capture, whose maximum lifetime remains the original
lease. No background worker or persistent general diagnostic switch is added.

## Evidence and bounds

Only one selected tool item's decoded argument strings are retained as JSONL:

- `start`: schema version, bot/request/attempt, translated message ID, upstream
  item ID and call ID, block index, lease expiry. No headers, prompts, history,
  provider tokens, account identifiers, or whole request/response objects.
- `delta`: exact original `event.delta` values, in arrival order, before buffer
  accumulation, including empty strings. JSON escaping is reversible and does
  not trim, join words or normalize whitespace/Unicode.
- `before_selection`: authoritative `event.arguments` (or custom-tool `input`),
  independently of the accumulated buffer, plus the actual selection rule
  (`done`, `buffer`, or `empty_object`). Absent values remain null; empty strings
  remain empty. The adapter's pre-existing selection rule is unchanged.
- `translator_partial_json`: the exact cleaned `partial_json` string at the
  translator's SSE yield point, after the existing sanitizers. It is **not**
  proof of SDK receipt, parsing, execution or downstream delivery: route guards
  still run later. Correlate the existing SDK and delivery evidence by call ID.
- `end`: `captured` only after the normal argument emission resumes; all other
  statuses are incomplete/failure evidence. Missing/invalid footer is also a
  failed capture, never a silently complete trace.

These are the OpenAI client's decoded Responses string values, **not raw HTTP
bytes or original SSE wire JSON**. Re-parse each JSONL record to compare strings;
comparison of the escaped on-disk representation to prose is invalid.

Hard limits: **65,536 artifact bytes total, 256 records total**, including header
and reserved terminal marker. Limit hits stop immediately, preserve only prior
whole records, and record `byte_limit` or `event_limit`; no truncated argument is
represented as complete. Missing `.done` uses the distinct `fallback_partial_json`
and `missing_done` failure status. Cancellation/error/early closure is incomplete.
I/O failures are swallowed without logging captured text or changing translated
arguments; partial filesystem writes may leave invalid JSONL and no usable footer.

Do not paste captures into shared logs/history. Keep the owner-only original
local; give the diagnostic owner only bounded, secret-free comparisons/hashes.
Retain/delete it under the approved diagnostic handling decision. This temporary
instrumentation should be removed once the upstream-versus-adapter boundary is
resolved; it is not a reusable logging platform.

## Verification surface

`tests/test_relay_argument_trace.py` uses synthetic streams and private `tmp_path`
artifacts only. It must not arm the real directory or send a message. Include the
existing translator/retry/truncated-tool regressions in focused verification.
No live provider, shared history, auth file, database or service probe is needed.
Tests and source review do not authorize runtime activation.
