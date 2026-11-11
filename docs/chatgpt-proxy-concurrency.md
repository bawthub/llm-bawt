# ChatGPT proxy concurrency and identity contract

The Claude Code bridge proxy intentionally permits parallel upstream streams.
TASK-685's deterministic 1/2/3/5-stream tests show independent progress with
no serialization, starvation, payload crossover, header crossover, or event
loop blocking. The live evidence that motivated the task also showed a healthy
uvicorn loop at three active streams. Therefore there is **no concurrency
semaphore by default**. Latency growth alone is not evidence that serializing
bots would improve throughput.

## Credential and connection ownership

`GET /v1/providers/openai_chatgpt/token` returns `expires_at` as Unix epoch
**seconds** (the JWT `exp` claim). The bridge caches it until five minutes
before expiry and uses an async single-flight lock for cold/near-expiry broker
resolution.

Every registered provider adapter owns one lifecycle-scoped async HTTP pool.
FastAPI starts the pools at proxy startup and closes them once at shutdown.
Responses API requests use request-local OpenAI client views over that shared
pool, so a refreshed bearer token and provider headers cannot mutate another
concurrent request or leak stale state.

## Durable conversation identity and caching

The app already resolves the canonical durable DB `thread_session_id` before
Redis dispatch. The bridge hashes a versioned tuple of:

1. bot ID;
2. bot/user `session_key`;
3. durable `thread_session_id`.

Only the 32-character opaque result crosses into the proxy through Claude
CLI's `ANTHROPIC_CUSTOM_HEADERS`; raw user/thread identifiers never do. `/new`
rotates the durable thread before dispatch, so it necessarily rotates the
opaque identity. Tool-loop calls and later resume turns retain it.

For the ChatGPT codex backend, this identity is used for both:

- upstream `session_id` header (routing/affinity/concurrency identity);
- `prompt_cache_key` request field (explicit prompt-cache routing key).

Using one durable value makes the isolation boundary explicit and avoids
volatile cache busting. A direct/legacy proxy caller without bridge metadata
falls back to the deterministic opening-prefix hash; real bridge traffic does
not derive identity from prompt content.

## Observability

Each stream emits structured `proxy_stream_start` and `proxy_stream_complete`
logs with request ID, provider, non-secret account hash, bot, conversation hash,
provider/account active counts, queue time, local setup time, upstream TTFB,
total stream duration, input/cached/output tokens, cache-hit percentage, status,
and cumulative provider 429/5xx counts. `queue_ms` is currently zero because no
limiter exists.

If production data later proves account saturation, add a configurable
provider/account semaphore at the observability boundary, report its actual
queue wait, and leave the default unlimited until a measured safe limit is
known. Do not add a bridge-global lock.

## Controlled live smoke

After review, restart only the Claude Code bridge (explicit operator approval
required) so it imports the changed proxy modules. Then run two distinct bots'
threads plus one `/new` turn and verify:

- same-thread logs retain `session_hash` across tool loops/resume;
- the two bots' hashes differ;
- `/new` produces a new hash;
- overlapping streams show `active_provider` above one;
- `setup_ms`, `upstream_ttfb_ms`, and `stream_ms` are independently populated;
- repeated tool-loop calls show high `cache_hit` after the opening call;
- broker-fetch log volume is far below proxy-start volume.

No live smoke should be run from an in-flight bridge-hosted agent turn: the
required bridge restart terminates that turn.
