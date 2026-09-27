# Abort lifecycle (TASK-952)

## Authority and ordering

`POST /v1/chat/abort` accepts `turn_id` and optional `source` (`unknown`,
`chat_stop`, `bot_list_stop`). Source is a caller-reported action, not verified
human identity. Peer is the immediate HTTP peer (often a proxy); actor remains
unknown unless a trusted identity mechanism supplies one.

1. Lock the requested turn row. A finished turn is not reclassified.
2. Persist `status=cancelling` and the request provenance in
   `turn_logs.request_json.abort_request`, **before** calling the bridge.
   `cancelling` is nonterminal: `ended_at` remains NULL.
3. Signal the exact registered worker. Native and pre-dispatch turns return
   `cancelling:worker_signalled` (acceptance, **not** termination). Native workers
   stop consuming and close the provider iterator on their owning thread; a
   blocking provider/tool must return control before termination is confirmed.
   Agent dispatch binds and persists its request-local identity **before**
   publishing chat.send, not on first output. A Stop during command publication
   is forwarded after chat.send; it cannot overtake dispatch and disappear.
   Already-dispatched bridge turns receive `chat.abort` with both `sessionKey`
   and the exact `requestId`. Bridges ignore RPCs addressed to another backend.
   Only boolean `cancelled: true` or `aborted: true` confirms cancellation;
   generic `ok: true`/`no_active_task` does not. No shared bot request scalar is
   used for targeting.
4. The worker finalizer or acknowledged bridge cancellation commits the
   terminal outcome. If cancellation intent won the row lock, a racing EOF or
   normal result cannot overwrite it with success. Later enrichment may still
   add partial text, usage and file metadata.
5. Unconfirmed/failed abort RPCs leave the turn nonterminal and return HTTP 503.
   They are retryable against the same exact request. No timer hides the turn
   and no synthetic completion claims that execution stopped.

The existing explicit stale-reaper timeout repair is retained only for ordinary
abandoned turns. Reaping excludes registered live workers and cancellation intent,
and settles unresolved tools transactionally for rows it does terminate. Other
terminal outcomes cannot regress/reopen. The request audit survives final
prompt-payload replacement and records the acknowledgement outcome and timestamp.

## Tools

Terminal persistence settles unresolved tool rows in the same transaction as
the parent turn. A missing result stays NULL, `is_error` is not invented,
`result_complete=False`, and `ended_at` records when observation ended—not proof
that an external process exited. History returns `status=interrupted`.

Tool starts serialize against parent termination; late starts on ended parents
cannot create new running cards. Late genuine results reconcile by canonical
call/tool-use IDs and can replace interrupted outcomes without reopening the
turn. Legacy unfinished records on ended parents use the same interrupted
projection on history reads; no manual database rewrite is needed.

## Delivery and UI

`turn_complete` carries the canonical status, assistant/trigger/turn identities,
session, partial response and changed-file summary. Repeated cancelled snapshots
are idempotent. A diff summary means work was recorded, not that the turn
succeeded or a detached job completed.

BawtHub settles only the matching stream, timer and tool projection, including
background threads. Stop discovery is scoped to the displayed conversation, and
HTTP failures are visible without clearing live state. Resume/reconnect treats
`cancelling` as pending and never invents completion from a missing recent row.
It flushes buffered text before retiring transport, and late
reader callbacks cannot overwrite a replacement stream. Interrupted tools have
a neutral label, not a success check or a running spinner. Actual tool results
outrank interruption; interruption outranks stale pending snapshots.

The existing client completion ledger retains 200 entries. It is not an
indefinite tombstone store; history hydration remains authoritative beyond that
window. Browser state is not persisted as a competing lifecycle source.

## Failure containment and boundaries

- Stream queue closure and generation-done signals are in `finally`, even if
  persistence, logging or event publication throws.
- Response/error strings are literal logging data, not Rich markup.
- Claude cancellation terminal events preserve partial text, usage and IDs.
- OpenClaw receives an exact gateway runId at chat.send acknowledgement. Abort
  waits at most five seconds for that identity, then fails retryably if unknown.
- Local inference retains the session lock until its cancelled native worker
  actually exits; cancellation is observed at a token boundary.
- Detached SSH/host jobs are **not** claimed to be killed by agent cancellation.
- A native/non-bridge execution without an exact cancellation handle remains
  unconfirmed rather than being falsely marked aborted by this endpoint.

## Activation and incident provenance

Backend/bridge processes must reload the source to activate this contract. No
schema migration is required. The original Caid incident's retained access logs
identify a proxy-originated HTTP abort, not a proven human Stop action. Provenance
added here is prospective; it cannot reconstruct that missing historical actor.
