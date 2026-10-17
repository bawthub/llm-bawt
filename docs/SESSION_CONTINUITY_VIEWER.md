# Session continuity viewer (TASK-1117)

The viewer is read-only. Its anchor is the existing `sessions.id`; it does not
create a parallel session registry, join table, or copy of message bodies.

## Persistence and ownership

`history_seed.build_context_seed` carries `source_id` and `shortened` alongside
each selected message. Summary compaction preserves `Message.db_id`. These
annotations are transport/audit metadata, not part of the rendered SDK briefing.
The app's streaming and non-streaming dispatch paths call the same
`record_session_continuity` **after** `/new` has resolved the destination session.
The live caller's owner-scoped instance is used for seed assembly.

The existing `sessions.session_metadata.continuity` JSON object contains:

- version, captured_at, model, applied sizing/scope settings
- ordered source_message_ids (null for an unreferencable source), shortened_indices
- summary_count, message_count, approx_tokens, oldest/newest timestamps
- payload_digest, status, and completed_at when a bridge receipt arrives

No message content is persisted in this manifest. Source messages keep their
original `messages.session_id`; inheritance does not change message ownership.

`prepared` means selected, **not successfully restored**. The bridge posts a
receipt only after attempting the synthetic transcript write. A SHA-256 digest
of ordered source IDs, roles, text and timestamps binds the receipt to the exact
selected payload. Stale/different receipts cannot acknowledge a newer capture;
a failure cannot downgrade an already restored capture. Existing top-level
session metadata is merged atomically, and a restored/empty capture is retained
on subsequent hydration. `empty` means no earlier messages were selected; a warm
resume (`inject_messages is None`) does not overwrite provenance.

This captures the initial continuity record, not every later SDK compaction or
model switch. The model is a recorded value, not a complete model history.

## API

- `GET /v1/sessions/{session_id}/details?bot_id=...&user=...&offset=0&limit=50`
- `GET /v1/sessions/by-message/{message_id}/details?...` resolves a reset receipt
  through the canonical message-to-session join, including continuous chat view.
- `POST /v1/sessions/{session_id}/continuity-receipt?bot_id=...` is an internal
  bridge callback with `{payload_digest, restored}`; the dedicated public viewer
  routes do not expose it.

Reads return `{session, continuity, messages, total_count, offset, has_more}`.
Each source result has its original index, ID, available/shortened flags, and
current role/content/timestamp/source-session when available. Pages are bounded
at 100 sources. Destination ownership is checked before reading provenance;
resolving sources also checks bot and original source-session user ownership.
Deleted, foreign-owner, missing, and unreferencable sources are unavailable—never
looked up across users or by approximate timestamp. Session registry timestamps
are serialized as UTC explicitly.

The BawtHub public proxy in `server/routes/session-search.ts` binds authenticated
identity ahead of the LAN user hint. Its GET-only search/details routes are
shared with TASK-1119, not duplicated by the viewer.

## UI contract

`@/app/chat/threads/SessionDetailsDialog` exports:

```
SessionDetailsDialog({sessionId, botId, userId, open, onClose})
// onClose(reason?: 'dismiss' | 'navigate')
```

The shared viewer is used from assistant reset receipts and Spotlight session
results. Navigation uses the existing exact-thread bot navigation hook, never
session creation/activation. Spotlight uses the close reason to reopen search on
dismissal, but not when navigating to a conversation.

The default **Carried context** view is a readable transcript, not an accordion
per message. A wider desktop dialog (responsive to phone/landscape) keeps bot
identity, title, restore status and counts above the scroll area. Short messages
render immediately using shared Markdown; long messages have an in-place Read
full message control with no duplicate preview/body. Summaries have a distinct
surface. Consecutive exact `/new` commands and assistant reset receipts collapse
into an Earlier session reset disclosure without dropping or reordering sources.

**Session details** is a separate secondary view for dates, recorded model,
lineage navigation and the expandable technical record. Open conversation remains
visible in the fixed footer. The single content scroller resets on view/page
changes; source pagination and the public component/close contract are unchanged.
Presentation grouping lives in `sessionDetailsPresentation.ts`, source rendering
in `SessionContinuityTranscript.tsx`, and modal/query ownership in
`SessionDetailsDialog.tsx`.

Legacy sessions explicitly say provenance was not recorded. Sources show their
**current stored text**, not an immutable reconstruction of the original SDK
briefing. Summary compaction and per-message shortening are explained. Raw IDs,
settings, and additional metadata stay under Technical details.

## Verification and activation

- `tests/test_session_continuity.py`: hermetic ownership, paging, summary ID,
  manifest/no-content, legacy API, bridge success/failure, warm-resume tests.
- `tests/test_session_continuity_pg.py`: real JSONB merge and stale-receipt CAS in
  the shared private temporary PostgreSQL fixture; never live data.
- Frontend `sessionDetailsApi.test.ts`, `sessionDetailsPresentation.test.ts` and
  `sessionDetails.browser.tsx` fixture (legacy/prepared/failed/empty, paging/errors).
- BawtHub `scripts/verify_session_details.cjs`: 80 checks across desktop, 390px,
  320px and landscape, plus paging/retry/close/navigation. Isolated fixtures only.
- `scripts/verify_spotlight_sessions.cjs`: integration with preserved search and
  distinct dismiss/navigation close reasons.

Frontend HMR does not activate Python changes. App code needs its authorized ops
restart; bridge receipt code needs Nick's explicit authorization for that exact
bridge restart. Until both are active, do not claim an end-to-end live reset was
verified. Older sessions cannot be backfilled accurately from a fresh preview.
