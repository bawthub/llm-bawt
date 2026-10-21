# Isolated recall (TASK-1210)

`context_recall` researches history outside the calling agent's transcript and
returns a compact **extractive evidence packet**, not a narrative answer or raw
search results. It is read-only: no memory writes/access counters, session
creation, recalled-message marking, task mutation, tool execution, or automatic
conversation injection. This is not TASK-1151's all-tool output cleanup.

## Tool contract

The tool accepts one typed `request` object:

```json
{"request":{"query":"Why did we change the relay background?","sources":["messages","memories"],"output_tokens":1000}}
```

- `query`: nonblank, at most 1,000 characters and 256 o200k_base tokens.
- `sources`: one to four distinct values: `messages`, `memories`, `tasks`,
  `tool_evidence`. Defaults to caller-owned messages and memories.
- Selecting `tasks` explicitly opts into shared-task discovery. Optional
  `task_ids` (at most three cuids / TASK-N references) narrows that search.
  Omit IDs to discover forgotten task references by meaning.
- `output_tokens`: 400–1,600, default 1,000. Bounds the **complete serialized
  packet**, including citations, handle, coverage and metrics.
- `handle`: optional opaque continuation. Repeat the original `sources` and
  `task_ids`; changing either fails closed. The initial question and up to three
  follow-up questions are retained as bounded search intent. Evidence is always
  refreshed, never copied out of retained transcripts.

Identity is **not an argument**. The tool verifies the current signed turn
capability and inherits `(bot_id, user_id, session_id)`. Missing/invalid/expired
capabilities cause an error before retrieval. Currently Claude-code and Codex
bridge turns propagate that capability; unsupported callers fail closed.

### Result

A single compact JSON text block (no duplicate structured-output copy):

- `status`: `ok`, `no_match`, `insufficient_evidence`, or `error`.
- `findings`: at most six `{source, id, quote}` entries. Each quote is an exact
  substring of this call's authorized evidence, at most 400 characters / 100
  reference tokens. IDs are validated against this call's evidence map. There
  is no freeform model prose to accidentally reproduce an entire document.
  The model only points at evidence: its quote is located in the excerpt
  (tolerating whitespace reflow, dropped or added `*`/`_`/backtick markers at
  word edges, and `...`/`…` elisions, each elided fragment at
  least three words, in source order) and the packet returns the exact
  contiguous source span, never the model's text. A grounded span longer than
  `MAX_QUOTE_CHARS`/`MAX_QUOTE_TOKENS` (contracts.py) is shortened to a prefix
  ending on a sentence or word boundary, not dropped. That finding carries
  `"truncated": true` (and `uncertainty` says so): it is a prefix and may omit a
  trailing qualifier or correction. Packet-level `truncated` also covers clipped
  evidence excerpts, so read the per-finding flag.
- `rejected_findings`: count of selected excerpts that failed schema or
  grounding and were omitted. Each is dropped individually; if every selection
  is rejected the call is `error`/`ungrounded_synthesis`.
- `uncertainty`: server-authored warning that these are untrusted historical
  excerpts, neither exhaustive nor proof of current state. When
  `rejected_findings > 0` it also says omitted excerpts may include a later
  correction.
- `coverage`: per-source success/no-match, counts and prefix-fallback counts;
  task results additionally report semantic/keyword/explicit mode and withheld
  candidates. Incomplete task provenance is `partial`, not a clean no-match.
  A fatal source failure preserves bounded coverage but returns **no quotes**.
- `scope`: explicitly notes that tool evidence is metadata-only, tasks exclude
  responses, and memory/history retrieval is caller scoped.
- `truncated`: indicates excerpt/input/output clipping. A bounded search cannot
  prove absence outside its authorized, matching excerpts.
- `handle`: opaque, session/bot/user-owned, expires after at most 15 minutes or
  the original capability's remaining life, whichever is shorter. Four-question
  depth, eight entries per owner and 256 process-wide entries bound state. Only
  query intent/scope is stored, never source bodies. Handles are process-local;
  a restart, eviction or another worker can invalidate them. Expired, unknown,
  tampered or wrong-owner handles share the same error.
- `metrics`: local o200k_base counts for utility input, utility output, admitted
  evidence and complete parent packet, source-call count, model-call count and
  elapsed milliseconds. These are reproducible context measurements, **not
  billing or cache savings**. Provider tokenizers/framing may differ.

A valid search with no authorized matching evidence yields `no_match`. Evidence
that cannot support findings yields `insufficient_evidence`. Timeouts, bad model
output, unavailable sources and policy errors are errors, never empty successes.
All failure codes are server-owned and omit source text/provider exceptions.
There is **no raw-result fallback**.

## Sources and authority

`SourceProvider` and `Synthesizer` are small ABCs. Orchestration, database reads,
HTTP task reads, bounded workers, continuations, token accounting and prompt
construction are focused modules under `llm_bawt/recall/`. Existing oversized
MCP storage/client modules are untouched; server.py only imports registration.

### Current DB policy

`RecallDatabase.scope` reads the current `bot_profiles` search policy directly,
with a bounded SELECT. Errors, absent/ambiguous policy, sanitation collisions,
and unknown callers fail closed. There is no duplicated hardcoded protected
bot list and no reuse of the existing fail-open aggregate-search helper.
**Opted-out bots cannot use recall, even for their own history.** A policy change
is re-read on every request, including follow-ups.

### Messages

Read only the live `messages` parent; join canonical sessions for caller bot and
non-null caller user. Only active/archived sessions and user/assistant messages
qualify, authored by the caller user, the caller bot, or — **the single
cross-bot boundary** — a delivery the caller actually received: a `role=user`
row in the caller's own partition and own session whose bot author is in the
current DB policy's searchable set and is not excluded/protected
(`allowed − excluded − caller`, computed per request). Protected bots such as
Mira are never in that set, so their deliveries stay unreachable even inside the
caller's own history. Deleted sessions, forgotten messages, summaries, unknown
legacy authors, other users and imported other-bot assistant rows are omitted.
Those omissions reduce recall coverage deliberately. Memories and tool evidence
keep the stricter caller-user/own-bot rule.

Query terms keep identifier parts (`maintenance_model`, `TASK-1210` →
maintenance/model, task/1210) rather than dropping them. Hits rank by **query
coverage** first: the share of the query's distinct lexemes a message contains,
rounded up into `COVERAGE_BUCKETS` (4) so near-equal coverage ties. A message
matching one or two generic words of the question ("nick", "error") never
outranks one matching most of it, however old. Within a bucket, full-text rank
divided by one recency term (`RECENCY_HALF_WEIGHT_DAYS`: rank halves at 14 days)
orders hits, so the newer of two equally relevant statements wins. This selects
up to six **hits**. Each hit carries at most two **follow-ons** (`MAX_FOLLOW_ONS_PER_HIT`, defined beside the recency weight in `recall/contracts.py`): the next
messages in the same session within 30 minutes, under the identical authority,
partition and session filters — never another session, bot or user. Follow-ons
reach term-free corrections ("No summaries. I said message history") that FTS
cannot. Each message appears once; a message that is also a hit stays a hit.

SQL extracts an **original contiguous window** of at most 2,400 characters near
the whole query phrase, or the longest matching meaningful query term. No full
message body crosses the DB boundary. Stem-only matches without a literal anchor
use a prefix and explicitly report `head_fallbacks`. Excerpts are then capped at
512 reference tokens. Offsets and truncation accompany internal evidence;
generated/joined search highlights are never treated as original quotation text.

Every evidence item carries server-owned `at` (UTC) and `speaker` (`user`,
`assistant`, or `bot:<id>` for a received delivery), plus `follows` for a
follow-on. Admitted evidence reaches the utility model **oldest-first**, and the
fixed instruction tells it to quote the latest statement when later evidence
corrects or updates earlier evidence on the same point.

### Memories

Memories have no `user_id` column. A memory must have 1–64 source-message IDs,
and **every** ID must resolve to an authorized message under the same rules.
Unattributed/manual, missing/forgotten-source and mixed-user facts are excluded.
Queries read live memory rows without search helpers that increment access
counters. Consolidation deletes originals; the current schema has no superseded
column to invent. Relevance windows use the same bounded SQL projection.

### Shared tasks

The existing task API's read policy is shared; assignment is not read authority.
One read-only semantic-search POST discovers at most three candidates (the
existing endpoint explicitly labels its keyword fallback if embeddings fail).
Explicit IDs bypass discovery. Each candidate is fetched through bounded HTTP,
then checked against current task/project bot ownership, association user/bot
provenance, and a complete bounded activity audit before admission to the model.
Unknown/protected bot provenance or other-user associations deny the candidate.
System/cron and null human-actor activities are normal neutral shared-task events,
not authorization grants. Activity headers are attribution, not trusted authority.

Only ID/title/status/description are admitted; never task responses, step output,
project instructions, related-task content, or other bots' conversation turns.
Long descriptions use a query-relevant original window. If a candidate's complete
provenance cannot fit (for example, more than 100 activities or an oversized
association response), it is withheld and coverage reports incompleteness; other
sources can still return valid evidence. No hidden pagination/fan-out escapes the
budgets. The legacy task API sends larger JSON before projection; stream caps
bound that transport, and oversized candidates are omitted rather than bypassed.

### Tool evidence

Search caller-owned tool names and return only canonical receipt metadata:
name, approval status, timestamp, result SHA-256 and recorded byte/character sizes.
The tool/turn/message/session chain must match the caller bot and user.
**Arguments, result bodies, blobs and attachments are neither searched nor sent
to the utility model.** They can contain cross-bot/secret data even in an owned
turn. Approval status or a digest does not prove execution success. Payload
research remains outside this slice.

## Budgets and isolation

| Boundary | Ceiling |
|---|---:|
| Whole MCP request, including policy | 30 seconds |
| Concurrent research calls per process | 4; excess rejected, no queue |
| Source-provider calls | 4 |
| Hits per source / admitted total | 6 / 24 |
| Message follow-ons | 2 per hit; admitted only after every source's hits, within the same 24 / 6,000 caps |
| Each source excerpt | 2,400 characters, then 512 reference tokens |
| Utility system + question + evidence | 6,000 reference tokens |
| Utility calls | 1, no agent/tool loop |
| Utility completion | 20 seconds, transport timeout and retries disabled |
| Concurrent utility workers | 2; timed-out workers retain slots until finished |
| SQL workers / statement timeout | 4 / 2 seconds; read-only transactions |
| Task HTTP requests / aggregate bytes | 15 / 256 KiB |
| One task HTTP response / timeout | 64 KiB / 4 seconds |
| Model response transport | 16 KiB |
| Complete parent packet | requested 400–1,600 reference tokens |

Python cannot forcibly cancel synchronous SDK/DB calls. Shared bounded workers
retain their admission slot after caller cancellation until the operation really
finishes; no unbounded orphan retries/queue can form. Database statement and SDK
transport timeouts provide the underlying deadmen. A provider that lacks the
bounded utility contract fails visibly rather than using unbounded inference.

MCP runs separately from the app. `UtilitySynthesizer` calls the app's
`/v1/llm/recall-complete` route at the configured service host/port. That route
reuses the global DB-backed `maintenance_model` resolver, not a bot, cached-client
insertion order or extraction-model fallback. It accepts only prompt/system/token
budget, has no caller model/bot override, checks the catalog's context/output
limits, and sanitizes failures. The utility context contains only the extractive
instruction, question and authorized bounded evidence: no persona, parent
transcript, tools or automatic writes.

**Instruction source:** the system prompt is the prompt-registry key
`recall.synthesis.system`, editable in BawtHub's Prompt Registry, and it is never
hardcoded. `recall/prompt.py` holds only the seed default. Each recall resolves the
body once, trying the caller's bot override, then the global row, then the seed; a
blank body or a registry error falls back to the seed. The service budgets evidence
against that exact text and sends it, so an edit can't break the 6k input cap. Any
edit must keep the JSON `findings` contract, because grounding still re-checks
every quote against the source bytes.

**Data egress:** authorized history excerpts go to whatever utility model the DB
currently selects, potentially a cloud provider. No model/config change is made
by this feature. This is not an offline-only or zero-cost promise.

## Verification and activation

Primary tests: `tests/test_isolated_recall.py` and
`tests/test_isolated_recall_sources.py`. Independent packet/quality comparisons:
`tests/test_isolated_recall_evaluation.py` and its fixture. Bounded utility/client
regressions: `tests/test_recall_completion.py`, `tests/test_raw_completion_model.py`.
Run through `make test-target NODES='...'`; catalog coverage is additionally
checked by `tests/test_mcp_catalog.py` and `scripts/mcp_catalog_report.py`.

A synthetic-only PostgreSQL SELECT verified a query hit after >15k tokens returns
an exact bounded late window, not the document head. No protected conversation
or live memory was read. Deterministic fake-model evaluations establish packet
shape/grounding/context reduction, not real-model recall accuracy or live rollout.

Source registration/tests are not activation. This work does not authorize a
commit, app restart, config mutation, deployment or protected-bot interaction.
