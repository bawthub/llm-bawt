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
- `sources`: one or more distinct values: `messages`, `memories`, `tasks`,
  `task_details`, `tool_evidence`, `tool_args`. Defaults to `messages`,
  `memories` and `tasks` (TASK-1557: a quarter of real misses were answers
  that lived only in a task). `task_details` (reviews and step outputs),
  `tool_evidence` (receipt metadata) and `tool_args` (attempted git clones)
  are opt-in depth.
- `tasks`/`task_details` discover shared tasks by meaning. Optional
  `task_ids` (at most three cuids / TASK-N references) narrows that search
  and requires one of the two task sources.
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
- `findings`: at most six `{source, id, quote}` entries, with source-owned
  `provenance` (`source_bot`, `session_id`, `at`, `follows`, and `review_ref` only
  when present in the authorized excerpt) and offsets for split quotes. Each
  quote is an exact substring of this call's authorized evidence, at most 400
  characters / 100 reference tokens. IDs are validated against this call's
  evidence map. There is no freeform model prose to reproduce a document.
  The model only points at evidence: its quote is located in the excerpt
  (tolerating whitespace reflow, dropped or added `*`/`_`/backtick markers at
  word edges, and `...`/`…` elisions, each elided fragment at
  least three words, in source order) and the packet returns the exact
  contiguous source span, never the model's text. A grounded span longer than
  `MAX_QUOTE_CHARS`/`MAX_QUOTE_TOKENS` (contracts.py) is emitted as multiple
  ordered, source-offset-backed spans (`part: "k/n"`). **What was found stays
  found (TASK-1557):** when the spans exceed the six-finding or parent-packet
  budget, the packet gives up the least material first, the last finding,
  then a span's middle parts. A shortened span keeps its head and its last
  part (where a trailing qualifier or reversal lives), and every kept part
  followed by omitted text is marked `truncated`. `uncertainty` counts the
  shortened spans and omitted findings; the handle expands them. When one
  part of one finding is left and still too long, its quote is clipped to a
  shorter exact prefix (same `start_char`, word boundary, marked `truncated`,
  floor `MIN_CLIPPED_QUOTE_CHARS` = 80), and at the floor the finding drops
  its `provenance` block, which the handle can still expand. The only case
  that clears findings is when not even that bare floor quote fits beside the
  coverage report; that packet is `insufficient_evidence`. At the minimum
  `output_tokens` (400) a bare floor quote fits beside all five sources'
  coverage, so it doesn't happen at any accepted budget today; a test pins
  that. Packet-level `truncated` still reports any clipping.
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
  An individual source timeout/failure is `unavailable` with a generic code;
  healthy sources may still yield cited findings, marked incomplete. A mixed
  unavailable/empty search is `insufficient_evidence`, never `no_match`; all
  sources unavailable is `error`. Policy, authority and source-contract failures
  remain fatal and return **no quotes**.
- `scope`: explicitly notes that memories are caller scoped, history spans only
  live searchable public partitions for the trusted user, tasks are searched by
  key fields (reviews and steps only with `task_details`), tool evidence is
  metadata, and `tool_args` yields attempted-clone facts, never command text.
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

A fully searched set of sources with no authorized matching evidence yields
`no_match`. Evidence that cannot support findings or a search with at least
one unavailable source yields `insufficient_evidence` unless another healthy
source supplies grounded findings. An all-source outage, bad model output and
policy errors are errors, never empty successes.
All failure codes are server-owned and omit source text/provider exceptions.
There is **no raw-result fallback**.

## Sources and authority

`SourceProvider` and `Synthesizer` are small ABCs. Orchestration, database reads,
HTTP task reads, bounded workers, continuations, token accounting and prompt
construction are focused modules under `llm_bawt/recall/`. Existing oversized
MCP storage/client modules are untouched; server.py only imports registration.

### Current DB policy

`RecallDatabase.scope` reads the current bot-tag snapshot (`bot_tags.SNAPSHOT_SQL`,
TASK-1329) directly, with a bounded SELECT, and validates it with `TagSnapshot`.
The searchable set is every bot that is neither `private` nor `no_global_search`
(private implies `no_global_search`; the legacy `include_in_global_search=false`
column still counts as private). Errors, absent/ambiguous/malformed snapshots,
sanitation collisions, and unknown callers fail closed. There is no hardcoded bot
list and no reuse of the existing fail-open aggregate-search helper.
**Bots outside the searchable set cannot use recall, even for their own history.**
A tag change is re-read on every request, including follow-ups. Public history
is selected through a positive map of the current searchable partitions;
private or excluded partitions cannot enter even the candidate stage.

### Messages

Read only the live `messages` parent via the positive public partition map;
join each partition to its canonical active/archived session for the trusted
caller user. A row is authorized only when it is a user-authored `role=user`
message for that user, a canonical assistant authored by its partition bot,
or a `role=user` delivery authored by a currently searchable bot. Excluded and
private bots can neither contribute their own partition nor author a delivery.
Deleted sessions, forgotten messages, summaries, unknown legacy authors,
other users and imported other-bot assistant rows are omitted. Memories and
tool evidence keep the stricter caller-user/own-bot rule.

The messages_find parser supplies literal identity terms (TASK-N, SHA prefixes,
paths and snake_case); recall discards dotted prose abbreviations, bare clock
times and ranges such as B1-B5 from that exact set. Numbered review queries
first search the indexed review marker and list endpoints (r9, B1 and B5),
requiring any explicit identity such as a task ID before capping candidates.
An empty focused search falls back to exact and lexical tiers.
Questions with one exact identity and enough topical words first intersect it
with the query's final three substantive terms via the indexed tsvector; a
miss falls back to relevance-ranked exact-all/exact-any candidates, then an
indexed high-coverage lexical tier. Broad OR runs only when stronger tiers
are empty. Within the same coverage bucket an authorized bot named in the
query can outrank newer recaps by other speakers; naming a bot never changes
which partitions or deliveries are authorized. A multi-word question needs at
least two lexical matches on the lexical tiers, even with a missing exact
identity; only an actual exact-tier match waives that floor. Each tier caps
authorized candidates before final ranking, so broad OR cannot force per-row
rank work across a whole history partition. Missing single-piece identifiers
do not degrade to generic old fragments. Coverage buckets (`COVERAGE_BUCKETS`)
rank the share of query lexemes matched, then full-text rank decays by a
14-day recency term.
This selects at most six **hits**. Each hit carries up to two **follow-ons**
(`MAX_FOLLOW_ONS_PER_HIT`): the next messages in the same partition and session
within 30 minutes, under the identical authority filters. Follow-ons reach
term-free corrections. Every evidence row carries its real source bot and
session as server-owned provenance, not a model-inferred review label.

SQL extracts an **original contiguous window** of at most 2,400 characters near
the whole query phrase, or the longest matching meaningful query term. No full
message body crosses the DB boundary. Stem-only matches without a literal anchor
use a prefix and report their count as `head_fallbacks` in that source's
coverage (present only when nonzero). Excerpts are then capped at
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
Unknown/private bot provenance or other-user associations deny the candidate.
System/cron and null human-actor activities are normal neutral shared-task events,
not authorization grants. Activity headers are attribution, not trusted authority.

What is admitted is what `tasks_get` would show the caller (TASK-1557), one
evidence record ("part") per passage:

- `tasks` (default): `TASK-N` (title, status, description),
  `TASK-N#response` and `TASK-N#outcome`.
- `task_details` (opt-in): `TASK-N#step:<n>` (step title and output) and
  `TASK-N#review:<cuid>` (one page of peer reviews, newest first). A review
  whose author the caller may not read is dropped and counted as withheld.
  More reviews than one page reports coverage `partial` (`reviews_truncated`);
  an unreadable review list keeps the task and its steps and reports
  `reviews_unavailable`.

Never project instructions, related-task content, or other bots' conversation
turns. Each part is an original window of at most 1,000 characters, opened where
the most query terms are, so an answer deep in a long response still reaches the
model; at most three parts per task. If a candidate's complete
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
turn. Approval status or a digest does not prove execution success.

### Tool args: attempted clones (TASK-1557)

`tool_args` answers "where did a bot clone X?" without letting command text
cross the boundary. It reads `Bash` calls under the **public-history** rule
(the call's trigger message must pass the same partition/delivery authority as
`messages`, for the trusted user), prefiltered in SQL to arguments containing
`git clone`. `recall/command_facts.py` then parses the command and the only
admitted text is a server-rendered fact:

    attempted clone repo=<url> dest=<absolute path when known>

The grammar is deliberately narrow. Segments are split only on `&&`, `;` and
newlines, and parsing **stops at the first unsafe token**: any other operator
(`|`, `||`, redirects, subshells), `$`, a backtick, a backslash, an unbalanced
quote, or an unrecognized flag. Facts come only from segments completed before
that stop, so a later `$(...)` in the same command can't leak and the clone
before it still counts. Only `cd <path>` and `git clone` with
`-q`/`--quiet`/`--single-branch`/`--depth N`/`-b|--branch NAME` are understood;
`pushd`, `popd`, `source`, `eval` and the like make the working directory
unknown. Repos must be `https://host/path` or `git@host:path` with no
userinfo, port, query, fragment or `..`; paths must match a plain charset, with
no `..` and nothing on the credential denylist (`.env`, `.ssh`, keys, …). A
fact says *attempted*: a recorded call is not proof the clone succeeded.

Ranking and selection use only the rendered facts, never the dropped text.
Drill-down re-projects the fact under the current authority. Facts about file
paths (`Read`/`Write`/`Edit`) are deferred: matching them over every tool
call needs an expression index first.

## Budgets and isolation

| Boundary | Ceiling |
|---|---:|
| Whole MCP request, including policy | 30 seconds |
| Concurrent research calls per process | 4; excess rejected, no queue |
| Source-provider calls | one per selected source |
| Task sources' own deadline | 4 seconds each; a timeout is `unavailable`, other sources' evidence kept |
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

**Known limits.** Grounding proves every quote is real source bytes; it does
not stop a model from choosing a misleading true excerpt, so selection quality
is not a security property. And if a packet's fixed parts ever outgrow the
requested budget so that not even a bare 80-character quote fits beside the
coverage report, the found evidence is withheld as `insufficient_evidence`
rather than returned unmarked.

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
