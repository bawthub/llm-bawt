# BawtHub MCP tool contract (TASK-888)

## Three levels of detail

1. **Catalog:** purpose, selection distinction, immediate side effects and
   call-critical constraints. Budget: at most 100 o200k_base description tokens
   per tool, including its skill pointer; 3,200 total for the current 87 tools.
2. **Typed input schema:** fields, requiredness, defaults, enum values, constraints.
   Never replace a typed input with an unstructured payload to make it shorter.
3. **On-demand skill:** `bawthub-mcp/SKILL.md` routes to four focused references.
   Workflows, examples, advanced combinations and recovery recipes live there.
   Skill pointers are relative to the installed skills root; harnesses without
   Skill invocation can read the files. Missing skills are reported, not worked
   around with direct API/DB calls.

Each tool remains usable for ordinary calls without reading the whole skill.
Destructive scope, full-replace semantics, approval boundaries, asynchronous
completion and surprising defaults belong in its immediate description.
No examples, deployment history, Args/Returns essays or repeated schema defaults
unless a default is an important hazard. Server implementation docstrings may
retain detailed developer documentation; they are not the model-facing catalog.

## Implementation pattern

- Tool modules still register typed callables on the shared registry.
- `CatalogFastMCP` extends the existing `ApprovalAwareFastMCP`, overriding only
  the public `list_tools` discovery seam. It does not override tool invocation,
  approval evaluation, trusted execution, ownership, return values or validation.
- `catalog_contracts.py` owns concise descriptions and skill-reference routing.
  Tests require exactly one entry for every registered tool; unknown tools fall
  back to their normal descriptions at runtime rather than disappear.
- The copied input schema removes generated `title` annotations only at schema
  nodes. A property NAMED title, enum/default/const/examples data, descriptions,
  required fields, nullable types, constraints, `$defs` and `$ref` are preserved.
  Validation still uses FastMCP's original schema/callable.
- Output schemas, annotations, metadata, names and order are unchanged.

Names and input fields are compatibility contracts. Do not rename/remove a tool
or merge calls until callers, saved approvals, policy selectors, prompt-registry
instructions, skills, and provider adapters have an explicit migration path.
Preserving old dispatch aliases while changing discovery is a possible later
migration, not authorization to silently hide existing tools now.

Keep read-only and mutating operations separate when designing new families.
Do not hide a delete operation behind a generic read-like name. A bounded typed
action enum can cover related mutations, but never use an arbitrary API path,
SQL string or unvalidated JSON blob as a universal escape hatch.

## Measurement and budgets

From the llm-bawt repository:

```sh
.venv/bin/python scripts/mcp_catalog_report.py --snapshot /tmp/mcp-local.json
.venv/bin/python scripts/mcp_catalog_report.py --url http://app:8001/mcp
.venv/bin/python scripts/mcp_catalog_report.py --baseline /tmp/mcp-before.json
.venv/bin/python -m pytest tests/test_mcp_catalog.py -q
```

The report counts compact UTF-8 JSON of namespaced name/description/input_schema
with o200k_base. This is a repeatable full-load footprint, NOT a provider usage
invoice. Results, skill contents, system instructions and provider framing are
excluded. A skill costs context only after loading it. A names-only discovery
index and a fully loaded tool catalog have different costs.

TASK-888 baseline: 87 tools, 19,070 tokens; descriptions 8,705, schemas 7,904.
First compatibility-preserving refactor: 10,250 total (-46.25%); descriptions
2,916 (-66.50%), schemas 5,693 (-27.97%). No functionality removed.
Regression budget: 11,400 total for this catalog. TASK-900 adds explicit paid
`x_search` with typed time bounds and pagination; the measured combined catalog
is 11,239 tokens (up approximately 160). The 3,200-description-token cap stays
unchanged. X is not included in implicit fan-out, preserving cost visibility.
The previous budget was 11,200. The home-audio addition adds five
tools (discovery, silent generation, enqueue, status, cancel), approximately 600
tokens; measured combined working-tree catalog: 93 tools / 11,081 tokens,
3,152 description tokens. This also includes the independently added initiative
tool. Read/mutate boundaries and audible/asynchronous warnings remain intact.
Increasing limits requires an explicit explanation with measurements; do not
delete warnings to meet a budget. TASK-1082 adds the optional short `key` field
to `sessions_rotate` for atomic queued `/new` retry identity: compact full-load
catalog measured 11,626 tokens (+8 from 11,618); the test cap is now 11,626.
The keyed rotation and its predecessor link commit in one session transaction.
TASK-1126 corrects task/step/project identifier wording from "UUID" (wrong:
BawtHub ids are cuids) to "ID (cuid)" and documents short `TASK-N` dependency
refs: compact full-load catalog measured 11,645 tokens (+19 from 11,626); the
test cap is now 11,645. TASK-1147 adds typed task review list/add tools with
separate read/write contracts, author attribution and retry guidance: measured
98 tools, 3,380 description tokens and 11,945 total tokens (+300 over 11,645).
The updated test caps are 3,380 description and 11,945 total tokens.
TASK-1188 adds `accept_pointer: bool = False` to `sessions_rotate` so accepted
new-session transitions can atomically stamp the bot pointer through the MCP
storage adapter. Measured full-load catalog: 11,957 total tokens (+12 schema
tokens); the total cap is 11,957, with the description cap unchanged. Keyed
replays do not re-stamp the pointer. No warnings were removed to meet the cap.

TASK-1210 adds typed `context_recall` (bounded read-only isolated evidence research)
and reconciles the missing description contract for the already-registered
TASK-1202 `bots_route_message`. The latter was a baseline catalog regression,
not a new routing feature. Measured with `scripts/mcp_catalog_report.py` on local
source: **100 tools, 3,482 description tokens, 7,039 input-schema tokens,
12,404 full-definition tokens**. Caps are now 3,482 / 12,404 (+102 / +447 over
3,380 / 11,957); the per-description cap remains 100. Recall contributes 250
full-definition tokens and retains authority, shared-task opt-in, metadata-only,
expiry and utility-model warnings. No existing warnings were removed. This is
local discovery, not live activation, and is not a provider billing measurement.
See `docs/ISOLATED_RECALL.md` for source/output budgets and failure contracts.

TASK-1239 makes `bots_deliveries_list` a bounded diagnostic list: compact rows
(`to_compact()`: ~15 scalar fields, clipped `preview`/`last_error`, no metadata),
default limit 10 (max 50), user chat queue rows and self-addressed
(sender == target) rows excluded unless `include_chat`,
plus `bot_id` (either direction), `active_only`, comma-separated `status`,
`task_id` and `since_minutes` filters applied in SQL before LIMIT. The HTTP
route keeps its full default shape (`view=full`, chat included) for BawtHub.
Measured against the live catalog snapshot: **3,514 description / 12,520
full-definition tokens** (+32 / +116 over 3,482 / 12,404); caps updated. The
growth is the filter schema that lets one call replace two unfiltered list
calls that measured ~10k and ~8.6k result tokens.

TASK-1233 adds `ops_quiesce_ack(status, note=None)`: a turn held by an ops
quiesce answers its steer with `stopping` or `cannot_stop`. Identity comes from
the signed per-turn capability, never from arguments. Measured: **3,545
description / 12,611 full-definition tokens** (+31 / +91 over 3,514 / 12,520);
caps updated. No existing warnings were removed.

The `ops_run` and `ops_job_status` results also carry `caller_instruction`:
`null`, or, while a quiescing job holds the calling bot's own scope,
`{action: "end_turn", held_bot, result_delivery, message}`. The caller's
running turn is in the hold roster, so polling in the same turn would keep the
job held until `quiesce_timeout`. `result_delivery` says truthfully how the
caller learns the outcome:

- `new_turn`: the terminal receipt arrives as a new turn. An approved job's
  continuation delivers it, and a direct job pushes it through its receipt
  callback (`docs/OPERATIONS_RUNNER.md` § Quiesce receipt callback).
- `none` (TASK-1233 (f)): a direct job holds the caller, but the caller's turn
  had no session to bind a callback to. Nothing is pushed; the message tells it
  to call `ops_job_status` with the job id in a later turn.

`message` is prompt-registry text resolved for the caller's bot
(`ops.quiesce_end_turn.new_turn`, `ops.quiesce_end_turn.no_receipt`). This is
a result field only; tool descriptions and schemas are unchanged.

## Provider loading caveat

`claude_code_bridge/proxy/translate.py::_tools_to_responses` forwards supplied
custom schemas to OpenAI Responses; it does not implement deferred tool loading.
`translate_cc.py` likewise converts supplied custom tools. Passthrough adapters
may support deferral (Anthropic) or inline schemas for unsupported backends.
The SDK may choose a subset before translation. Therefore inspecting catalog
size or a defer_loading flag alone cannot establish a turn's billed input.
Verify actual request-level metadata before claiming provider-side savings.
No provider-loading behavior is changed by this refactor.

## Activation and review

The MCP server runs inside the app. Source tests and local registration do not
prove live activation. Activate only through an authorized app operation, then
re-run the live report. Skills must be available on the target harness before
calling the skill-linked catalog operational. A resumed SDK may retain previously
loaded tool schemas; verify fresh discovery rather than deleting user sessions.
No restart, commit or deployment is implied by a catalog cleanup request.
