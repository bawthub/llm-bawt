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
