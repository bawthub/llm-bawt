# Explicit compaction summary policy (TASK-1071)

## Contract

The Claude bridge adds default guidance only to a bare `/compact`, including a
whitespace-only suffix. The target is **about 2,000 tokens**, not a hard output
cap. Preserve active goals, exact identifiers/paths, authorization constraints,
unfinished work, verification state/gaps, and durable receipt pointers. Nothing
chops or rewrites generated/persisted summaries.

Any non-whitespace instruction after `/compact` disables this default entirely.
The bridge forwards the complete original command unchanged, including multiline
and Unicode instructions and explicit requests for very long summaries. Ordinary
messages mentioning the command and similarly named commands are unchanged.
Existing provider/CLI physical limits remain in effect; no new environment,
system-prompt guidance, output-token cap, or threshold setting was added.

Registry seed/reset body (one line; editable in **Tools → Prompts** under
**Agent Compaction Summary Guidance**, key `agents.compaction_summary`):

> Write a concise continuity summary targeting about 2,000 tokens. Preserve active goals, exact identifiers and paths, authorization constraints, unfinished work and next steps, verification results and remaining gaps, and pointers to durable task receipts. Prefer current state over chronological retelling; omit repeated discussion and superseded details. Keep essential continuity even if it needs extra space.

`send_preflight.prepare_send` checks for a bare command after `/new` preprocessing,
fetches the current global prompt through the app registry, and then constructs
the SDK prompt. It never caches it across compactions; edits take effect in the
same resumed conversation without session restart. `SendRequest.message`
remains the original input; attribution/history are not rewritten. The bridge
contains only the registry key, NOT a hardcoded fallback copy of this text.

## Entry points

- **Explicit chat command:** shared bridge preflight applies the policy.
- **MCP `agent_context_compact`:** existing tool emits an idle durable `/compact`
  with `message_kind=CONTEXT_MAINTENANCE` and `metadata.context_action=compact`.
  The HTTP delivery route now preserves that raw command instead of prefixing
  `Message from bot ...`, which previously turned it into ordinary prose. Author
  metadata remains on the durable delivery. This exception requires an idle
  Claude backend and an actual `/compact` command; normal callbacks keep their
  existing attribution wrapper. Tagged caller-directed commands also pass
  unchanged. The MCP tool itself exposes no custom-instructions parameter.
- **Automatic CLI compaction (follow-up source, not activated):** the bridge's
  startup bootstrap adds a CLI `PreCompact` **command hook** to its existing
  settings without replacing other hooks. It fetches the same global registry
  prompt on each trigger and prints it only for `trigger=auto|manual` with
  null/empty/whitespace `custom_instructions`; a non-whitespace caller instruction
  yields no stdout or registry lookup. Existing explicit-chat preflight populates
  bare `/compact` first, so the hook sees a nonempty suffix and cannot duplicate
  it. Prompt edits apply on the next auto compact even in resumed SDK sessions,
  without changing the cached system prompt, model or threshold. The hook cannot
  affect running bridge processes until an approved restart; standalone tests
  use only disposable settings and sessions. Installed SDK Python callbacks do
  not support instruction replacement; this is CLI *command-hook stdout*.
- **Commands steered into an already-running SDK:** TASK-1071 policy acts on
  `chat.send` preflight, not directly on CLI command injection. Separate
  TASK-1072 routes a steered `/compact` as its own durable turn, so preflight can
  apply on that later send; this is not automatic CLI compaction.

## Editable registry lookup and outage behavior

Both paths GET `LLM_BAWT_API_URL/v1/prompts/agents.compaction_summary` on EACH
eligible compaction, using the app-owned DB-backed registry; the bridge does
not access the database directly or cache this body. Tools → Prompts already
lists `GET /v1/prompts?include_defaults=true`, edits via `PUT /v1/prompts/{key}`
and offers version history/reset. `updated_at` + a SHA-256 body fingerprint,
key and source are logged on selection (never the prompt body or conversation).
Source may be a saved DB override or a registry seed/reset default. A saved
edit needs **no** restart for subsequent compactions once source is activated.

The registry GET uses a 1.5s socket timeout; explicit preflight has a 2s
wall-clock deadline and the CLI hook has its existing 10s command timeout.
If lookup fails, is oversized/invalid, or the app DB is unavailable, it logs
and adds **no** new guidance; the CLI still compacts under its own behavior.
For this key, the app returns 503 instead of pretending the seed is current
when its DB is unavailable. No stale/hardcoded bridge fallback can silently
override an edited prompt. Custom suffixes skip lookup before any network call.

## Maintenance isolation is separate: TASK-1072

TASK-1072 contains the later queue/isolation implementation and its own scoped
receipt in [COMPACTION_MAINTENANCE.md](COMPACTION_MAINTENANCE.md). TASK-1071's
policy review and measurements must not be mistaken for verification or
activation of maintenance isolation; see that receipt for supported entry points,
publication fencing, and remaining PostgreSQL/live limitations.

## Controlled installed-CLI evidence (2026-10-13)

CLI **2.1.283**, driven by test SDK **0.2.158**, used a fresh HOME,
CLAUDE_CONFIG_DIR, cwd and synthetic transcript for each case. No live bot or
protected-bot transcript was resumed, compacted or edited. Tools, MCP servers and
settings sources were disabled. The existing proxy and app-owned credential
broker made real provider requests; no production service was restarted.

Fixture: 40 repeated development handoffs, 45,100 pre-compaction tokens reported
by the CLI. Model explicitly selected for the probe: `openai_chatgpt/gpt-6-astra`.
The same seed text was independently compacted in each case.

| Case | Persisted summary characters | CLI post-compaction resident tokens | Wall time |
|---|---:|---:|---:|
| Previous bare command (policy bypassed) | 10,200 | 3,114 | 65.094 s |
| New bare-command default | 9,818 | 3,019 | 75.931 s |
| Custom request for at least 3,500 tokens | 23,992 | 6,562 | 160.094 s |

The default was **3.7% smaller by persisted summary characters** and **16.6%
slower** in this single sample. Meaningful summary-bloat reduction and faster
completion acceptance remain **inconclusive/unmet**, not fully verified. This
experiment verifies policy routing, custom bypass, and fixture continuity; it is
not a representative reproduction of the earlier bloated transcript. These are
historical PRE-HOOK measurements; they are not invalidated retroactively. The
current helper's original `policy=False` baseline was briefly contaminated by
its new hook settings; the isolated fake-upstream negative-baseline regression
now disables BOTH the preflight guidance and the command hook, and a distinct
hook-only bare test verifies that path. No paid rerun of the benchmark was made.
These character counts include CLI
continuation framing; `postTokens` includes retained context. Actual persisted
summary-only token count and generated-output-token count were **not measured**.
The instruction is a soft target, not proof of a 2,000-token ceiling.

Continuity checks retained TASK-4242, `src/export/naïve.py`, `export_7f3a`, the
no-commit/push/restart/deploy constraint, timeout regression and duplicate
callback work, 17 passing unit tests, explicitly unverified integration,
`docs/receipts/TASK-4242.md`, and blocked TASK-4243. Manual inspection of the default
summary confirmed these facts. The long custom summary also retained the explicit
`CUSTOM-LONG-OVERRIDE` marker. It was substantially longer than the default.

Temporary full measurement output:
`/tmp/pytest-of-bridge/pytest-3/test_real_provider_summary_siz0/compact-measurements.json`.
This document is the durable measurement receipt; temporary files may expire.

Local fake-provider tests additionally drove the installed CLI with bare and
multiline custom commands. They inspected actual summary requests and persisted
an intentionally oversized (>50k-character) response intact, proving no blind
truncation. A provider 400 produced a failed compact status with no successful
boundary/summary. These protocol tests are not evidence of model quality or
latency.

**Additional read-only command-hook evidence (2026-10-13):** Snark's isolated
`/tmp/snark-precompact-hook-probe.py` was reproduced locally with CLI 2.1.283,
runtime SDK 0.2.160, disposable HOME/config/session, local fake provider, no
paid requests or live settings. It configures a per-probe `PreCompact` command
hook through `ClaudeAgentOptions.settings` with all other setting sources disabled.
The hook logs the input, emits a marker only when `custom_instructions` is empty,
and checks for the marker in the upstream summary request. Results:

| Trigger | Hook input custom instructions | Marker in request | Persisted boundary | Result errors |
|---|---|---|---:|---|
| Auto (fake usage-triggered) | null | yes | 1 | none |
| Manual bare | null | yes | 1 | none |
| Manual custom | exact `CUSTOM_KEEP_EVERYTHING_82` | no | 1 | none |

Reproduce (read-only isolated probe; temporary script may not persist):

```sh
PYTHONPATH=/app/.venv/lib/python3.12/site-packages:/home/bridge/dev/llm-bawt/src \
  /home/bridge/dev/llm-bawt/.venv/bin/python /tmp/snark-precompact-hook-probe.py
```

The earlier read-only probe established the mechanism. The follow-up now tests
both policies together with the installed CLI: fake-upstream manual bare and
custom cases include actual hook settings and SDK callback; bare contains the
registry response **exactly once** in the provider request, while multiline
Unicode custom instructions do not contain it or fetch the prompt. Baseline
with BOTH paths disabled contains zero guidance; hook-only bare contains one.
A fake-usage-triggered auto case verifies a compact boundary and an edited
registry body in exactly one provider request (not the seed text). The separate
isolated registry/API test saves two versions via the same endpoints the editor
uses and verifies both paths see each edit without reinitializing the bridge,
plus custom bypass, 503 and timeout behavior. These are protocol tests, not a
live deployment or proof of smaller/faster generated summaries.

## Reproduce

Focused hermetic + isolated editor/API coverage:

```sh
.venv/bin/python -m pytest -q tests/test_compact_registry.py tests/test_compact_hook.py tests/test_claude_bridge_bootstrap.py tests/test_compact_policy.py tests/test_compact_delivery.py
```

Installed CLI with fake upstream (6 passed; no provider credentials/traffic):

```sh
RUN_COMPACT_SDK_PROBE=1 COMPACT_PROBE_CLI=/app/.venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude \
  .venv/bin/python -m pytest -q tests/test_compact_sdk.py -k 'not real_provider'
```

`COMPACT_PROBE_CLI` may explicitly select the installed runtime binary when the
test venv bundles a different version. A *separately authorized* real-provider
probe would also require `COMPACT_PROBE_MODEL` (explicit selected model) AND
`LLM_BAWT_API_URL` (registry endpoint); this follow-up ran none. Results depend
on provider sampling/load and must not be presented as a benchmark guarantee.

## Activation

Source and tests only. The new hook is added to persistent Claude settings **at
bridge startup**, not by editing the live settings during tests. Existing hook
groups are preserved, and repeated bootstrap is idempotent. App route changes
and bridge policy need their respective process reloads before deployment can
be claimed. A bridge restart requires Nick's approval for that exact run and
managed ops. No commit, push, restart or production deployment was performed
for this implementation. The existing global compaction threshold (250000),
selected models and provider caching are unchanged. The compaction prompt is
now an editable global registry entry with seed/reset text only in the app;
subsequent edits use the live registry without a bridge/session restart once
the new source is activated. Do not treat the new source as live until the app
and bridge are reloaded with Nick's exact-run bridge approval.
