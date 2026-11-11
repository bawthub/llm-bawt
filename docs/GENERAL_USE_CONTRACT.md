# General-use contract guard

`make contract-guard` scans tracked text in `src/`, `docs/`, `skills/`,
`tenant-skills/`, `prompts/`, `seeds/`, Compose YAML files at any depth, and
the legacy routing fixture named in the scanner. Untracked files and binary
assets are not source text. Tracked symlink targets are scanned as link text,
never followed outside the checkout.

`make gate REV=...` invokes `test-all` in its exact-revision snapshot;
`test-all` requires this guard before running the offline suites. It is not
a scan of the author's unrelated working tree. No live database or service
is contacted by the guard.

## Private inputs

The built-in rules cover RFC1918 IPv4 literals, literal installation home/data
paths and literal Compose `container_name` values. Deployment/operator names,
personal domains, container identifiers and allowed live bot slugs come only
from a **private, untracked** JSON file or environment value. They are not
hardcoded in the scanner, baseline, fixtures or this document.

Prefer **one shared `GENERAL_USE_DENYLIST_FILE`** across repositories, hosts
and houses. `--denylist-file` overrides it. Alternatively use
`GENERAL_USE_DENYLIST_JSON`; a relative key path is then relative to the Git
common directory. With neither override, the default is
`general-use-denylist.json` in the Git common directory, so a checkout and its
snapshot worktrees use the same inputs.

Private JSON schema (the example literals below are deliberately empty):

```json
{
  "version": 1,
  "mode": "private",
  "hmac_key_file": "general-use-hmac.key",
  "operator_identities": [],
  "personal_domains": [],
  "container_names": [],
  "bot_slugs": []
}
```

Fill the four lists from the installation's authorized inventory; populate
`bot_slugs` from its allowed live profile records and refresh after inventory
changes. This is an offline exported policy, not a DB query during a gate.
Do not query protected profiles. Relative key paths are relative to the
private JSON file. Generate a 32-byte random hex key into an untracked file
with restrictive permissions; never print the key or put it in commands,
repository files or public logs. Share the same policy **and key** wherever
the private gate runs. Both tracked denylist files and tracked keys are refused.
Missing or malformed private configuration exits 2.

Private literal fingerprints are HMAC-SHA256 truncated to 128 bits. The key
prevents dictionary guessing of names absent from source, but per-file count
patterns can reveal which existing source tokens are classified as private.
The baseline and diagnostics contain only file/rule, opaque fingerprint and
count, never matched text. Public built-in-rule fingerprints are truncated
SHA-256.

Public/fork CI deliberately sets `mode: "public"`, with all four lists empty
and `hmac_key_file: ""`. That explicit mode checks built-in rules but cannot
check deployment-specific names. It can leave private baseline entries
inactive; a private gate **fails** on inactive entries, including policy/key
drift. Public CI is not a replacement for the private prepublication gate.

## Ratchet and removals

`general-use-baseline.json` records the initial inventory, not an exemption
for whole files. Each `(file, rule, literal fingerprint)` has an occurrence
count and inherits a reason and removal task from its rule metadata. Every
hit therefore has both pieces of accountability without repeating the same
prose hundreds of times. Initial cleanup belongs to **TASK-2193**, not to the
guard's implementation task. Public tool names can collide with private bot
slugs; those existing counts require triage, not a blanket exclusion of the
tool name. TASK-2194 completed that triage: deployment bot references were
removed, and slugs that are guaranteed collisions (a public harness/provider
name, an ordinary English word) were dropped from the private list rather than
exempted in the scanner. Keep such slugs out of `bot_slugs`; their false
positives cost more than the leak they could catch. First-party origins are not automatically classified as personal
domains: any future exclusion needs an explicit reason.

New literals, new files and net count increases fail. Reduced or removed
counts also fail as `STALE`: trim the corresponding baseline counts in the
same change. Moving an existing occurrence within the same file, or a
count-neutral replacement with the same literal, is intentionally not detected.
The contract is a maintenance ratchet, not a semantic or security scanner.

`python3 scripts/general_use_contract.py --inventory` prints redacted
candidates with empty reason/task metadata. It never approves or updates the
baseline. Review additions; do not blindly accept an inventory dump. A
private-policy/key rotation requires deliberately rebuilding the private
fingerprints and reviewing the resulting baseline.

The legacy routing fixture is explicitly included so its previously noted
personal-domain occurrence is inventoried even though tests are otherwise
outside this product-source scan. The removal task should replace that fixture
with a reserved documentation domain where applicable.

## BawtHub finding and shared mechanism

BawtHub's current gate checks build/test contracts but has no corresponding
deployment/operator-literal scan. This change does not add a second scanner
there. A later BawtHub guard should consume the **same private JSON file and
HMAC key**, and adapt the tracked roots to its frontend/server/docs/Compose
surfaces. Do not duplicate private literal lists in either repository.
