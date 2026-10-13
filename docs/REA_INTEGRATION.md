# REA integration — TASK-1060

## State (2026-10-13 UTC)

**Installed, published, activated and verified through Snark's actual BawtHub
turn.** One separately approved Claude bridge restart completed. No production
frontend release, bot profile change, or session reset was performed.

Activation job `9f468cf8d8dd47b9ae4b5763c41b00cf` succeeded, exit 0, at
2026-10-13 00:23:13 UTC. Approval:
`mcp-appr-622ede840902c121c4cbfb93603419d50ee2a114`.
Post-restart bridge health returned `{"redis": true}`. REA deferred tools were
discovered and `mcp__rea__analyze_javascript_application` called directly from
Snark's resumed chat with the fixture path and `detail: "summary"`. It returned
summary Evidence `ev_a61a0a90412a631dd7dfa79696478de0f17cc404bcce0d9c00790a10831a8d4c`
and the same complete parent Evidence recorded below. Two JavaScript files
parsed, zero parse failures; fixture bytes still match commit `becfb83`, and no
execution sentinel exists. Installed release selection and registration survived
the restart. The imported skill loaded through the actual Skill tool.

The result reports complete application-graph coverage but **unknown semantic
coverage with nine unknowns**. Static import recovery is verified, not arbitrary
runtime execution or all 170 advertised capabilities.

Skill commit `030306f` was pushed and synchronized to echo and the Ubuntu sibling.
Only `mcpServers.rea` was added; all other settings compared equal before/after.
Mode-0600 backup:
`/home/bridge/.claude/tools/rea/backups/settings-before-rea-20261013T002043940808Z.json`.
Settings remain mode 0600 and `ENABLE_TOOL_SEARCH=true`. Al and Loopy confirmed
safe boundaries; Gavel approved the TASK-1060-only commit split. Their unrelated
TASK-1066/1067 changes remain untouched and uncommitted by this task.

REA is an MCP extension, not a BawtHub frontend dependency. Official upstream:
<https://github.com/morluto/rea>, <https://rea.tools/>, npm `rea-agents` **6.4.0**.
The live Node prerequisite (TASK-1061/1062) is satisfied: Node **24.21.0** and
npm **11.19.0** at `/usr/local/bin/{node,npm}`. Do not install a private runtime.

## Provenance and installation

- Tarball: `https://registry.npmjs.org/rea-agents/-/rea-agents-6.4.0.tgz`
- SHA-512 integrity:
  `sha512-Vj1zWfSlna5bphGoaoI/TcVBuYvze9u+OyPC3Shz8YkBqaCTqB4COnnRoX7XDwMpwI24ZlFEel2yEj0k8LTWEQ==`
- `scripts/rea/package-lock.json` pins resolved registry URLs and integrity for
  all dependencies (150 package records, including the root).
- Tarball integrity was independently checked. The dispatcher and upstream
  `prepare` script were reviewed; no lock entry declares an install lifecycle
  script. Installation always disables lifecycle scripts anyway. This is not
  a complete security audit of REA or its dependency graph.
- Persistent root: `/home/bridge/.claude/tools/rea` maps beneath echo's
  `/home/nick/.config/claude-code-bridge/tools/rea`.
- Selected release:
  `releases/6.4.0-7bd2a2eaa0b02994125c0f9faf6d22285b3f9b5b6045dbe32eb6fd568dd17843`.
  The suffix hashes manifest + NUL + lockfile. `receipt.json` also records the
  lock-only digest, package integrity, Node/npm versions and install timestamp.

Run inside the Claude bridge:

```sh
/home/bridge/dev/llm-bawt/scripts/rea/install.sh
/home/bridge/dev/llm-bawt/scripts/rea/rea --help
```

The shell installer delegates filesystem/locking operations to a small Python
stdlib `Installer` class. It validates the existing Node engine before creating
state, refuses concurrent installs with `flock`, runs frozen `npm ci
--ignore-scripts`, smoke-tests staged CLI help, and atomically selects a release.
Existing release directories are never rewritten. A failed install or smoke
leaves the current selection intact. An identical install rechecks help and
returns without changing the pointer. `REA_INSTALL_ROOT` (or installer `--root`)
permits isolated tests. The launcher resolves `current` once, then `exec`s REA;
no startup fetches, setup wizard, Node download, or service action.

## Registration and activation

After shared skill publication, atomically merge **only** this entry into
`/home/bridge/.claude/settings.json`, backing up the original with mode 0600 and
refusing a conflicting existing `rea` value:

```json
{
  "mcpServers": {
    "rea": {
      "type": "stdio",
      "command": "/home/bridge/dev/llm-bawt/scripts/rea/rea",
      "args": ["mcp"]
    }
  }
}
```

That snippet is a merge, **not a replacement settings file**. Preserve other
servers, hooks, permission rules, environment and attribution. Never print the
whole secret-bearing settings. Keep `ENABLE_TOOL_SEARCH=true`; do not inject
170 tool schemas into every turn. Load the harness configuration skill before
editing settings.

Bridge startup caches MCP registrations. Settings changes require one
**separately authorized** Claude bridge restart through the enabled ops action.
No Redis/whole-stack restart, dependency rebuild, or frontend release is needed.
Save the task handoff before dispatch; verify terminal job state and health after
reconnection, then discover and invoke REA from Snark's actual BawtHub turn. A
standalone MCP smoke does not prove the bridge is active. This registration is
bridge-wide, not a per-bot access-control mechanism; no protected bot is used to
verify it.

## Matching skill

Import `skills/reverse-engineer-anything` from this exact installed package, not
upstream main, with its six reference files and package MIT `LICENSE`. The skill
has metadata version **37** (different from npm version 6.4.0), tool_count **170**.
Add `references/bawthub-installation.md` for the persistent launcher, local gates
and limitations, with one link from SKILL.md. Publish in the same turn using the
shared skill workflow, after commit/push authorization; do not strand a live
unpublished edit. Do not replace the existing skills symlink.

Ambient skill installation does not extend explicit skill bundles. Do not
change bot profiles or reset sessions just to force visibility. Ordinary source
repository architecture analysis uses repository tools, not REA.

## Verification receipts

- `uv run pytest tests/test_rea_install.py tests/test_claude_bridge_bootstrap.py -q`
  on echo: **23 passed**. The bridge Python environment lacks pytest, so tests
  use the existing host environment rather than adding dependencies.
- `shellcheck scripts/rea/{install.sh,rea}` and shell syntax checks: passed.
- Focused Ruff and `git diff --check`: passed.
- Offline tests cover engine boundaries, missing runtime, failure rollback,
  locking, idempotency, existing-release preservation, spaces/argument forwarding,
  exit status, protocol-clean failure stdout and bootstrap preservation.
- Bootstrap retains custom REA. Existing bootstrap intentionally normalizes
  built-in server configuration and blanks harness commit attribution; the test
  starts with those canonical settings, not a promise to change that behavior.
- Live CLI and MCP both verified `index.js` imports `describeFeature` from
  `feature.js`; two JS files parsed, zero parse failures.
- Both yielded Evidence:
  `ev_9df7ccb9d889873a00b933a8eac980b95272cbd7dae2cb6225935b4aa12f79e1`.
- MCP initialized as `rea` **6.4.0**, advertised **170 tools**, successfully served
  summary and complete analysis, and returned an explicit `invalid_input` error
  for a missing path. Client closed; no REA launcher process remained.
- Fixture files were hashed before/after: unchanged, no `EXECUTED` sentinel.
- Second install selected the same release without reinstalling dependencies.

Repeat the opt-in live smoke with the bridge's existing MCP Python environment:

```sh
/app/.venv/bin/python /home/bridge/dev/llm-bawt/scripts/rea/smoke.py
```

Normal pytest never downloads packages, starts live MCP services, or mutates
real settings. The smoke explicitly runs only the self-owned static fixture.
Static relationships are not proof of runtime calls: this fixture's imported
call remains subject to the analyzer's cross-module/semantic limitations.

## Update and rollback

For an update, review the new exact package, engine, lockfile and lifecycle
changes first; update installer provenance constants as well as manifest/lock.
Run offline tests and smoke the staged candidate before registration/activation.
Keep old release directories while any clients may still be using them.

To roll back a package, create a temporary symlink to an existing verified
`releases/<id>` and atomically replace `current` using `os.replace`. Do not edit
or delete an in-use release. New launches use that selection; existing clients
continue against their resolved release until their authorized reconnect.
The first installation has no previous release, so disabling registration is
its rollback rather than selecting an invented earlier package.

To remove registration, delete **only** `mcpServers.rea` if it still exactly
matches the installed entry. If it changed, stop for review. Preserve all other
current settings; never restore an old whole-settings backup over subsequent
work. Activation of that removal likewise needs exact-run bridge authorization.
Remove a shared skill only through its authorized publish workflow. Retaining
unselected package files is safer than deleting files beneath clients.

## Boundaries

Static JS analysis needs no REA API key. Native decompilation needs separately
provided Ghidra/Java or licensed Hopper/IDA; browser/device/runtime capture needs
its own environment and target authorization. Do not install those to make a
broad `doctor` check green. No native/provider/device capability is claimed here.

Package-local installation is **not a sandbox**: REA inherits bridge filesystem
and network privileges. Local analysis Evidence enters the conversation/model
provider, so this is not a claim of fully offline/private processing. Avoid
credential-bearing targets. Use separate task-scoped exports and session handles
for concurrent investigations.
