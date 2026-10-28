# Compose configuration drift guard and same-image reconcile (TASK-1518)

## Status

**Code only; not activated.** The guard and the operation need a rebuilt ops
worker image and a catalog update (see [Activation](#activation-order)). A
running app keeps its earlier behaviour until it restarts on this code. From that
restart until activation steps 2–4 are done, production deploys are **refused**
(the live deploy row has no `compose_config`, and an unrebuilt worker lacks the
capability label). Treat any app restart as the start of activation.

## Problem

`deploy_image` clones the running container forward with a new image. Anything
the container was last *created* with (environment, labels, mounts, networks,
restart policy) is carried as-is. If the Compose files changed since then, every
deploy silently keeps the stale configuration, and nothing reports it.

## The Compose renderer

Neither the app image nor the stdlib-only worker ships the Compose CLI or the
project files. Both ask the Docker daemon to run a **digest-pinned Docker CLI
image** (`compose_config.renderer_image`) as a sealed one-shot container
(`ops/compose_render.py`):

- no network, no Docker socket, read-only root filesystem, all capabilities
  dropped, `no-new-privileges`, memory and PID limits;
- log driver `none`, so rendered values never reach daemon logs. Output is read
  once through `attach`, and the container is removed in every outcome;
- exactly one read-only bind: the target's own Compose working directory at the
  same path, so relative paths and the project `.env` resolve as they do for the
  operator. Directory and files come from the target container's
  `com.docker.compose.project.working_dir`/`config_files` labels and must equal
  the spec's `compose_files` basenames in order. No host path lives in a seed;
- fixed argv (`compose -p <project> -f … config --format json <service>`, then
  `config --hash <service>`) and one injected variable: the pinned image
  reference, under `compose_config.image_variable`.

Renderer stderr is never echoed. An unset-variable warning yields
`unset_compose_variable` with the variable **names**; any other stderr yields
`renderer_warning`; a non-zero exit yields `renderer_failed`. An unset variable
fails closed: the operator's shell may have supplied it when the target was
created, and the renderer cannot reproduce that shell. Put such variables in the
project `.env`.

The renderer image is public; the app pulls it anonymously in preflight when it
is absent (`renderer_image_missing` if that fails). Workers never pull.

## What is compared

`ops/compose_config.py` translates the rendered service for an **allowlist** of
keys: `command`/`entrypoint` (must be null), `environment`, `extra_hosts`,
`image` (must equal the pinned ref), `labels`, `networks` (no per-network
settings), `pull_policy` (`missing`/`never`), `restart` and `volumes` (bind or
named volume, `read_only`, `bind.create_host_path`). Any other key refuses with
`unsupported_compose_key: <names>`, so an operation can never apply a privilege
or setting it does not model. Unmodelled fields (ports, devices, user, …) are
carried from the running container exactly as a plain deploy would.

Both sides are projected onto environment, labels, mounts, networks, extra hosts
and restart policy. Values the image supplies, and Compose/ops bookkeeping labels
(`com.docker.compose.*`, `llm-bawt.ops.*`), are dropped on both sides. Drift is
reported as names only: `env:+KEY`, `env:-KEY`, `env:~KEY`, `label:…`,
`mount:±/path`, `network:±name`, `extra_hosts:~`, `restart:~`. Values never
appear in errors, approvals, receipts or job output.

## Deploy guard

`deploy_image` specs now require `compose_config` (`validate_image_spec`);
`rollback_image` accepts but never checks it, so a rollback stays possible under
drift.

- **Approval** (`OpsService.prepare_invocation`): the deploy is checked at the
  verified release's `repo@digest`. Drift refuses with `config_drift` and the
  names; no approval card is shown.
- **Release approval** (`ReleaseExecutor.approval_source`): checked at the
  running ref, since the new digest does not exist yet. A release whose deploy
  would refuse is never approved.
- **Release deploy step**: if drift appears after the release was authorized,
  the deploy prepare fails, the coordinator records `deploy_unverifiable`, and
  after `DEPLOY_PREP_WINDOW` (30 min) declines the release.
- **Worker** (authoritative, under the per-target lock): re-renders at the new
  ref and refuses (`config_drift`, known outcome, nothing changed) before
  creating anything. On success the clone carries
  `com.docker.compose.config-hash` equal to Compose's own hash, so a later
  `compose up` does not recreate the target.
- **Preflight** checks only cheap prerequisites (worker capability, renderer
  image); it never renders, so a queued job does not churn renderer containers.

## `bawthub.reconcile-prod-config`

A disabled-by-default Docker operation with **no arguments**
(`reconcile_frontend_config`, `ops/config_reconcile.py`). It recreates the
target from its rendered Compose service at the **same pinned image**: no build,
no pull.

The approval binds the container ID, image ID and ref, the Compose source, a
fingerprint of everything a clone would carry (`current_config_hash`), a hash of
the rendered desired configuration (`desired_hash`, values hashed only), the
rendered Compose hash, drift names, the mount list after reconcile, the baked
release identity, and `display_values`. A display value is shown only on an exact
match of the spec's closed enum (`<unset>` or `<unexpected value>` otherwise).
The approval card (`approval_text`) shows names, mounts and those enum values.

The worker re-renders and recomputes the whole binding under the per-target lock.
Any difference refuses with `config_changed_since_approval (<fields>)` and
nothing changes. A stopped or unhealthy target is refused. With drift it swaps
through the shared deploy machinery (Docker health, release identity, restore on
failure; a failed restore is an unknown outcome). Without drift it verifies only.
After success it reports mode, uid and gid (never contents) of each `stat_paths`
entry. Receipts use `llm-bawt.ops.config-reconcile/v1`.

## Worker capability

The worker image must carry `llm-bawt.ops.config-reconcile="1"`
(`docker/Dockerfile.ops-worker`). Without it the app refuses both the reconcile
operation and every deploy with `config_reconcile_unavailable`, at approval and
in preflight: an older worker would deploy without the guard.

## Activation order

Each step needs Nick's authorization; none is a catalog action. **Do not touch
bridges or Redis.**

1. Make sure every variable the Compose files interpolate is set in the project
   `.env` (the renderer refuses unset variables by name).
2. Build `docker/Dockerfile.ops-worker`, confirm the label, and repin
   `LLM_BAWT_OPS_WORKER_IMAGE` for app and ops-reconciler.
3. Recreate app and ops-reconciler.
4. PATCH the live `bawthub.deploy-prod-image` row to add `compose_config` (seeds
   never update existing rows). Between steps 3 and 4 the deploy is refused, and
   the old app rejects the key, so keep this order and keep the gap short.
5. Create `bawthub.reconcile-prod-config` disabled through the operator API,
   install and preview its `require_approval` policy, reload policies, and enable
   it only with approval. Run it through `ops_run` if a deploy reports drift.

## Cutover service plan (TASK-1476)

**Code only; not activated.** One approval-gated job that recreates several
llm-bawt services in one window, each at its **same image**. Recreating them one
at a time would leave the others reading configuration the first no longer
serves.

### Target kinds

- **App kind** (`reconcile_app_config`, `ops/app_config_reconcile.py`): a
  Compose-built service with a bind-mounted source checkout. The checkout must be
  exactly the published branch head (source guard). The image is bound by ID,
  never by tag. Rows: `llm-bawt.reconcile-app-config`,
  `llm-bawt.reconcile-codex-bridge-config` and
  `llm-bawt.reconcile-claude-code-bridge-config`. For a bridge, the source guard
  binds only the bridge's own source. The agents it hosts run code from the
  mounted workspace by design, so that workspace is not guarded.
- **Tagged kind** (`reconcile_tagged_config`, `ops/tagged_config_reconcile.py`):
  a service that runs a public image by moving tag (Crawl4AI,
  `llm-bawt.reconcile-crawl4ai-config`). It binds and swaps to the **running
  image ID** and never re-resolves the tag, so a reconcile is never an upgrade. A
  tag that now names another local image shows on the card as `image_tag_moved`
  and is not drift. It renders with the pinned process profile
  (`container_name`, `depends_on`, `entrypoint`/`command`, published ports,
  `tmpfs` added to the keys above) and inherits the image's own healthcheck.

Both kinds check `depends_on` as a precondition and verify the target's own
in-container health URL. A failed verify restores the previous container.

### The plan (`llm-bawt.cutover-plan`, `ops/plan_reconcile.py`)

The spec (`reconcile_service_plan`) is an ordered list of existing catalog
rows plus a closed probe list. The rows stay the allowlist: their env names and
data mounts remain DB-owned, and each target runs through its own reconciler
(bind, compare, source guard, dependency check, swap, verify, restore). Nothing
is forked. Seed order: app, Crawl4AI, codex bridge, claude-code bridge.

- **Approval** (`ops/plan_approval.py` `bind_plan`): every target must be an
  **enabled**, argument-free row on the same executor (`plan_target_unavailable`
  or `plan_target_invalid` otherwise). Each one is bound exactly as for its own
  approval. The app's `records` probe and its env **names**, now and as
  declared, set the credential classes. The card shows each target's own card
  plus the classes. Values never leave the containers.
- **Credential classes:** `migrated` (stored record connected; its probe must
  pass afterwards), `env_kept`, `unconfigured`, `accepted_loss`. A credential the
  declared app would drop with no stored record refuses with
  `credential_without_record:<destination>` unless the invocation names it in
  `accept_loss` (a closed enum of `env_import` destinations). A dropped
  non-secret name in the same destination also counts as a loss, which is
  conservative.
- **Worker order**, under every target's lock (taken in plan order, never
  waiting):
  1. Read-only preflight of every target, the credential re-check and the volume
     check. Changed records, env names or classes refuse with
     `*_changed_since_approval`, and nothing changes.
  2. Missing project volumes are created with Compose's labels (the first
     write), so a later `compose up` adopts them. A volume that another project
     or volume key owns refuses.
  3. Each target in order: the last-read recheck, then verify-only (no drift) or
     swap. The previous container stays stopped as `<name>-prev-<job8>`. A later
     target's dependency check ignores those `-prev` containers.
  4. End checks: the new app reports the same records and carries exactly the
     approved credential env names, and every plan probe passes. A probe line
     that does not parse is `probe_unverifiable`, never a pass.
  5. Commit: every `-prev` container is deleted.
- **Revert all:** any failure at or after step 2 reverts every swapped target in
  reverse order. It continues past a failed revert, and that makes the outcome
  unknown. Only a full revert removes the volumes this job created. If any revert
  failed, the volumes stay.
- **Probes** (`ops/probe_contract.py`, standard library only): a fixed-argv exec
  in the target prints one closed JSON line (`probe`, `ok`, `status`, `code`;
  the `records` probe adds booleans only). The app runs
  `python -m llm_bawt.service.cutover_probe <id>`; the claude-code bridge runs
  `python -m claude_code_bridge.cutover_probe broker`.
- **Card history:** plan runs do not feed `_last_classification`, so a target's
  own card still shows FIRST RUN after a plan has run.
- The plan ships disabled, critical and with quiesce off. It KILLS every agent
  session (the claude-code bridge hosts the agents) and cuts in-flight chat
  turns, so it runs only in a window Nick schedules.

### Worker labels

The app refuses an action unless the configured worker image carries each label
it needs (`CONFIG_WORKER_LABELS`, `ops/executor.py`):

| Action | Labels |
|---|---|
| `reconcile_frontend_config`, `deploy_image` | `config-reconcile` |
| `reconcile_app_config` | `app-config-reconcile`, `app-source-guard`, `built-profile-v2` |
| `reconcile_tagged_config` | `config-reconcile`, `process-profile-v3` |
| `reconcile_service_plan` | all of the above for app and tagged, plus `service-plan-reconcile` |
| `reconcile_dev_config` | `config-reconcile`, `process-profile-v3`, `dev-config-reconcile` |

All are `llm-bawt.ops.<name>="1"` in `docker/Dockerfile.ops-worker`. The image
must also COPY `plan_reconcile.py` and `probe_contract.py`. The bridge and plan
rows need a rebuilt, repinned worker before they can run.

### Activation

Each step needs Nick's authorization.

1. Rebuild and repin the ops worker (as in the activation order above), then
   recreate app and ops-reconciler.
2. Insert the target rows and the plan row disabled (seeds never update existing
   rows), and enable each target row. A disabled target refuses the plan.
3. Install and preview the plan's `require_approval` policy, then enable the plan.
4. In Nick's window, run it through `ops_run`, with `accept_loss` only for
   losses Nick has accepted by name.

## Dev frontend kind (TASK-1649)

`bawthub.reconcile-dev-frontend-config` (`reconcile_dev_config`,
`ops/dev_config_reconcile.py`) recreates the BawtHub hot-reload dev frontend
from its own Compose service at the **same running image**. It never builds or
pulls, and it binds the image by ID, never by tag.

- **Why it is its own kind.** The service is Compose-built like the app, but a
  dev server serves its live bind-mounted working tree and HMR reloads every
  edit without a recreate. The app kind's source guard could only refuse it.
  The dev kind has no source check, and the card says so (`DEV target`).
- **Confined to a declared dev target.** The target's own rendered Compose
  service must carry the label `llm-bawt.ops.dev-target: "1"`. Otherwise it
  refuses with `dev_target_unmarked`, both at approval and when the worker
  re-binds before create. The marker lives in the reviewed Compose files, so a
  catalog edit that repoints a production row at this action cannot bind it.
  A marker that exists only on the running container is not enough. The action
  is approval-only (`APPROVAL_ONLY_ACTIONS`): there is no Run button and no
  direct dispatch.
- **Profile** (`DEV_PROFILE`, `ops/compose_config.py`): the built profile plus
  the memory cap. `mem_limit` and `memswap_limit` are both declared or both
  absent. A lone one refuses (`unsupported_compose_key`) because Docker's
  default swap is not modelled. The limit must be positive, and the swap must
  be `-1` or at least the limit. A changed cap is the drift name `memory:~`, and
  the recreate sets `HostConfig.Memory`/`MemorySwap`. No other profile accepts
  either key.
- **Card:** changed names, ports, mounts, tmpfs, `Memory after:` and the tag
  note (`image_tag_moved`). It never shows environment values.
- **Verify:** the declared Docker healthcheck, then the in-container
  `/api/health` field `status == healthy`, then the stat paths (mode and owner
  of the service token, never its contents). A failed verify restores the
  previous container.
- **Activation:** the BawtHub `frontend` service declares the marker label.
  The row ships disabled. Rebuild and repin the worker so it
  carries `dev-config-reconcile`, recreate app and ops-reconciler, insert the
  row, enable it, then `ops_run`. The generic `ops_run` approval policy gates
  the run. The target is dev only, so the approver is the coordinating arbiter,
  not Nick.
