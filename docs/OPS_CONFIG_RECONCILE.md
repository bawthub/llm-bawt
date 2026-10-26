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
