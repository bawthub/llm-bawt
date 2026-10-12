# Shared bridge Node runtime (TASK-1061)

## Image contract

`Dockerfile.agent-bridge` installs Node **24.21.0** and its bundled npm/npx
**11.19.0** from the official Node distribution. The x64 and arm64 SHA-256
checksums are pinned alongside the version. Unsupported architectures fail
closed. There is no distro Node/npm installation to compete on PATH.

Corepack **0.36.0** is explicitly installed, replacing any bundled copy, and
pnpm **10.32.1** is preloaded as the `bridge` user. This matches the current
BawtHub frontend package-manager version. Corepack can still honor another
repository's `packageManager`; that may require network access on first use.
Do not use `sudo npm` or change the host Node installation for this upgrade.

`docker/agent-bridge-constraints.txt` preserves all 49 Python dependency
versions from the deployed bridge image during this Node rebuild. Update these
constraints deliberately when upgrading Python harnesses; bumping an SDK lower
bound without updating its constraint will correctly fail dependency resolution.
The separate full-dependency `llm` CLI environment is not pinned by this file.
It explicitly includes `httpx`, whose absence already broke `llm --help` in the
old image. OS packages, Python patch version, uv and CLI-tool dependencies can
still advance in a fresh build; this is not a fully reproducible image lock.

## Build and validate without affecting services

From the repository root on the Docker host:

```sh
docker build --progress=plain -f Dockerfile.agent-bridge \
  -t llm-bawt-agent-bridge:task-1061-node24 .
docker run --rm --network none --cap-drop ALL \
  --security-opt no-new-privileges --entrypoint sh \
  llm-bawt-agent-bridge:task-1061-node24 \
  /usr/local/libexec/test-agent-bridge-toolchain.sh
```

The smoke script runs as `bridge`, checks exact versions/PATH, absence of apt
Node/npm binaries, Node ESM/fetch/WebSocket/crypto, Python bridge module imports,
all constrained Python versions, bundled Claude/Codex `--version`, uv and
`llm --help`. A temporary package.json exercises npx and pnpm command execution.
No service is started, no live home/config/source mount is attached, and no bot
or credential is used. Only amd64 was exercised on the deployment host; arm64
has a verified upstream checksum but still needs a native build/runtime test.

`.dockerignore` excludes `.env` and `.env.*`, including incident backups.

## Activation: separate authorization required

Discover the live ops catalog first. A plain bridge restart **cannot** apply an
image change. At implementation time the catalog has no build/recreate executor;
manual host Docker/Compose activation requires explicit approval for that exact
run. Recreate **only `claude-code-bridge`** for the REA prerequisite. Do not
restart Redis, app, Codex or OpenClaw. This interrupts the agent running the
operation, so save the receipt/task state and explain the handoff before running
it. Do not create a delayed background restart script.

Before authorization, record:

- Current target container ID, image ID and health.
- Candidate image ID after the final successful build; verify it again before
  activation so a moved tag cannot substitute an untested image.
- The resolved Compose service image and source/mount configuration (redact
  secrets; do not print the full resolved environment).
- Pending work and active sessions that will be interrupted.

After exact-run manual authorization, preserve the old image under a rollback
tag, point the shared image tag at the verified candidate, then use the existing
Compose configuration to recreate only the target:

```sh
# Substitute IDs recorded and revalidated for this exact run, not stale examples.
docker tag OLD_IMAGE_ID llm-bawt-agent-bridge:task-1061-rollback
docker tag VERIFIED_CANDIDATE_IMAGE_ID llm-bawt-agent-bridge:latest
docker compose up -d --no-deps --no-build --force-recreate claude-code-bridge
```

All three bridges share the image tag, but running Codex/OpenClaw containers keep
their old image until explicitly recreated. Updating the tag affects subsequent
recreates: report that remaining rollout scope rather than claiming every bridge
is upgraded. No git commit/push is implied by image activation.

After reconnecting, inspect the same target, confirm its image ID, health,
Node/npm/Corepack/pnpm versions as `bridge`, and run a normal Snark turn. Check
for startup/import failures. Compare other bridge container IDs/start times with
the baseline. TASK-1060 remains gated until the live bridge verification passes.

## Rollback

If activation fails, inspect the container/logs before retrying. With appropriate
operator authorization, restore the shared tag to the recorded old image ID and
recreate only `claude-code-bridge` using the same command. Verify old image identity,
health and a normal turn. Do not rebuild an old image from floating dependencies
as a substitute for the preserved image, and do not restart the whole stack.
