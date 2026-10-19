# Same-image production routing repair (TASK-1179)

## Status

**Activated and repair verified on echo 2026-10-19.**
The new worker is installed and `bawthub.reconcile-prod-routing` is enabled with a
high-risk require-approval policy. Approved job
`389fd3f7eb2742a8944f82158b51154b` succeeded with exit code 0, preserving the exact
v0.1.78 frontend image. Independent checks through Traefik with the production
Host header returned HTTP 200 for `/api/system/health` and
`/api/system/health/checks`; the missing-route 404 was resolved. At verification,
the report correctly retained an unrelated OpenClaw outage. External unauthenticated
HTTPS reached the login gate; authenticated browser rendering was not verified by
that check. See TASK-1179 for activation and execution receipts.

## Contract

`bawthub.reconcile-prod-routing` is a disabled-by-default Docker operation with
**no arguments**. Its typed `reconcile_frontend_routing` spec permits only adding
`/api/system` to the existing production host-scoped API allowlist. It does not
interpret a Compose file or execute a shell. Existing hostnames and API prefixes
are preserved; unfamiliar Traefik grammar fails closed instead of being rewritten.

The approval binds:

- exact container ID and Compose identity;
- locally present image ID and immutable GHCR digest, with baked release identity;
- current Config, HostConfig and stable network settings via SHA-256 (private
  environment values and mount paths are not exposed in approval/receipt data);
- before/after router rule, shown in the approval prompt.

Worker-side validation repeats under the **same per-target lock used by image
deploy and rollback**. A changed image, container or runtime configuration rejects
stale approval before creation. The worker reuses the existing clone/swap/recovery
path with the same image on both sides. Only the routing rule and normal ops
provenance labels change. No pulls, builds, environment changes, socket-group
changes, mount changes, or other service actions are supported.

The replacement must pass Docker health and baked release checks. It then requests
`http://traefik/api/system/health` from the frontend with the catalogued production
Host header (fixed curl argv, not a shell). This exercises Traefik routing, not
merely localhost. A structured report with individual checks is required;
`status: down` is valid when an observed service is actually down. Docker-provider
convergence gets a bounded retry window. If verification fails, the old container
is restored by the shared replacement machinery. Failed restoration is reported as
an **unknown/lost** outcome, never successful or automatically retried.

If the route already exists, the action verifies release and routed report without
recreating the container. Durable worker receipts prevent replay of a completed
logical invocation. Receipts use `llm-bawt.ops.routing-reconcile/v1` in output_tail;
they are intentionally not image-release receipts or image rollback authorizations.

## Activation gates

Adding a catalog row alone is insufficient. Existing immutable workers cannot run
new code. The configured worker must carry `llm-bawt.ops.routing-reconcile=1`;
approval preparation and dispatch preflight reject a worker without it.

1. Run routing, image deployment, approval and full repository tests.
2. Review the exact scoped source diff; preserve unrelated release-workflow WIP.
3. Under explicit infrastructure rollout authorization, build
   `docker/Dockerfile.ops-worker` from a verified local Python base image. It copies
   the three stdlib-only modules; no dependency install or new frontend image.
4. Update the immutable `LLM_BAWT_OPS_WORKER_IMAGE` pin in app and reconciler
   deployment configuration. Preserve all existing Compose overlays, receipt
   mounts, credentials and historical worker images/receipts. Changing environment
   requires recreation, not an ordinary restart. **Do not touch bridges or Redis.**
   This bootstrap is outside the existing operations catalog and needs explicit
   authorization; do not disguise it as a restart or bypass ops policy.
5. Verify app and reconciler health and the actual installed worker capability.
6. Create the new operation disabled through the operator API, preserving existing
   rows. Install/preview its dedicated `require_approval` policy, reload policies,
   and enable the operation only after the implementation is available. Never
   overwrite existing operator-edited policies through a bulk seed.
7. Discover the live operation and request its normal `ops_run` approval. Inspect
   the same returned job through terminal receipt; do not resubmit uncertain work.
8. Independently verify the **production hostname** `/api/system/health` and
   `/api/system/health/checks`, the running image identity, detailed report rendering,
   and unrelated routes. Record evidence in TASK-1179/TASK-1157.

This is a bounded repair capability, not the larger ops-service consolidation and
not a general-purpose runtime reconfiguration API. Future route additions still
require both Compose labels and live hostname acceptance; releases intentionally
preserve current runtime configuration rather than silently applying all Compose
changes.
