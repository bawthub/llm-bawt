"""Digest-bound image deploy / rollback for ONE fixed container. STANDARD LIBRARY ONLY.

Runs inside the one-shot ops worker (TASK-997) next to ``worker.py``; the app
imports only the constants and :func:`parse_deployment` from here. The worker
still has no network, no repository checkout and no shell: every effect is a
Docker Engine API call on the fixed target named by the validated spec, and it
holds no registry credential.

Deploy and rollback share ONE replace path:

  inspect target → (deploy: ``repo@digest`` must already be on the daemon —
  the app pulls it in preflight) → verify image identity →
  (deploy: the running config must equal the rendered Compose service apart
  from the image, TASK-1518, else ``config_drift`` and nothing changes) →
  create clone ``<name>-next-<job8>`` → stop old → rename old ``-prev-<job8>`` →
  rename new to canonical → start → wait for Docker health → verify release
  identity via in-container ``/api/health`` → remove old container.

Any failure after the old container is stopped first tries to RESTORE it. A
verified restore is a known outcome (``failed``); anything else is uncertain
(``lost``) with the phase recorded. The deployment record is the JSON
``output_tail`` of the receipt (schema :data:`DEPLOY_SCHEMA`).
"""
from __future__ import annotations

import fcntl
import json
import re
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

if __package__:
    from .compose_config import CONFIG_HASH_LABEL, evaluate
    from .compose_render import ComposeRenderer
    from .service_identity import ComposeServiceRef
else:  # standalone immutable worker
    from compose_config import CONFIG_HASH_LABEL, evaluate
    from compose_render import ComposeRenderer
    from service_identity import ComposeServiceRef

DEPLOY_SCHEMA = "llm-bawt.ops.image-deploy/v1"
DEPLOY_ACTION = "deploy_image"
ROLLBACK_ACTION = "rollback_image"
IMAGE_ACTIONS = frozenset({DEPLOY_ACTION, ROLLBACK_ACTION})
HEALTH_URL = "http://127.0.0.1:3000/api/health"

# Container.Config keys a Compose service may override; when equal to the OLD
# image's value they were inherited and must come from the NEW image instead.
_INHERITABLE = ("Cmd", "Entrypoint", "WorkingDir", "User", "Healthcheck", "ExposedPorts",
                "StopSignal", "Volumes", "Shell", "OnBuild")
_COPIED = ("Domainname", "AttachStdin", "AttachStdout", "AttachStderr", "Tty", "OpenStdin",
           "StdinOnce", "StopTimeout")


def parse_deployment(output_tail: str | None) -> dict | None:
    """Return the deployment record from a receipt/job output, else None."""
    if not output_tail or not output_tail.lstrip().startswith("{"):
        return None
    try:
        value = json.loads(output_tail)
    except ValueError:
        return None
    return value if isinstance(value, dict) and value.get("schema") == DEPLOY_SCHEMA else None


class StepError(RuntimeError):
    """A deployer check whose message is built only from non-secret facts."""


def safe_error(exc: BaseException) -> str:
    """Describe a failure for receipts without echoing external text.

    Docker and transport errors can quote the request they rejected, and a
    create body carries environment values (TASK-1518 r1 B1). Keep only the
    exception type and the Docker HTTP status; StepError messages are ours.
    """
    if isinstance(exc, StepError):
        return str(exc)
    match = re.match(r"Docker HTTP (\d{3})\b", str(exc))
    return f"{type(exc).__name__}: Docker HTTP {match.group(1)}" if match else type(exc).__name__


def release_from_labels(labels: dict | None) -> dict | None:
    labels = labels or {}
    if labels.get("com.bawthub.release.schema") != "1":
        return None
    return {"version": labels.get("org.opencontainers.image.version", ""),
            "source_sha": labels.get("org.opencontainers.image.revision", ""),
            "workflow_run_id": labels.get("com.bawthub.release.workflow-run-id", "")}


class DeployFailed(RuntimeError):
    """Carries the deployment record; ``outcome_known`` means no uncertainty."""

    def __init__(self, message: str, detail: dict, *, outcome_known: bool):
        super().__init__(message)
        detail["error"] = message
        self.detail = detail
        self.outcome_known = outcome_known

    @property
    def output(self) -> str:
        return json.dumps(self.detail, sort_keys=True)


@dataclass(frozen=True)
class Swap:
    """One completed swap: the stopped previous container (renamed
    ``prev``) and the new container now under the canonical name."""

    old_id: str
    new_id: str
    prev: str


class ImageDeployer:
    def __init__(self, api, snapshot: dict, job_id: str, *, lock_root: Path,
                 sleep=time.sleep, clock=time.monotonic, renderer=None):
        self.api = api
        self._renderer_override = renderer
        self._config_hash = None
        self._declared = None
        self.spec = snapshot["spec"]
        self.snapshot = snapshot
        self.job_id = job_id
        self.lock_root = lock_root
        self.sleep = sleep
        self.clock = clock
        self.name = self.spec["container_name"]
        self.detail = {"schema": DEPLOY_SCHEMA, "job_id": job_id, "target": self.name,
                       "action": "deploy" if self.spec["action"] == DEPLOY_ACTION else "rollback",
                       "phase": "init", "previous": None, "deployed": None,
                       "restored": False, "warnings": [], "error": None}

    # ── entry ────────────────────────────────────────────────────────────
    def run(self) -> str:
        with self._target_lock():
            try:
                self._perform()
            except (DeployFailed, TimeoutError):
                raise
            except Exception as exc:  # noqa: BLE001 - never let external text reach a receipt
                self._fail(f"{self.detail['phase']} failed ({safe_error(exc)})",
                           known=not getattr(self.api, "side_effect_started", False))
        self.detail["phase"] = "done"
        return json.dumps(self.detail, sort_keys=True)

    @contextmanager
    def _target_lock(self):
        """Hold this target's lock, non-blocking: a busy lock fails known with
        nothing changed. A multi-target plan (TASK-1476) takes each target's
        lock in plan order through this same method."""
        self.lock_root.mkdir(mode=0o700, exist_ok=True)
        with (self.lock_root / f"{self.name}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._fail("another deploy/rollback holds this target", known=True)
            yield

    def _perform(self):
        if self.spec["action"] == DEPLOY_ACTION:
            self._deploy()
        else:
            self._rollback()

    def _create_body(self, old, old_image, new_image, ref):
        body = clone_create_body(old, old_image, new_image, ref, self.job_id)
        if self._config_hash:
            # Matches Compose's own hash of the service at this ref, so a later
            # `compose up` sees no drift and does not recreate the target.
            body["Labels"][CONFIG_HASH_LABEL] = self._config_hash
        if self._declared:
            # r1 B2: clone_create_body drops values equal to the OLD image's
            # defaults. A Compose-declared value stays explicit, so a changed
            # NEW image default can never override it.
            env = dict(e.partition("=")[::2] for e in body["Env"])
            env.update(self._declared["env"])
            body["Env"] = [f"{k}={v}" for k, v in sorted(env.items())]
            body["Labels"].update(self._declared["labels"])
        return body

    def _renderer(self):
        """Sealed one-shot Compose renderer (TASK-1518), injectable in tests."""
        return self._renderer_override or ComposeRenderer(self.api, self.spec, tag=self.job_id[:12])

    def _check_config(self, old, old_image, ref):
        """r2 B3, authoritative and under the target lock: refuse to clone a
        running configuration that differs from the declared Compose service.
        Compares everything but the image, which the deploy swaps on purpose."""
        self._phase("config-check")
        try:
            result = evaluate(self.spec, old, old_image, self._renderer(), ref)
        except ValueError as exc:  # ComposeError: fixed codes and names only
            self._fail(f"compose config check failed: {exc}; nothing changed", known=True)
        if result["drift"]:
            self.detail["config_drift"] = result["drift"]
            self._fail("config_drift: the running target differs from its Compose service ("
                       + ", ".join(result["drift"]) + "); run the config reconcile operation first; "
                       "nothing changed", known=True)
        self._config_hash = result["rendered_hash"]
        self._declared = {"env": result["desired"]["env"], "labels": result["desired"]["labels"]}

    def _fail(self, message: str, *, known: bool):
        raise DeployFailed(message, self.detail, outcome_known=known)

    def _phase(self, phase: str):
        self.detail["phase"] = phase

    # ── deploy / rollback specifics ──────────────────────────────────────
    def _deploy(self):
        release = self.snapshot["release"]
        ref = release["image_ref"]
        self.detail.update(release={k: release[k] for k in (
            "workflow_run_id", "workflow_run_url", "source_sha", "version", "tag", "digest", "image_ref")})
        old, old_image = self._target(expected_image_id=release["expected_current_image_id"])
        if old["Config"].get("Image") == ref:
            self._fail("target already runs this release digest; nothing to deploy", known=True)
        self._phase("verify-image")
        new_image = self._image(ref)
        if new_image is None:
            self._fail("approved image is not on the daemon (deploy preflight pulls it); nothing changed", known=True)
        repo_digest = f"{self.spec['image_repository']}@{release['digest']}"
        if repo_digest not in (new_image.get("RepoDigests") or []):
            self._fail(f"image does not carry RepoDigest {repo_digest}", known=True)
        expect = {"version": release["version"], "source_sha": release["source_sha"],
                  "workflow_run_id": str(release["workflow_run_id"])}
        if release_from_labels(new_image["Config"].get("Labels")) != expect:
            self._fail("image release labels do not match the approved run/source/version", known=True)
        if not self.spec.get("compose_config"):
            # Fail closed: a silent clone of stale configuration must not be possible.
            self._fail("deploy spec declares no compose_config; nothing changed", known=True)
        self._check_config(old, old_image, ref)
        self._replace(old, old_image, new_image, ref, expect)

    def _rollback(self):
        binding = self.snapshot["rollback"]
        self.detail["rollback_of"] = binding["deploy_job_id"]
        old, old_image = self._target(expected_image_id=binding["from_image_id"])
        self._phase("verify-image")
        target = self._image(binding["to_image_id"])
        if target is None:
            self._fail("last-known-good image is no longer present locally (rollback never pulls)", known=True)
        ref = binding["to_image_id"]
        pinned = binding.get("to_image_ref") or ""
        if pinned and pinned != ref:
            by_ref = self._image(pinned)
            if by_ref is not None and by_ref["Id"] == target["Id"]:
                ref = pinned  # Keep a digest reference when it still resolves identically.
        self._replace(old, old_image, target, ref, binding.get("to_release"))

    # ── shared replace path ──────────────────────────────────────────────
    def _target(self, *, expected_image_id: str):
        self._phase("inspect")
        old = self.api.request("GET", f"/containers/{quote(self.name, safe='')}/json")
        if not ComposeServiceRef.from_spec(self.spec).matches(old["Config"].get("Labels")):
            self._fail("target container does not carry the expected Compose identity", known=True)
        if old["Image"] != expected_image_id:
            self._fail("target image changed since approval; refusing a stale deploy/rollback", known=True)
        old_image = self._image(old["Image"]) or {"Config": {}}
        self.detail["previous"] = {"container_id": old["Id"], "image_id": old["Image"],
                                   "image_ref": old["Config"].get("Image", ""),
                                   "release": release_from_labels(old_image["Config"].get("Labels"))}
        return old, old_image

    def _replace(self, old, old_image, new_image, ref, expect_release):
        self._commit_swap(self._swap(old, old_image, new_image, ref, expect_release))

    def _swap(self, old, old_image, new_image, ref, expect_release) -> "Swap":
        """Create, stop, rename to -prev, rename, start, health and verify.
        Restores the previous container on any failure. On success the old
        container stays stopped as ``<name>-prev-<job8>`` until
        :meth:`_commit_swap` or :meth:`_revert`."""
        job8 = self.job_id[:8]
        temp, prev = f"{self.name}-next-{job8}", f"{self.name}-prev-{job8}"
        self._phase("create")
        body = self._create_body(old, old_image, new_image, ref)
        try:
            new_id = self.api.request("POST", "/containers/create?" + urlencode({"name": temp}), body=body)["Id"]
        except Exception as exc:  # noqa: BLE001 - production untouched; never echo the body
            self._fail(f"create failed ({safe_error(exc)}); nothing changed", known=True)
        self.detail["deployed"] = {"container_id": new_id, "image_id": new_image["Id"], "image_ref": ref,
                                   "release": release_from_labels(new_image["Config"].get("Labels"))}
        self._phase("stop-old")
        # From here production is affected: failures are restored or uncertain.
        self.api.side_effect_started = True
        try:
            self.api.request("POST", f"/containers/{old['Id']}/stop?t={self.spec.get('stop_grace_seconds', 10)}")
            self._phase("swap")
            self._rename(old["Id"], prev)
            self._rename(new_id, self.name)
            self._phase("start")
            self.api.request("POST", f"/containers/{new_id}/start")
            self._phase("health")
            self._wait_healthy(new_id)
            self._phase("verify-release")
            self.detail["health"] = self._verify_release(new_id, expect_release)
        except Exception as exc:  # noqa: BLE001 - every failure must attempt restore
            self._restore(Swap(old["Id"], new_id, prev), safe_error(exc))
        return Swap(old["Id"], new_id, prev)

    def _commit_swap(self, swap: "Swap"):
        """Delete the stopped previous container. Deployed and verified, so a
        failure here is only a warning."""
        self._phase("cleanup")
        try:
            self.api.request("DELETE", f"/containers/{swap.old_id}?v=0")
        except Exception as exc:  # noqa: BLE001 - deployed and verified; report only
            self.detail["warnings"].append(f"previous container {swap.prev} not removed: {safe_error(exc)}")

    def _restore(self, swap: "Swap", reason):
        failed_phase = self.detail["phase"]
        self.detail["failed_phase"] = failed_phase
        self._phase("restore")
        try:
            self._revert(swap)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"{failed_phase} failed ({reason}); RESTORE FAILED ({safe_error(exc)}); production state unknown", known=False)
        self.detail["restored"] = True
        self._fail(f"{failed_phase} failed ({reason}); previous container restored and healthy", known=True)

    def _revert(self, swap: "Swap"):
        """Put the previous container back: stop the new one and rename it
        ``-failed-<job8>``, rename the previous one to the canonical name,
        start it, wait for health, then delete the new one. Raises if the
        previous container is not back and healthy."""
        old_id, new_id = swap.old_id, swap.new_id
        for step in (lambda: self.api.request("POST", f"/containers/{new_id}/stop?t=5"),
                     lambda: self._rename(new_id, f"{self.name}-failed-{self.job_id[:8]}")):
            try:
                step()
            except Exception:  # noqa: BLE001 - best effort; verified below
                pass
        current = self.api.request("GET", f"/containers/{old_id}/json")
        if current["Name"].lstrip("/") != self.name:
            self._rename(old_id, self.name)
        self.api.request("POST", f"/containers/{old_id}/start")
        self._wait_healthy(old_id)
        self.api.request("DELETE", f"/containers/{new_id}?force=1&v=0")

    # ── Docker helpers ───────────────────────────────────────────────────
    def _image(self, ref: str):
        try:
            return self.api.request("GET", f"/images/{quote(ref, safe='')}/json")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def _rename(self, container_id, name):
        self.api.request("POST", f"/containers/{container_id}/rename?" + urlencode({"name": name}))

    def _wait_healthy(self, container_id):
        deadline = self.clock() + int(self.spec["health_timeout_seconds"])
        while True:
            state = self.api.request("GET", f"/containers/{container_id}/json")["State"]
            health = (state.get("Health") or {}).get("Status")
            if not state.get("Running"):
                raise StepError(f"container not running (status {state.get('Status')})")
            if health == "healthy":
                return
            if health is None:
                raise StepError("container has no Docker healthcheck; cannot verify")
            if health == "unhealthy" or self.clock() >= deadline:
                raise StepError(f"container health {health!r} within {self.spec['health_timeout_seconds']}s")
            self.sleep(2)

    def _verify_release(self, container_id, expect):
        exec_id = self.api.request("POST", f"/containers/{container_id}/exec", body={
            "AttachStdout": True, "AttachStderr": True,
            "Cmd": ["curl", "-fsS", "--max-time", "10", HEALTH_URL]})["Id"]
        raw = self.api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
        code = self.api.request("GET", f"/exec/{exec_id}/json").get("ExitCode")
        stdout = demux_stdout(raw)
        if code != 0:
            raise StepError(f"in-container health request exited {code}")
        payload = json.loads(stdout)
        if payload.get("status") != "healthy":
            raise StepError(f"health status {payload.get('status')!r}")
        baked = payload.get("release")
        if expect is not None:
            got = None if not isinstance(baked, dict) else {
                "version": baked.get("version"), "source_sha": baked.get("sourceSha"),
                "workflow_run_id": baked.get("workflowRunId")}
            if got != expect:
                raise StepError(f"running release {got} != expected {expect}")
        return {"status": payload.get("status"), "release": baked}


def clone_create_body(old: dict, old_image: dict, new_image: dict, ref: str, job_id: str) -> dict:
    """Container create body: the live Compose container with the image swapped.

    Compose-set values survive; values the OLD image supplied are dropped so the
    NEW image's labels/env/healthcheck/cmd apply (watchtower semantics).
    """
    oc, ic = old["Config"], (old_image.get("Config") or {})
    body = {k: oc[k] for k in _COPIED if k in oc}
    for key in _INHERITABLE:
        if oc.get(key) is not None and oc.get(key) != ic.get(key):
            body[key] = oc[key]
    image_env = set(ic.get("Env") or [])
    body["Env"] = [e for e in oc.get("Env") or [] if e not in image_env]
    image_labels = ic.get("Labels") or {}
    # Compose labels are container identity, never image-supplied: a locally
    # compose-BUILT image also carries project/service/version, and dropping
    # them makes the next deploy refuse the target (TASK-1035).
    labels = {k: v for k, v in (oc.get("Labels") or {}).items()
              if k.startswith("com.docker.compose.") or image_labels.get(k) != v}
    labels["com.docker.compose.image"] = new_image["Id"]
    labels["llm-bawt.ops.deployed-by"] = job_id
    body["Labels"] = labels
    body["Image"] = ref
    body["HostConfig"] = old["HostConfig"]
    short_id = old["Id"][:12]
    body["NetworkingConfig"] = {"EndpointsConfig": {
        net: {"Aliases": [a for a in (ep.get("Aliases") or []) if a != short_id],
              "IPAMConfig": ep.get("IPAMConfig"), "Links": ep.get("Links"), "DriverOpts": ep.get("DriverOpts")}
        for net, ep in (old.get("NetworkSettings", {}).get("Networks") or {}).items()}}
    return body


def demux_stdout(raw: bytes) -> str:
    """Extract stdout from Docker's multiplexed (non-TTY) attach stream."""
    out, i = bytearray(), 0
    while i + 8 <= len(raw):
        stream, size = raw[i], struct.unpack(">I", raw[i + 4:i + 8])[0]
        if stream == 1:
            out += raw[i + 8:i + 8 + size]
        i += 8 + size
    return out.decode(errors="replace")
