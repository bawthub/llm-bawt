"""Durable Docker worker dispatch, no app-owned side-effect threads or SSH.

The Docker daemon owns a deterministic one-shot worker. The app submits an
immutable request on a named volume; only a persisted receipt can mean success.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..service.providers.github_deploy import GhcrPullAdapter
from ..utils.config import Config
from .app_config_reconcile import (ACTION as APP_CONFIG_ACTION, BUILT_PROFILE_LABEL, PROCESS_PROFILE_LABEL,
                                   SOURCE_GUARD_LABEL, WORKER_CAPABILITY_LABEL as APP_CONFIG_WORKER_LABEL,
                                   app_config_binding, uses_process_profile, validate_app_spec)
from .compose_config import evaluate
from .compose_render import ComposeError, ComposeRenderer
from .container_remove import (ACTION as REMOVE_ACTION, WORKER_CAPABILITY_LABEL as REMOVE_WORKER_LABEL,
                               RemoveRefused, removal_binding, target_name, validate_remove_spec)
from .config_reconcile import (ACTION as CONFIG_ACTION, WORKER_CAPABILITY_LABEL as CONFIG_WORKER_LABEL,
                               config_binding, validate_compose_config, validate_config_spec)
from .image_deploy import DEPLOY_ACTION, IMAGE_ACTIONS
from .plan_reconcile import (ACTION as PLAN_ACTION, WORKER_CAPABILITY_LABEL as PLAN_WORKER_LABEL,
                             validate_plan_spec)
from .release_spec import is_release_spec, validate_release_spec
from .service_identity import SPEC_PATTERNS as COMPOSE_SPEC_PATTERNS
from .routing_reconcile import (ACTION as ROUTING_ACTION, WORKER_CAPABILITY_LABEL,
                                routing_binding, validate_routing_spec)
from .source_state import SourceChecker
from .tagged_config_reconcile import ACTION as TAGGED_CONFIG_ACTION, tagged_config_binding, validate_tagged_spec
from .validation import canonical_json
from .worker import (OPS_JOB_LABEL, QUIESCE_FENCE_LABEL, DockerAPI, atomic_json, is_ops_worker,
                     target_selector)

SUPPORTED_ACTIONS = frozenset({"restart", "start", "stop", "pull"})
# Ops whose target configuration is compared with its Compose service (TASK-1518),
# and the worker capability labels each needs (TASK-1474: the app reconcile and
# its source guard; an app worker without the guard would swap unpublished source;
# and the v2 built profile, whose spec shape an older worker would refuse mid-job;
# TASK-1476: the tagged-image kind lives in the v3 process profile, which an app
# target also needs once its desired configuration uses a v3 key).
CONFIG_WORKER_LABELS = {CONFIG_ACTION: (CONFIG_WORKER_LABEL,), DEPLOY_ACTION: (CONFIG_WORKER_LABEL,),
                        APP_CONFIG_ACTION: (APP_CONFIG_WORKER_LABEL, SOURCE_GUARD_LABEL, BUILT_PROFILE_LABEL),
                        TAGGED_CONFIG_ACTION: (CONFIG_WORKER_LABEL, PROCESS_PROFILE_LABEL)}
# A service plan runs every kind it orders, plus its own module and probe contract.
CONFIG_WORKER_LABELS[PLAN_ACTION] = (*dict.fromkeys((*CONFIG_WORKER_LABELS[APP_CONFIG_ACTION],
                                                     *CONFIG_WORKER_LABELS[TAGGED_CONFIG_ACTION])),
                                     PLAN_WORKER_LABEL)
CONFIG_CHECKED_ACTIONS = frozenset(CONFIG_WORKER_LABELS)


class ExecutorError(RuntimeError):
    pass


class QuiesceFenceUnsupported(ExecutorError):
    """The worker this executor would start provably cannot enforce the quiesce
    start fence (TASK-1169 S3). Definitive, unlike a fence that cannot be read
    right now: a job pinned to that worker can never run fenced."""


class ConfigCheckError(ExecutorError):
    """A Compose config check or a container-removal check refused. ``code`` is
    fixed (``config_drift``, ``config_reconcile_unavailable``, a
    :class:`ComposeError` code or a :class:`RemoveRefused` code) and the
    message carries names only, never a value."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class DispatchResult:
    host_unit_name: str
    status_file_path: str = ""
    log_file_path: str = ""
    terminal_state: str | None = None  # Compatibility only; never trusted by service.
    exit_code: int | None = None
    output: str | None = None


@dataclass
class ReconcileResult:
    state: str | None
    exit_code: int | None = None
    output_tail: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class Executor(ABC):
    @abstractmethod
    def kind(self) -> str: ...
    @abstractmethod
    def available(self) -> bool: ...
    @abstractmethod
    def dispatch(self, **kwargs) -> DispatchResult: ...
    @abstractmethod
    def reconcile(self, **kwargs) -> ReconcileResult: ...

    def execution_settings(self, spec: dict | None = None) -> dict:
        return {}

    def readiness(self) -> str | None:
        """None when ready to dispatch, else the operator-facing reason.

        Health reporting reads this, so executors with a specific failure
        reason override it and derive ``available()`` from it.
        """
        return None if self.available() else "executor not available"

    def preflight(self, snapshot: dict) -> None:
        """Read-only prerequisite validation; no worker may be created here."""
        if not self.available():
            raise ExecutorError("executor not available")

    def inspect_target_image(self, spec: dict) -> str:
        """Image ID the fixed image-action target runs now (approval-time CAS)."""
        raise ExecutorError(f"{self.kind()} executor cannot inspect image deploy targets")

    def inspect_target_health(self, spec: dict) -> str:
        """Live health of the fixed target after a worker's verified swap."""
        raise ExecutorError(f"{self.kind()} executor cannot inspect deploy target health")

    def routing_approval(self, spec: dict) -> dict:
        """Public routing delta plus a private-config fingerprint; read-only."""
        raise ExecutorError(f"{self.kind()} executor cannot reconcile frontend routing")

    def config_approval(self, spec: dict) -> dict:
        """Public config-reconcile binding (names, hashes, A1 values); read-only."""
        raise ExecutorError(f"{self.kind()} executor cannot reconcile Compose configuration")

    def config_bind(self, spec: dict) -> tuple[dict, dict, dict]:
        """``(binding, desired, container attrs)`` of a config target; read-only."""
        raise ExecutorError(f"{self.kind()} executor cannot reconcile Compose configuration")

    def probe(self, container_name: str, argv: list[str]) -> str:
        """Stdout of a fixed cutover probe argv in a running target; read-only."""
        raise ExecutorError(f"{self.kind()} executor cannot run cutover probes")

    def remove_approval(self, spec: dict, args: dict) -> dict:
        """Binding for removing one stopped Compose orphan (TASK-1439 T2c); read-only."""
        raise ExecutorError(f"{self.kind()} executor cannot remove containers")

    def config_check(self, spec: dict, image_ref: str | None = None) -> None:
        """Refuse (ConfigCheckError) when the target's running configuration
        differs from its Compose service; read-only (TASK-1518)."""
        raise ExecutorError(f"{self.kind()} executor cannot check Compose configuration")

    def check_quiesce_fence(self, settings: dict) -> None:
        """Read-only: the worker this executor would start enforces the quiesce
        start fence (TASK-1233). Fails closed unless an executor proves it."""
        raise QuiesceFenceUnsupported(f"{self.kind()} executor cannot enforce the quiesce start fence")

    def check_target(self, spec: dict, args: dict) -> None:
        """Approval-time, read-only: the spec's target exists where this executor
        acts. Raise ExecutorError so nobody approves a guaranteed failure."""

    def target_compose_identity(self, spec: dict, args: dict) -> tuple[str | None, str | None]:
        """Read-only: the ``(project, service)`` Compose labels of the container
        the spec names. The GPU voice guard fails closed unless an executor
        can read them (TASK-1171)."""
        raise ExecutorError(f"{self.kind()} executor cannot inspect a target's Compose identity")


_IMAGE_SPEC_PATTERNS = {
    "container_name": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",
    **COMPOSE_SPEC_PATTERNS,
    "image_repository": r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*",
    "github_repository": r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
    "workflow_path": r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml",
    "canonical_branch": r"[A-Za-z0-9][A-Za-z0-9_./-]{0,99}",
}


def validate_image_spec(spec: dict) -> dict:
    """Fixed digest deploy/rollback target (TASK-997). No selector arguments:
    every identity is pinned in the operator-reviewed spec, never in job args.
    ``compose_config`` (TASK-1518) is required for deploy and optional for
    rollback, which never checks it: a rollback must stay possible."""
    allowed = {"action", "stop_grace_seconds", "health_timeout_seconds", "compose_config", *_IMAGE_SPEC_PATTERNS}
    if set(spec) - allowed or set(_IMAGE_SPEC_PATTERNS) - set(spec):
        raise ValueError(f"image spec requires exactly {sorted(allowed)} "
                         "(stop_grace_seconds optional, compose_config required for deploy)")
    if spec["action"] == DEPLOY_ACTION and "compose_config" not in spec:
        raise ValueError("deploy image spec requires compose_config (TASK-1518 drift guard)")
    if "compose_config" in spec:
        validate_compose_config(spec["compose_config"])
    for key, pattern in _IMAGE_SPEC_PATTERNS.items():
        if not isinstance(spec[key], str) or not re.fullmatch(pattern, spec[key]):
            raise ValueError(f"image spec {key} is invalid")
    grace = spec.get("stop_grace_seconds", 10)
    if type(grace) is not int or not 0 <= grace <= 300:
        raise ValueError("stop_grace_seconds must be an integer in 0..300")
    health = spec.get("health_timeout_seconds")
    if type(health) is not int or not 10 <= health <= 900:
        raise ValueError("health_timeout_seconds must be an integer in 10..900")
    return spec


def worker_request(job_id: str, snapshot: dict, fence: dict | None = None) -> dict:
    """The immutable worker request. ``fence`` (TASK-1233 quiesce:
    ``{generation, start_not_after}``) is present only for a quiesced job, so
    every other job's request, and its digest, is unchanged."""
    request = {"job_id": job_id, "snapshot": snapshot}
    if fence is not None:
        request["fence"] = fence
    return request


def validate_spec(command_script: str) -> dict:
    try:
        spec = json.loads(command_script)
    except (ValueError, TypeError) as exc:
        raise ValueError("command_script must be a Docker JSON spec") from exc
    if isinstance(spec, dict) and spec.get("action") == ROUTING_ACTION:
        return validate_routing_spec(spec)
    if isinstance(spec, dict) and spec.get("action") == CONFIG_ACTION:
        return validate_config_spec(spec)
    if isinstance(spec, dict) and spec.get("action") == APP_CONFIG_ACTION:
        return validate_app_spec(spec)
    if isinstance(spec, dict) and spec.get("action") == TAGGED_CONFIG_ACTION:
        return validate_tagged_spec(spec)
    if isinstance(spec, dict) and spec.get("action") == PLAN_ACTION:
        return validate_plan_spec(spec)
    if isinstance(spec, dict) and spec.get("action") in IMAGE_ACTIONS:
        return validate_image_spec(spec)
    if isinstance(spec, dict) and spec.get("action") == REMOVE_ACTION:
        return validate_remove_spec(spec)
    if is_release_spec(spec):
        return validate_release_spec(spec)
    allowed = {"action", "container_name", "container_name_from_arg", "compose_project",
               "compose_service", "compose_service_from_arg", "stop_grace_seconds"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Docker spec must be an object with supported fields only")
    if not isinstance(spec.get("action"), str) or spec["action"] not in SUPPORTED_ACTIONS:
        raise ValueError("unsupported Docker action")
    for k, v in spec.items():
        if k != "stop_grace_seconds" and (not isinstance(v, str) or not v.strip()):
            raise ValueError(f"Docker spec {k} must be a nonempty string")
    names = sum(k in spec for k in ("container_name", "container_name_from_arg"))
    services = sum(k in spec for k in ("compose_service", "compose_service_from_arg"))
    if not ((names == 1 and not services and "compose_project" not in spec) or
            (names == 0 and services == 1 and "compose_project" in spec)):
        raise ValueError("Docker spec requires exactly one name or project+service selector")
    grace = spec.get("stop_grace_seconds", 10)
    if type(grace) is not int or not 0 <= grace <= 3600:
        raise ValueError("stop_grace_seconds must be an integer in 0..3600")
    return spec


def _stored_pull_auth() -> dict | None:
    """ghcr.io pull credential from the encrypted CredentialStore (TASK-997)."""
    return GhcrPullAdapter(Config()).pull_auth()


class DockerExecutor(Executor):
    def __init__(self, *, client_factory=None, worker_image: str | None = None,
                 receipt_volume: str | None = None, receipt_root: str | None = None,
                 pull_auth=None, render_api_factory=None):
        self._client_factory = client_factory
        self._client = None
        # TASK-1518: the sealed Compose renderer speaks the worker's stdlib
        # Docker API (attach before start); injectable in tests.
        self._render_api_factory = render_api_factory or (lambda: DockerAPI(60))
        self.worker_image = worker_image or os.getenv("LLM_BAWT_OPS_WORKER_IMAGE", "")
        self.receipt_volume = receipt_volume or os.getenv("LLM_BAWT_OPS_RECEIPT_VOLUME", "")
        self.receipt_root = receipt_root or os.getenv("LLM_BAWT_OPS_RECEIPT_ROOT", "/var/lib/llm-bawt-ops")
        # TASK-997: release images are pulled HERE (deploy preflight) with the
        # DB-stored ghcr-pull credential; the worker never holds one.
        self._pull_auth = pull_auth or _stored_pull_auth

    def kind(self):
        return "docker"

    @property
    def client(self):
        if self._client is None:
            if self._client_factory:
                self._client = self._client_factory()
            else:
                import docker
                self._client = docker.from_env(timeout=15)
        return self._client

    def execution_settings(self, spec=None):
        return {"worker_image": self.worker_image, "receipt_volume": self.receipt_volume,
                "receipt_root": self.receipt_root}

    def inspect_target_image(self, spec):
        try:
            return self.client.containers.get(spec["container_name"]).attrs["Image"]
        except Exception as exc:
            raise ExecutorError(f"cannot inspect deploy target {spec['container_name']}: {exc}") from exc

    def inspect_target_health(self, spec):
        try:
            target = self.client.containers.get(spec["container_name"])
            target.reload()
            return ((target.attrs.get("State") or {}).get("Health") or {}).get("Status") or "unknown"
        except Exception as exc:
            raise ExecutorError(f"cannot inspect deploy target health {spec['container_name']}: {exc}") from exc

    def _require_worker_label(self, settings, label, *, missing, unverifiable, missing_error=ExecutorError):
        """The configured immutable worker image proves a capability by label."""
        try:
            worker = self.client.images.get(settings["worker_image"])
            labels = worker.attrs.get("Config", {}).get("Labels") or {}
            if labels.get(label) != "1":
                raise missing_error(missing)
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(unverifiable) from exc

    def _check_routing_worker(self, settings):
        self._require_worker_label(
            settings, WORKER_CAPABILITY_LABEL,
            missing="configured immutable ops worker lacks routing-reconcile v1; deploy the tested worker first",
            unverifiable="cannot verify routing worker capability")

    def check_quiesce_fence(self, settings):
        # The worker runs from its image, not the bind-mounted source: an image
        # built before the fence would start a quiesced job after its hold.
        self._require_worker_label(
            settings, QUIESCE_FENCE_LABEL,
            missing=("configured immutable ops worker does not enforce the quiesce start fence; "
                     "rebuild the worker image and repin LLM_BAWT_OPS_WORKER_IMAGE first"),
            unverifiable="cannot verify quiesce fence support of the ops worker",
            missing_error=QuiesceFenceUnsupported)

    def routing_approval(self, spec):
        self._check_prerequisites(self.execution_settings(spec))
        self._check_routing_worker(self.execution_settings(spec))
        try:
            target = self.client.containers.get(spec["container_name"])
            target.reload()
            image = self.client.images.get(target.attrs["Image"])
            return routing_binding(spec, target.attrs, image.attrs)
        except (ValueError, KeyError) as exc:
            raise ExecutorError(f"routing target cannot be bound: {exc}") from exc
        except Exception as exc:
            raise ExecutorError("cannot inspect routing target") from exc

    def _check_config_worker(self, settings, action=CONFIG_ACTION, extra=()):
        """The worker image carries the renderer, the reconciler and the deploy
        drift guard. An older one would deploy without the guard."""
        try:
            for label in (*CONFIG_WORKER_LABELS[action], *extra):
                self._require_worker_label(
                    settings, label,
                    missing=(f"configured immutable ops worker lacks {label}; "
                             "rebuild the worker image and repin LLM_BAWT_OPS_WORKER_IMAGE first"),
                    unverifiable="cannot verify the config-reconcile capability of the ops worker")
        except ExecutorError as exc:
            raise ConfigCheckError("config_reconcile_unavailable", f"config_reconcile_unavailable: {exc}") from exc

    def ensure_renderer_image(self, spec):
        """The digest-pinned renderer must be on the daemon: workers never pull
        and have no network. A public image, so the pull is anonymous."""
        ref = spec["compose_config"]["renderer_image"]
        try:
            self.client.images.get(ref)
            return
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise ExecutorError(f"cannot inspect the Compose renderer image: {exc}") from exc
        try:
            self.client.images.pull(ref)
        except Exception as exc:
            raise ConfigCheckError("renderer_image_missing",
                                   f"renderer_image_missing: pinned Compose renderer pull failed: {exc}") from exc

    def _config_inputs(self, spec):
        """(renderer, container attrs, image attrs) after every prerequisite."""
        settings = self.execution_settings(spec)
        self._check_prerequisites(settings)
        self._check_config_worker(settings, spec["action"])
        self.ensure_renderer_image(spec)
        try:
            target = self.client.containers.get(spec["container_name"])
            target.reload()
            image = self.client.images.get(target.attrs["Image"])
        except Exception as exc:
            raise ExecutorError(f"cannot inspect config target {spec['container_name']}") from exc
        renderer = ComposeRenderer(self._render_api_factory(), spec, tag="app" + os.urandom(4).hex())
        return renderer, target.attrs, image.attrs

    def _image_tag_moved(self, tag, image):
        """TASK-1474: does the Compose tag still name the running image? Shown
        to the approver only; the worker swaps by image ID either way."""
        try:
            return self.client.images.get(tag).attrs["Id"] != image["Id"]
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return True
            raise ExecutorError(f"cannot inspect the Compose image tag {tag}") from exc

    def config_approval(self, spec):
        return self.config_bind(spec)[0]

    def config_bind(self, spec):
        """``(binding, desired, container attrs)``. A service plan (TASK-1476)
        also reads the desired env names and the running Config.Env, which
        stay in this process: approval derives names from them, never values."""
        renderer, container, image = self._config_inputs(spec)
        try:
            if spec["action"] == APP_CONFIG_ACTION:
                checker = SourceChecker(self._render_api_factory(), spec["compose_config"]["renderer_image"],
                                        tag="src" + os.urandom(4).hex())
                binding, desired = app_config_binding(spec, container, image, renderer, source_checker=checker,
                                                      tag_moved=lambda tag: self._image_tag_moved(tag, image))
                if uses_process_profile(desired):
                    # TASK-1476: an older worker would refuse it mid-job; refuse first.
                    self._check_config_worker(self.execution_settings(spec), spec["action"],
                                              extra=(PROCESS_PROFILE_LABEL,))
                return binding, desired, container
            if spec["action"] == TAGGED_CONFIG_ACTION:
                return (*tagged_config_binding(spec, container, image, renderer,
                                               tag_moved=lambda tag: self._image_tag_moved(tag, image)), container)
            return (*config_binding(spec, container, image, renderer), container)
        except ComposeError as exc:  # fixed code and names only
            raise ConfigCheckError(exc.code, f"config target cannot be bound: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise ExecutorError(f"config target cannot be bound: {exc}") from exc

    def probe(self, container_name, argv):
        """Stdout of a fixed probe argv run in a running target (TASK-1476).
        The probe line carries booleans and closed codes only; stderr is dropped."""
        try:
            result = self.client.containers.get(container_name).exec_run(argv, stdout=True, stderr=False)
        except Exception as exc:
            raise ExecutorError(f"cannot run the probe in {container_name}: {type(exc).__name__}") from exc
        if result.exit_code != 0:  # the probes always exit 0 (TASK-1476 r1 B1)
            raise ExecutorError(f"the probe in {container_name} exited {result.exit_code}")
        return (result.output or b"").decode(errors="replace")

    def config_check(self, spec, image_ref=None):
        if not spec.get("compose_config"):
            raise ConfigCheckError("config_check_unconfigured",
                                   "deploy spec declares no compose_config; the drift guard cannot run")
        renderer, container, image = self._config_inputs(spec)
        try:
            result = evaluate(spec, container, image, renderer, image_ref or container["Config"].get("Image", ""))
        except ComposeError as exc:
            raise ConfigCheckError(exc.code, f"compose config check failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise ExecutorError(f"compose config check failed: {exc}") from exc
        if result["drift"]:
            raise ConfigCheckError("config_drift", (
                "config_drift: the running target differs from its Compose service ("
                + ", ".join(result["drift"]) + "); run the config reconcile operation first"))

    def _check_remove_worker(self, settings):
        self._require_worker_label(
            settings, REMOVE_WORKER_LABEL,
            missing=(f"configured immutable ops worker lacks {REMOVE_WORKER_LABEL}; "
                     "rebuild the worker image and repin LLM_BAWT_OPS_WORKER_IMAGE first"),
            unverifiable="cannot verify the container-remove capability of the ops worker")

    def remove_approval(self, spec, args):
        """TASK-1439 T2c: every removal refusal, read-only, before an approval card."""
        settings = self.execution_settings(spec)
        self._check_prerequisites(settings)
        self._check_remove_worker(settings)
        self.ensure_renderer_image(spec)
        try:
            name = target_name(spec, args)
            try:
                target = self.client.containers.get(name)
                target.reload()
            except Exception as exc:
                if getattr(exc, "status_code", None) == 404:
                    raise RemoveRefused("not_found", f"container {name!r} does not exist on this "
                                                     "executor's Docker host") from None
                raise ExecutorError(f"cannot inspect removal target {name!r}") from exc
            renderer = ComposeRenderer(self._render_api_factory(), spec, tag="rm" + os.urandom(4).hex())
            return removal_binding(spec, name, target.attrs, renderer)
        except RemoveRefused as exc:
            raise ConfigCheckError(exc.code, str(exc)) from exc

    def check_target(self, spec, args):
        # TASK-1002: this executor only reaches its own Docker daemon. A target
        # that lives on another host (e.g. Unraid) must fail HERE, before an
        # approval card, not as a 404 after a human approved it.
        name, filters = target_selector(spec, args)
        try:
            if name is not None:
                self.client.containers.get(name)
                return
            rows = self.client.containers.list(all=True, filters=filters)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                raise ExecutorError(
                    f"target container {name!r} does not exist on this executor's Docker host") from exc
            raise ExecutorError(f"cannot inspect target container: {exc}") from exc
        matches = [row for row in rows if not is_ops_worker(row.labels)]
        if len(matches) != 1:
            raise ExecutorError(
                f"compose selector {filters['label']} matched {len(matches)} containers on this "
                "executor's Docker host, expected one")

    def target_compose_identity(self, spec, args):
        try:
            name = spec.get("container_name") or target_selector(spec, args)[0]
        except (KeyError, TypeError) as exc:
            raise ExecutorError(f"cannot resolve the target container: {exc}") from exc
        if not name:
            raise ExecutorError("spec names no container to inspect")
        try:
            labels = self.client.containers.get(name).labels or {}
        except Exception as exc:
            raise ExecutorError(f"cannot inspect target container {name!r}: {exc}") from exc
        return labels.get("com.docker.compose.project"), labels.get("com.docker.compose.service")

    def available(self):
        return self.readiness() is None

    def readiness(self):
        try:
            self._check_prerequisites(self.execution_settings())
            return None
        except ExecutorError as exc:
            return str(exc)

    def preflight(self, snapshot):
        if is_release_spec(snapshot.get("spec")):
            raise ExecutorError("release orchestration specs run only on the release executor")
        self._check_prerequisites(snapshot["execution"])
        if snapshot["spec"].get("action") == ROUTING_ACTION:
            self._check_routing_worker(snapshot["execution"])
        if snapshot["spec"].get("action") == REMOVE_ACTION:
            # The worker re-renders and re-checks every refusal under the lock.
            self._check_remove_worker(snapshot["execution"])
            self.ensure_renderer_image(snapshot["spec"])
        if snapshot["spec"].get("action") in CONFIG_CHECKED_ACTIONS:
            # Cheap prerequisites only. The worker re-renders and compares under
            # the target lock; rendering here would churn renderer containers on
            # every reconcile tick of a job queued behind a prerequisite.
            self._check_config_worker(snapshot["execution"], snapshot["spec"]["action"])
            for target in (snapshot.get("plan") or {}).get("targets") or [snapshot]:
                self.ensure_renderer_image(target["spec"])
        if snapshot.get("quiesce"):
            self.check_quiesce_fence(snapshot["execution"])
        if snapshot["spec"].get("action") == DEPLOY_ACTION:
            self.ensure_release_image(snapshot["release"]["image_ref"])

    def ensure_release_image(self, ref):
        """Pull the approved ``repo@digest`` if absent. Idempotent and touches no
        container, so a failure leaves the job QUEUED with a visible reason. The
        worker still re-verifies RepoDigest and release labels before any swap."""
        try:
            self.client.images.get(ref)
            return
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise ExecutorError(f"cannot inspect release image: {exc}") from exc
        try:
            auth = self._pull_auth()
        except Exception as exc:  # noqa: BLE001 - DB/key trouble is a prerequisite, not a crash
            raise ExecutorError(f"registry pull credential unavailable: {type(exc).__name__}") from exc
        if not auth:
            raise ExecutorError("connect the ghcr-pull provider credential (Providers: GitHub container registry pull)")
        try:
            self.client.images.pull(ref, auth_config=auth)
        except Exception as exc:
            # Docker errors carry the registry message, never the credential.
            raise ExecutorError(f"release image pull failed: {exc}") from exc

    def _check_prerequisites(self, settings):
        """Settings + daemon reachability; every failure is an ExecutorError."""
        try:
            self._check_settings(settings)
            self.client.ping()
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"Docker worker prerequisites unavailable: {exc}") from exc

    def _check_settings(self, settings):
        image = settings.get("worker_image", "")
        if not re.fullmatch(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", image):
            raise ExecutorError("configure LLM_BAWT_OPS_WORKER_IMAGE as immutable image ID or digest")
        volume = settings.get("receipt_volume", "")
        root = settings.get("receipt_root", "")
        if not volume or not root or not Path(root).is_absolute():
            raise ExecutorError("configure a dedicated Docker receipt volume and absolute app mount path")
        # Never let Docker implicitly create a misspelled volume. Deployment must
        # mount this SAME named volume at receipt_root in the app.
        self.client.volumes.get(volume)
        self.client.images.get(image)  # No implicit pull at dispatch time.
        if not Path(root).is_dir():
            raise ExecutorError("receipt volume is not mounted at configured app path")
        owner = os.getenv("LLM_BAWT_OPS_APP_CONTAINER") or os.getenv("HOSTNAME", "")
        if not owner:
            raise ExecutorError("cannot identify submitting container to verify receipt mount")
        mounts = self.client.containers.get(owner).attrs.get("Mounts", [])
        if not any(m.get("Type") == "volume" and m.get("Name") == volume and
                   m.get("Destination") == root and m.get("RW") for m in mounts):
            raise ExecutorError("configured receipt path is not the configured writable named volume in this container")

    @staticmethod
    def worker_name(job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ExecutorError("invalid job id")
        return f"llm-bawt-ops-{job_id}"

    def _get_worker(self, name):
        try:
            return self.client.containers.get(name)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def dispatch(self, *, job_id: str, snapshot: dict, fence: dict | None = None, **_kwargs):
        self.worker_name(job_id)
        self.preflight(snapshot)
        directory = Path(snapshot["execution"]["receipt_root"]) / job_id
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / "submission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (directory / "abandoned.json").exists():
                raise ExecutorError("submission abandoned by recovery; never replay")
            return self._dispatch_locked(job_id=job_id, snapshot=snapshot, fence=fence)

    def _dispatch_locked(self, *, job_id: str, snapshot: dict, fence: dict | None = None):
        settings = snapshot["execution"]
        try:
            self._check_settings(settings)
            directory = Path(settings["receipt_root"]) / job_id
            directory.mkdir(mode=0o700, exist_ok=True)
            request = worker_request(job_id, snapshot, fence)
            raw = canonical_json(request).encode()
            path = directory / "request.json"
            # Create immutable request using hard-link publication (no overwrite
            # or partially written destination), then fsync its directory.
            if not path.exists():
                temp = directory / f"request.{os.getpid()}.{os.urandom(8).hex()}.tmp"
                with temp.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temp, path)
                except FileExistsError:
                    pass
                finally:
                    temp.unlink()
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            if path.read_bytes() != raw:
                raise ExecutorError("immutable request conflict")
            digest = hashlib.sha256(raw).hexdigest()
            name = self.worker_name(job_id)
            worker = self._get_worker(name)
            volumes = {settings["receipt_volume"]: {"bind": "/receipts", "mode": "rw"},
                       "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}}
            if worker is None:
                try:
                    worker = self.client.containers.create(
                        settings["worker_image"],
                        command=[f"/receipts/{job_id}/request.json", digest],
                        name=name, detach=True, network_mode="none", read_only=True,
                        cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                        restart_policy={"Name": "no"}, mem_limit="128m", pids_limit=32,
                        # Docker merges labels inherited from the immutable base
                        # image with labels supplied here. The current worker
                        # image was derived from the Compose app image and thus
                        # carries project=llm-bawt/service=app; override those
                        # identities so the worker can never select itself as a
                        # Compose operation target.
                        labels={
                            "com.docker.compose.project": "llm-bawt-ops-worker",
                            "com.docker.compose.service": "ops-worker",
                            OPS_JOB_LABEL: job_id,
                            "llm-bawt.ops.request": digest,
                        },
                        volumes=volumes,
                    )
                except Exception as exc:
                    if getattr(exc, "status_code", None) != 409:
                        raise
                    worker = self._get_worker(name)
            self._verify_worker(worker, job_id, digest)
            self._start_created_once(worker, directory)
            return DispatchResult(name, str(directory / "receipt.json"))
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"Docker worker submission uncertain: {exc}") from exc

    @staticmethod
    def _start_created_once(worker, directory):
        # Container.start is not a compare-and-swap. Two observers of 'created'
        # could otherwise start, then RESTART a fast-exited worker. Serialize
        # status/start and persist the attempt BEFORE the daemon call. A lost
        # start response remains uncertain and is never retried blindly.
        with (directory / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker.reload()
            if worker.status != "created":
                return True
            marker = directory / "start-attempt.json"
            if marker.exists():
                return False
            atomic_json(marker, {"start_attempted": True})
            worker.start()
            return True

    @staticmethod
    def _verify_worker(worker, job_id, digest):
        labels = worker.labels or {}
        if labels.get("llm-bawt.ops.job") != job_id or labels.get("llm-bawt.ops.request") != digest:
            raise ExecutorError("deterministic worker identity conflict")

    def reconcile(self, *, job_id: str, snapshot: dict, output_tail_bytes=4096,
                  fence: dict | None = None, **_kwargs):
        self.worker_name(job_id)
        root = Path(snapshot["execution"]["receipt_root"])
        if not root.is_dir():
            raise ExecutorError("receipt mount unavailable; cannot determine worker outcome")
        directory = root / job_id
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / "submission.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return ReconcileResult(None)
            return self._reconcile_locked(job_id=job_id, snapshot=snapshot,
                                          output_tail_bytes=output_tail_bytes, fence=fence)

    def _reconcile_locked(self, *, job_id: str, snapshot: dict, output_tail_bytes=4096,
                          fence: dict | None = None):
        directory = Path(snapshot["execution"]["receipt_root"]) / job_id
        digest = hashlib.sha256(canonical_json(worker_request(job_id, snapshot, fence)).encode()).hexdigest()
        try:
            receipt = None
            path = directory / "receipt.json"
            if path.exists():
                if path.stat().st_size > 4 * 1024 * 1024:
                    raise ExecutorError("oversized worker receipt")
                receipt = json.loads(path.read_text())
                if receipt.get("job_id") != job_id or receipt.get("request_hash") != digest:
                    raise ExecutorError("worker receipt identity conflict")
                state = receipt.get("state")
                if state not in {"accepted", "running", "succeeded", "failed", "lost", "timed_out"}:
                    raise ExecutorError("invalid worker receipt state")
                if state in {"succeeded", "failed", "lost", "timed_out"}:
                    if not receipt.get("finished_at"):
                        raise ExecutorError("terminal worker receipt has no finish timestamp")
                    if state == "succeeded" and receipt.get("exit_code") != 0:
                        raise ExecutorError("invalid success receipt")
                    return ReconcileResult(state, receipt.get("exit_code"),
                        (receipt.get("output_tail") or "").encode()[-max(1, output_tail_bytes):].decode(errors="replace"),
                        receipt.get("error"), receipt.get("started_at"), receipt.get("finished_at"))
            worker = self._get_worker(self.worker_name(job_id))
            if worker is None:
                atomic_json(directory / "abandoned.json", {"reason": "worker missing during reconciliation"})
                return ReconcileResult("lost", error="worker missing; side effects unknown, not replayed")
            self._verify_worker(worker, job_id, digest)
            worker.reload()
            if worker.status == "created":
                # Creation succeeded but submitter died before start. Safe to
                # start this SAME never-run identity, never create a replacement.
                if not directory.is_dir() or not self._start_created_once(worker, directory):
                    return ReconcileResult("lost", error="worker start previously attempted without confirmation; not replayed")
                return ReconcileResult("accepted")
            if worker.status in ("exited", "dead", "removing"):
                # Receipt publication may have raced our first read, immediately
                # before Docker reported exit. Re-read once after that observation.
                if path.exists():
                    final = json.loads(path.read_text())
                    if final.get("job_id") != job_id or final.get("request_hash") != digest:
                        raise ExecutorError("worker receipt identity conflict")
                    if final.get("state") in {"succeeded", "failed", "lost", "timed_out"}:
                        return self._reconcile_locked(job_id=job_id, snapshot=snapshot,
                                                      output_tail_bytes=output_tail_bytes, fence=fence)
                return ReconcileResult("lost", error="worker exited without terminal receipt; side effects unknown")
            if receipt and receipt.get("state") == "running":
                return ReconcileResult("running", started_at=receipt.get("started_at"))
            return ReconcileResult("accepted")
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"worker reconciliation unavailable: {exc}") from exc
