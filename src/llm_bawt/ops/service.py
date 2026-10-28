"""Catalog validation, immutable approval snapshots and durable job orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from .app_config_reconcile import ACTION as APP_CONFIG_ACTION, newly_trusted, valid_app_binding
from .config_reconcile import ACTION as CONFIG_ACTION, valid_config_binding
from .tagged_config_reconcile import ACTION as TAGGED_CONFIG_ACTION, valid_tagged_binding
from .plan_reconcile import ACTION as PLAN_ACTION, valid_plan_snapshot
from .executor import (ConfigCheckError, DockerExecutor, Executor, ExecutorError, QuiesceFenceUnsupported,
                       validate_spec)
from .gpu_guard import GPU_VOICE_SERVICES, GpuGuardUnverifiable, gpu_voice_target
from .image_deploy import DEPLOY_ACTION, IMAGE_ACTIONS
from .release_spec import is_release_spec
from .routing_reconcile import ACTION as ROUTING_ACTION, valid_routing_binding
from .release_store import ReleaseStore
from .release import (GitHubReleaseVerifier, ReleaseVerificationError, ReleaseVerifier,
                      bind_image_invocation, binding_matches_args)
from .models import (JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, JOB_LOST, JOB_QUEUED,
                     JOB_QUIESCING, JOB_RUNNING, JOB_SUCCEEDED, JOB_TERMINAL_STATES, OpsOperation)
from .quiesce import cancel_quiesce_job
from .quiesce_settings import quiesce_binding
from .quiesce_status import attach_members
from .store import IdempotencyConflict, OpsStore
from .validation import ArgValidationError, canonical_json, validate_args

logger = logging.getLogger(__name__)

# TASK-1474 C: the app reconcile recreates the process serving every chat turn
# from bind-mounted source, so it runs only from an approved ops_run request:
# the card shows exactly what the snapshot trusts. A policy that allows
# ops_run outright, the operator Run button and other internal callers are
# refused.
APPROVAL_REQUIRED = ("this operation runs only through an approval card: ask an agent to run it with ops_run "
                     "and approve the card. The Run button and direct dispatch cannot start it")
# TASK-1476: a service plan recreates the app too, so the same rule holds.
APPROVAL_ONLY_ACTIONS = frozenset({APP_CONFIG_ACTION, PLAN_ACTION})


def _spec_action(operation) -> str | None:
    try:
        spec = json.loads(operation.command_script) if operation is not None else None
    except (TypeError, ValueError):
        return None
    return spec.get("action") if isinstance(spec, dict) else None


def _valid_config_snapshot(snapshot: dict) -> bool:
    """The config binding of a detached snapshot, by the kind that executes it."""
    action = snapshot["spec"].get("action")
    if action == PLAN_ACTION:
        return "config" not in snapshot and valid_plan_snapshot(snapshot)
    if "plan" in snapshot:  # only a plan executes a plan block
        return False
    if action == APP_CONFIG_ACTION:
        return valid_app_binding(snapshot)
    if action == TAGGED_CONFIG_ACTION:
        return valid_tagged_binding(snapshot)
    return valid_config_binding(snapshot)


class OpsDispatchError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


PREFLIGHT_DEADLINE_ENV = "LLM_BAWT_OPS_PREFLIGHT_DEADLINE_SECONDS"
DEFAULT_PREFLIGHT_DEADLINE_SECONDS = 300


def preflight_deadline_seconds() -> int:
    """How long a queued job may keep failing preflight before it fails
    (TASK-1169 S1e). A missing, non-integer or non-positive value is the default."""
    try:
        value = int(os.environ.get(PREFLIGHT_DEADLINE_ENV, ""))
    except ValueError:
        return DEFAULT_PREFLIGHT_DEADLINE_SECONDS
    return value if value > 0 else DEFAULT_PREFLIGHT_DEADLINE_SECONDS


def display_args(args, schema_json):
    """``args`` with every ``x-sensitive`` value (per ``schema_json``) masked."""
    def redact(value, schema):
        if schema.get("x-sensitive"):
            return "***"
        if isinstance(value, dict):
            return {k: redact(v, schema.get("properties", {}).get(k, {})) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v, schema.get("items", {})) for v in value]
        return value
    return redact(args, json.loads(schema_json))


def _timestamp(raw):
    return datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None


class _NoTargetReads:
    """Stands in for an executor where no Docker read may happen: the GPU guard
    then resolves a target only from the spec itself (TASK-1169 S3b)."""

    @staticmethod
    def target_compose_identity(spec, args):
        raise ExecutorError("no Docker read is available here")


class OpsService:
    def __init__(self, store: OpsStore, *, executor: Executor | None = None,
                 release_verifier: ReleaseVerifier | None = None,
                 release_store: ReleaseStore | None = None, quiesce_driver=None,
                 admission_only: bool = False, preparer=None):
        """``admission_only`` (the app, TASK-1169 S3): this service admits and
        reads jobs and never reaches an executor. ``preparer.prepare(slug,
        args)`` builds snapshots in the ops process, and the ops reconciler
        dispatches and reconciles every job this service creates."""
        self.store = store
        # Only the reconciler process drives quiesce jobs (TASK-1233); every
        # other OpsService just reports them.
        self.quiesce_driver = quiesce_driver
        self.releases = release_store or ReleaseStore(None, engine=store.engine)
        self._release_verifier = release_verifier or GitHubReleaseVerifier()
        self.admission_only = admission_only
        if admission_only:
            if executor is not None or preparer is None:
                raise ValueError("an admission-only OpsService takes a preparer and no executor")
            self.preparer = preparer
            self._executors = {}
            return
        self.preparer = None
        default = executor or DockerExecutor()
        self._executors = {default.kind(): default}

    def register_executor(self, executor):
        if self.admission_only:
            raise ValueError("an admission-only OpsService runs no executor")
        self._executors[executor.kind()] = executor

    def executors(self):
        """Registered executors by kind (read-only copy, for health reporting)."""
        return dict(self._executors)

    def executor_readiness(self) -> dict[str, str | None]:
        """Each executor kind's readiness reason (``None`` when ready), from
        the process that runs them: the ops service's for an admission-only
        one. Raises OpsDispatchError when that process can't be asked."""
        if self.admission_only:
            return self.preparer.executor_readiness()
        readiness = {}
        for kind, executor in sorted(self._executors.items()):
            try:
                readiness[kind] = executor.readiness()
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                readiness[kind] = f"{type(exc).__name__}: {exc}"
        return readiness

    def _resolve_executor(self, kind):
        if self.admission_only:
            raise OpsDispatchError("ops_admission_only",
                                   "this process admits ops jobs; the ops service prepares and runs them")
        executor = self._executors.get(kind)
        if executor is None:
            raise OpsDispatchError("executor_kind_unknown", f"no executor registered for {kind!r}")
        return executor

    @staticmethod
    def _require_quiesce_fence(executor, execution):
        """A quiesce job runs only on a worker that proves it enforces the start
        fence (TASK-1233); an older worker would start after its hold lapsed."""
        try:
            executor.check_quiesce_fence(execution)
        except ExecutorError as exc:
            raise OpsDispatchError("quiesce_unavailable", str(exc)) from exc

    def list_operations_for_agent(self, *, include_disabled=False):
        return [row.to_agent_summary() for row in self.store.list_operations(include_disabled=include_disabled)]

    def get_operation(self, slug):
        return self.store.get_operation_by_slug(slug)

    def _check_source_published(self, commit: dict) -> None:
        """TASK-1474: the app's checkout is clean at HEAD (the executor's sealed
        check); that HEAD must also be the published branch head, so a
        reconcile never runs code that exists only on this host."""
        try:
            published = self._release_verifier.branch_head(commit["repository"], commit["branch"])
        except ReleaseVerificationError as exc:
            raise OpsDispatchError("source_unverifiable", f"source_unverifiable: {exc}") from exc
        if published != commit["head"]:
            raise OpsDispatchError("source_not_published", (
                f"source_not_published: the checkout is at {commit['head'][:12]} but "
                f"{commit['repository']} {commit['branch']} is at {published[:12]}; publish that commit first"))

    def _last_classification(self, operation_slug: str):
        """The trust classification the last successful run executed, or
        ``None`` (first run, or a run from before classification existed)."""
        for job in self.store.list_jobs(operation_slug=operation_slug, state=JOB_SUCCEEDED, limit=1):
            try:
                previous = json.loads(job.invocation_snapshot_json or "null")["config"]["classification"]
            except (ValueError, TypeError, KeyError):
                return None
            return previous if isinstance(previous, dict) else None
        return None

    def prepare_invocation(self, operation_slug: str, args: dict | None) -> dict:
        """Trusted approval integration: persist this JSON BEFORE asking approval.

        Detached JSON snapshot; callers must never accept it from tool arguments.
        dispatch_job verifies the snapshot and original input, then executes this
        exact revision even if defaults/spec/config have subsequently changed.
        Current enabled/deleted state remains an emergency execution interlock.
        An admission-only service asks the ops service to build it (TASK-1169 S3a).
        """
        if self.admission_only:
            return self.preparer.prepare(operation_slug, args)
        op = self.store.get_operation_by_slug(operation_slug)
        if op is None:
            raise OpsDispatchError("operation_not_found", f"unknown operation slug: {operation_slug!r}")
        if not op.enabled or op.soft_deleted_at is not None:
            raise OpsDispatchError("operation_disabled", f"operation {operation_slug!r} is disabled")
        try:
            merged = validate_args({} if args is None else args, op.args_schema_json, op.args_defaults_json)
            spec = validate_spec(op.command_script)
            for field in ("container_name_from_arg", "compose_service_from_arg"):
                if field in spec and not merged.get(spec[field]):
                    raise ValueError(f"{field} references an absent/empty argument")
        except (ValueError, ArgValidationError) as exc:
            raise OpsDispatchError("args_invalid", str(exc)) from exc
        executor = self._resolve_executor(op.executor_kind)
        execution = {key: getattr(op, key) for key in ("executor_kind", "target_host", "run_as_user",
            "working_directory", "timeout_seconds", "start_delay_seconds", "max_output_bytes", "max_concurrent")}
        execution.update(executor.execution_settings(spec))
        snapshot = {"snapshot_version": 1, "operation": op.to_api(), "spec": spec,
                    "schema": json.loads(op.args_schema_json), "defaults": json.loads(op.args_defaults_json),
                    "input_args": {} if args is None else args, "resolved_args": merged, "execution": execution}
        if spec["action"] == ROUTING_ACTION:
            if merged:
                raise OpsDispatchError("args_invalid", "routing reconciliation accepts no arguments")
            try:
                snapshot["routing"] = executor.routing_approval(spec)
            except (ValueError, ExecutorError) as exc:
                raise OpsDispatchError("target_unavailable", str(exc)) from exc
        elif spec["action"] in (CONFIG_ACTION, APP_CONFIG_ACTION, TAGGED_CONFIG_ACTION):
            # TASK-1518: the approver sees drift NAMES, mounts and closed-enum
            # values; the worker re-renders and compares under the target lock.
            # TASK-1474: the same for the Compose-built app; TASK-1476: and for
            # a tagged public image.
            if merged:
                raise OpsDispatchError("args_invalid", "config reconciliation accepts no arguments")
            try:
                snapshot["config"] = executor.config_approval(spec)
            except ConfigCheckError as exc:
                raise OpsDispatchError(exc.code, str(exc)) from exc
            except (ValueError, ExecutorError) as exc:
                raise OpsDispatchError("target_unavailable", str(exc)) from exc
            if spec["action"] == APP_CONFIG_ACTION:
                self._check_source_published(snapshot["config"]["source_commit"])
                snapshot["newly_trusted"] = newly_trusted(snapshot["config"]["classification"],
                                                          self._last_classification(operation_slug))
        elif spec["action"] == PLAN_ACTION:
            # TASK-1476: every target bound as for its own approval, plus the
            # credential classes from the running app's records and env names.
            from .plan_approval import PlanError, bind_plan  # env_import: app-only, kept off the import path
            try:
                snapshot["plan"] = bind_plan(spec, merged.get("accept_loss"), executor=executor,
                                             get_operation=self.store.get_operation_by_slug,
                                             executor_kind=op.executor_kind,
                                             check_source=self._check_source_published,
                                             last_classification=self._last_classification)
            except PlanError as exc:
                raise OpsDispatchError(exc.code, str(exc)) from exc
            if not valid_plan_snapshot(snapshot):
                raise OpsDispatchError("plan_invalid", (
                    "the service plan does not validate: targets must share one Compose project, distinct "
                    "containers, depend only on earlier targets, and the records target must be the app"))
        elif spec["action"] in IMAGE_ACTIONS:
            # TASK-997: the approver sees and the worker executes ONE verified
            # release (or recorded last-known-good) plus an image CAS.
            try:
                snapshot.update(bind_image_invocation(spec=spec, args=merged, store=self.store,
                                                      releases=self.releases, executor=executor,
                                                      verifier=self._release_verifier))
            except (ReleaseVerificationError, ExecutorError) as exc:
                raise OpsDispatchError("release_unverified", str(exc)) from exc
            if spec["action"] == DEPLOY_ACTION:
                # TASK-1518: never ask a human to approve a deploy that would
                # clone a stale configuration; the worker re-checks under lock.
                try:
                    executor.config_check(spec, snapshot["release"]["image_ref"])
                except ConfigCheckError as exc:
                    raise OpsDispatchError(exc.code, str(exc)) from exc
                except ExecutorError as exc:
                    raise OpsDispatchError("target_unavailable", str(exc)) from exc
        else:
            # TASK-1002: never ask a human to approve a container this
            # executor's Docker daemon does not have.
            try:
                executor.check_target(spec, merged)
                if is_release_spec(spec):
                    snapshot["release_source"] = executor.approval_source(spec, merged)
            except ConfigCheckError as exc:  # TASK-1518: the release's deploy would refuse
                raise OpsDispatchError(exc.code, str(exc)) from exc
            except ExecutorError as exc:
                raise OpsDispatchError("target_unavailable", str(exc)) from exc
        try:
            binding = quiesce_binding(snapshot["operation"], merged)
        except ValueError as exc:
            raise OpsDispatchError("args_invalid", str(exc)) from exc
        if binding is not None:
            snapshot["quiesce"] = binding
            # Never put a quiesce action before an approver that cannot run fenced.
            self._require_quiesce_fence(executor, execution)
        # TASK-1169 S3b: the GPU voice guard's creation check reads this, so the
        # process that creates the job needs no Docker read of its own.
        try:
            snapshot["gpu_target"] = gpu_voice_target(spec, merged, executor, live=False)
        except GpuGuardUnverifiable as exc:
            raise OpsDispatchError("gpu_guard_unverifiable", str(exc)) from exc
        snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
        return json.loads(canonical_json(snapshot))

    def _verify_snapshot(self, snapshot, operation_slug, args):
        try:
            detached = json.loads(canonical_json(snapshot))
            digest = detached.pop("snapshot_hash")
            if hashlib.sha256(canonical_json(detached).encode()).hexdigest() != digest:
                raise ValueError("snapshot content hash mismatch")
            detached["snapshot_hash"] = digest
            op = detached["operation"]
            if detached["snapshot_version"] != 1 or op["slug"] != operation_slug or canonical_json(detached["input_args"]) != canonical_json(args):
                raise ValueError("snapshot does not match requested operation/args")
            if detached["schema"] != json.loads(op["args_schema_json"]) or detached["defaults"] != json.loads(op["args_defaults_json"]):
                raise ValueError("snapshot schema/defaults mismatch")
            if detached["spec"] != validate_spec(op["command_script"]):
                raise ValueError("snapshot spec mismatch")
            if detached["resolved_args"] != validate_args(args, op["args_schema_json"], op["args_defaults_json"]):
                raise ValueError("snapshot resolved args mismatch")
            for key in ("executor_kind", "target_host", "run_as_user", "working_directory", "timeout_seconds",
                        "start_delay_seconds", "max_output_bytes", "max_concurrent"):
                if detached["execution"][key] != op[key]:
                    raise ValueError(f"snapshot execution {key} mismatch")
            if detached.get("quiesce") != quiesce_binding(op, detached["resolved_args"]):
                raise ValueError("snapshot quiesce binding mismatch")
            if not valid_routing_binding(detached):
                raise ValueError("snapshot routing binding mismatch")
            if not _valid_config_snapshot(detached):
                raise ValueError("snapshot config binding mismatch")
            if detached["spec"].get("action") == PLAN_ACTION:
                from .plan_approval import credential_names
                if detached["plan"]["credential_names"] != credential_names():
                    raise ValueError("snapshot credential names mismatch")
                for target in detached["plan"]["targets"]:
                    if target["spec"] != validate_spec(target["operation"]["command_script"]):
                        raise ValueError("snapshot plan target spec mismatch")
            if not binding_matches_args(detached):
                raise ValueError("snapshot release/rollback binding mismatch")
            if is_release_spec(detached["spec"]):
                source = detached.get("release_source") or {}
                legacy = set(source) == {"expected_sha"}
                end_to_end = set(source) == {"expected_sha", "deploy_authorization", "expected_current_image_id",
                                             "deploy_operation_id", "deploy_operation_version", "deploy_operation_script_hash"}
                if (not (legacy or end_to_end)
                        or not isinstance(source["expected_sha"], str)
                        or len(source["expected_sha"]) != 40
                        or any(char not in "0123456789abcdef" for char in source["expected_sha"])):
                    raise ValueError("snapshot approved release source is invalid")
                if end_to_end and (source["deploy_authorization"] != "after_verified_build"
                                   or not isinstance(source["expected_current_image_id"], str)
                                   or len(source["expected_current_image_id"]) != 71
                                   or not source["expected_current_image_id"].startswith("sha256:")
                                   or any(char not in "0123456789abcdef" for char in source["expected_current_image_id"][7:])
                                   or not isinstance(source["deploy_operation_id"], str)
                                   or not source["deploy_operation_id"]
                                   or not isinstance(source["deploy_operation_version"], int)
                                   or source["deploy_operation_version"] < 1
                                   or not isinstance(source["deploy_operation_script_hash"], str)
                                   or len(source["deploy_operation_script_hash"]) != 64
                                   or any(char not in "0123456789abcdef" for char in source["deploy_operation_script_hash"])):
                    raise ValueError("snapshot production baseline is invalid")
            if detached["spec"].get("action") == DEPLOY_ACTION:
                binding = self.releases.verified_binding(detached["resolved_args"]["release_run_id"])
                release = detached.get("release") or {}
                for key in ("workflow_run_id", "workflow_run_attempt", "source_sha", "version",
                            "tag", "digest", "image_repository", "trigger_sha"):
                    if str(release.get(key)) != str(binding.get(key)):
                        raise ValueError(f"snapshot release binding changed: {key}")
            return detached
        except (ValueError, TypeError, KeyError) as exc:
            raise OpsDispatchError("snapshot_invalid", str(exc)) from exc

    def dispatch_job(self, *, operation_slug: str, args: dict[str, Any] | None,
                     idempotency_key: str | None = None, approved_snapshot: dict | None = None,
                     caller_actor=None, caller_bot_id=None, caller_user_id=None, caller_turn_id=None,
                     caller_session_key=None, caller_backend=None, approval_request_id=None,
                     create_fence=None, caller_session_id=None) -> dict:
        """Create (or, for a known key, read) a job. ``create_fence`` guards only
        creation; see ``OpsStore.create_job``. A key replay stays a plain read.
        ``caller_session_id`` (from the caller's signed turn capability) lets a
        direct quiesce job push its terminal receipt to the caller it holds."""
        supplied = {} if args is None else args
        try:
            payload = canonical_json({"operation": operation_slug, "args": supplied})
        except (ValueError, TypeError) as exc:
            raise OpsDispatchError("args_invalid", str(exc)) from exc
        key = idempotency_key.strip() if idempotency_key else uuid.uuid4().hex
        if not 1 <= len(key) <= 128:
            raise OpsDispatchError("args_invalid", "idempotency_key must be 1..128 characters")
        snapshot = self._verify_snapshot(approved_snapshot, operation_slug, supplied) if approved_snapshot is not None else None
        existing = self.store.get_job_by_key(key)
        if existing is not None:
            try:
                self.store.verify_payload(existing, payload, canonical_json(snapshot) if snapshot is not None else None)
            except IdempotencyConflict as exc:
                raise OpsDispatchError("idempotency_conflict", str(exc)) from exc
            # Including queued: a replay is a read, not another submission.
            result = existing.to_api()
            if existing.operation_slug == "bawthub.release-prod":
                release = self.releases.get_by_parent_job(existing.id)
                result["release_id"] = release.id if release else None
            return result
        if snapshot is None and _spec_action(self.store.get_operation_by_slug(operation_slug)) in APPROVAL_ONLY_ACTIONS:
            # Refused before prepare_invocation, which would start the sealed
            # source check for a caller that can never run it.
            raise OpsDispatchError("approval_required", APPROVAL_REQUIRED)
        snapshot = snapshot or self.prepare_invocation(operation_slug, supplied)
        if snapshot["spec"].get("action") in APPROVAL_ONLY_ACTIONS and (
                approved_snapshot is None or create_fence is None or not approval_request_id):
            # Keyed on what would EXECUTE, so a catalog edit between the check
            # above and the snapshot cannot slip past it (TASK-1474 C-b).
            raise OpsDispatchError("approval_required", APPROVAL_REQUIRED)
        current = self.store.get_operation_by_slug(operation_slug)
        if current is None or not current.enabled or current.soft_deleted_at is not None:
            raise OpsDispatchError("operation_disabled", "operation is currently disabled or deleted")
        if current.id != snapshot["operation"]["id"]:
            raise OpsDispatchError("snapshot_invalid", "operation identity changed")
        op = OpsOperation(**{k: v for k, v in snapshot["operation"].items()
                             if k not in {"created_at", "updated_at", "soft_deleted_at"}})
        self._require_gpu_handoff(snapshot, create_fence)
        quiesce = bool(snapshot.get("quiesce"))
        if quiesce and getattr(getattr(self.store.engine, "dialect", None), "name", None) != "postgresql":
            raise OpsDispatchError("quiesce_unavailable", "quiesce operations require the PostgreSQL ops store")
        if quiesce and not self.admission_only:
            # Re-checked for an approved snapshot: the worker it pins is what runs.
            # Admission-only, the ops driver checks it before opening any hold.
            self._require_quiesce_fence(self._resolve_executor(snapshot["execution"]["executor_kind"]),
                                        snapshot["execution"])
        callback = (self._receipt_callback(snapshot, bot_id=caller_bot_id, user_id=caller_user_id,
                                           session_id=caller_session_id)
                    if quiesce and not approval_request_id else None)
        try:
            job = self.store.create_job(operation=op, args_json=canonical_json(snapshot["resolved_args"]),
                display_args_json=canonical_json(display_args(snapshot["resolved_args"], op.args_schema_json)),
                idempotency_key=key, invocation_snapshot_json=canonical_json(snapshot), request_payload_json=payload,
                caller_actor=caller_actor, caller_bot_id=caller_bot_id, caller_user_id=caller_user_id,
                caller_turn_id=caller_turn_id, caller_session_key=caller_session_key, caller_backend=caller_backend,
                approval_request_id=approval_request_id, fence=create_fence,
                state=JOB_QUIESCING if quiesce else JOB_QUEUED, receipt_callback_json=callback)
        except IdempotencyConflict as exc:
            raise OpsDispatchError("idempotency_conflict", str(exc)) from exc
        # A quiescing job returns at once; the reconciler's driver holds,
        # drains and dispatches it. Admission-only, every job does: the
        # reconciler, woken by the create notify, dispatches it (TASK-1169 S3b).
        if not self.admission_only:
            self._dispatch_queued(job)
        result = self.store.get_job(job.id).to_api()
        if op.executor_kind == "release":
            release = self.releases.get_by_parent_job(job.id)
            result["release_id"] = release.id if release else None
        return result

    def _gpu_voice_target(self, snapshot, *, live):
        executor = self._resolve_executor(snapshot["execution"]["executor_kind"])
        return gpu_voice_target(snapshot["spec"], snapshot["resolved_args"], executor, live=live)

    def _creation_gpu_target(self, snapshot):
        """The GPU voice service a job would start or recreate, for the creation
        guard. Prepare records it as ``gpu_target`` (TASK-1169 S3b), which must
        agree with whatever the spec alone proves. A snapshot from before that
        field is resolved from the spec; one whose target needs a Docker read
        refuses in a process without Docker, and reads it in the ops service."""
        try:
            from_spec = gpu_voice_target(snapshot["spec"], snapshot["resolved_args"], _NoTargetReads(),
                                         live=False)
            proven = True
        except GpuGuardUnverifiable:
            from_spec, proven = None, False
        if "gpu_target" in snapshot:
            recorded = snapshot["gpu_target"]
            if recorded is not None and recorded not in GPU_VOICE_SERVICES:
                raise GpuGuardUnverifiable(f"the snapshot records an unknown gpu_target {recorded!r}")
            if proven and recorded != from_spec:
                raise GpuGuardUnverifiable("the snapshot's gpu_target contradicts its spec")
            return recorded
        if proven:
            return from_spec
        if self.admission_only:
            raise GpuGuardUnverifiable("the snapshot records no gpu_target and its target needs a Docker read; "
                                       "prepare the operation again")
        return self._gpu_voice_target(snapshot, live=False)

    def _require_gpu_handoff(self, snapshot, create_fence):
        """Creation point of the GPU voice guard (TASK-1171): a job that would
        start or recreate a GPU voice service is created only under a fence the
        handoff controller built, in the job-insert transaction."""
        from ..media.gpu_handoff_store import GpuHandoffFence

        try:
            service = self._creation_gpu_target(snapshot)
        except GpuGuardUnverifiable as exc:
            raise OpsDispatchError("gpu_guard_unverifiable", str(exc)) from exc
        if service is not None and not isinstance(create_fence, GpuHandoffFence):
            raise OpsDispatchError("gpu_handoff_required", (
                f"this operation starts or recreates the GPU voice service {service!r}; "
                "only a GPU handoff can run it"))

    def _check_gpu_handoff(self, job, snapshot):
        """Dispatch point: re-run the guard against the live container and
        require the handoff row to still admit this job. Raises ExecutorError,
        so a refusal keeps the job queued with no worker, as any preflight."""
        from ..media.gpu_handoff_store import GpuHandoffStore

        try:
            service = self._gpu_voice_target(snapshot, live=True)
        except GpuGuardUnverifiable as exc:
            raise ExecutorError(f"gpu_guard_unverifiable: {exc}") from exc
        if service is not None and not GpuHandoffStore(self.store.engine).admits(
                job.idempotency_key, job.operation_slug):
            raise ExecutorError(f"gpu_handoff_required: no GPU handoff admits this {service!r} job")

    def _receipt_callback(self, snapshot, *, bot_id, user_id, session_id) -> str | None:
        """The receipt callback of a direct quiesce job whose scope covers its
        caller (TASK-1233 e). That caller is told to end its turn, so nothing
        else would ever start the turn that reads the result. An approved job
        needs none: its approval continuation delivers the receipt."""
        if not bot_id or not session_id:
            return None
        from .quiesce import scope_covers_bot

        with self.store.engine.connect() as conn:
            if not scope_covers_bot(conn, snapshot, bot_id):
                return None
        return canonical_json({"bot_id": bot_id, "user_id": user_id, "session_id": session_id})

    def _dispatch_queued(self, job):
        if job.state != JOB_QUEUED:
            return
        if not job.invocation_snapshot_json:
            self.store.mark_terminal(job.id, state=JOB_LOST, error_text="legacy job has no immutable execution snapshot; not replayed")
            return
        snapshot = json.loads(job.invocation_snapshot_json)
        if (snapshot.get("spec") or {}).get("action") in APPROVAL_ONLY_ACTIONS and not job.approval_request_id:
            # Defense in depth (TASK-1474 C-c): only an approved request creates one.
            self.store.mark_terminal(job.id, state=JOB_FAILED,
                error_text=f"approval_required: {APPROVAL_REQUIRED}; no action submitted")
            return
        if snapshot.get("quiesce"):
            # create_job never queues one; refuse visibly rather than run it unheld.
            self.store.mark_terminal(job.id, state=JOB_FAILED,
                error_text="quiesce job found queued; no action submitted")
            return
        current = self.store.get_operation_by_slug(job.operation_slug)
        if current is None or not current.enabled or current.soft_deleted_at is not None:
            self.store.mark_terminal(job.id, state=JOB_FAILED, error_text="operation disabled before claim; no action submitted")
            return
        executor = self._resolve_executor(snapshot["execution"]["executor_kind"])
        # One preflight outcome at a time per job: the latest pass's result is
        # the one noted, expired or claimed on (TASK-1169 S1e r1).
        with self.store.preflight_lock(job.id) as owned:
            current = self.store.get_job(job.id) if owned else None
            if current is None or current.state != JOB_QUEUED:
                return  # Another pass is preflighting it, or one already claimed it.
            try:
                executor.preflight(snapshot)
                self._check_gpu_handoff(job, snapshot)
            except ExecutorError as exc:
                # No target action can have been submitted before the claim. Keep
                # QUEUED for an operator fixing prerequisites, with a visible reason,
                # until the preflight deadline fails it (TASK-1169 S1e).
                self.store.note_queued_error(job.id, str(exc))
                delay = int(snapshot["execution"].get("start_delay_seconds") or 0)
                self.store.fail_stale_queued(job.id, deadline_seconds=preflight_deadline_seconds() + delay)
                return
            if not self.store.claim_job(job.id, max_concurrent=snapshot["execution"].get("max_concurrent")):
                return
        self._submit(job, snapshot, executor)

    def _submit(self, job, snapshot, executor, fence=None):
        """Submit a job already moved to DISPATCHING (claim or quiesce CAS)."""
        try:
            # No available() against *current* executor config here: execution
            # settings come exclusively from the persisted snapshot.
            result = executor.dispatch(job_id=job.id, snapshot=snapshot, fence=fence,
                operation_slug=job.operation_slug, command_script=snapshot["operation"]["command_script"],
                env_args=snapshot["resolved_args"], **{k: snapshot["execution"].get(k) for k in
                ("target_host", "run_as_user", "working_directory", "timeout_seconds", "start_delay_seconds", "max_output_bytes")})
        except ExecutorError as exc:
            # A lost Docker response can hide an accepted worker. Leave claimed
            # for reconciliation of the deterministic identity, NEVER resubmit.
            logger.warning("ops job %s dispatch uncertain: %s", job.id, exc)
            return
        self.store.mark_accepted(job.id, host_unit_name=result.host_unit_name,
            status_file_path=result.status_file_path, log_file_path=result.log_file_path)
        # terminal_state from dispatch is intentionally ignored. Only reconcile
        # of a persisted receipt can report successful completion.

    def _drive_quiesce(self, job):
        """One quiesce driver step (reconciler only); submit when it dispatches."""
        if self.quiesce_driver is None:
            return
        snapshot = json.loads(job.invocation_snapshot_json)
        executor = self._resolve_executor(snapshot["execution"]["executor_kind"])
        if job.quiesce_gate_id is None:
            try:
                executor.preflight(snapshot)
                self._check_gpu_handoff(job, snapshot)
            except QuiesceFenceUnsupported as exc:
                # The app no longer re-checks the fence at creation (TASK-1169
                # S3c): a worker that provably can't run fenced fails the job here.
                self.quiesce_driver.refuse_unopened(
                    job.id, f"quiesce_unavailable: {exc}; no hold opened, no action submitted")
                return
            except ExecutorError as exc:
                # Before any hold: nothing is held while prerequisites are
                # fixed, until the job's open deadline fails it.
                self.quiesce_driver.note_unopened(job.id, str(exc))
                return
        if self.quiesce_driver.step(job.id).dispatch:
            fresh = self.store.get_job(job.id)
            self._submit(fresh, snapshot, executor, fence=json.loads(fresh.quiesce_fence_json))

    def get_job_status(self, job_id, *, output_tail_bytes=None, reconcile_if_active=True):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        include_output = output_tail_bytes is not None and output_tail_bytes > 0
        # Admission-only, every status read is a read: the ops reconciler
        # dispatches and reconciles (TASK-1169 S3b).
        if job.state not in JOB_TERMINAL_STATES and reconcile_if_active and not self.admission_only:
            if job.state == JOB_QUEUED:
                self._dispatch_queued(job)
                job = self.store.get_job(job_id)
            elif job.state == JOB_QUIESCING:
                self._drive_quiesce(job)
            elif not job.invocation_snapshot_json:
                self.store.mark_terminal(job.id, state=JOB_LOST, error_text="legacy active job has no immutable snapshot; side effects unknown")
            else:
                snapshot = json.loads(job.invocation_snapshot_json)
                executor = self._resolve_executor(snapshot["execution"]["executor_kind"])
                # A dispatcher may be between claim and Docker create. A grace
                # interval prevents a concurrent status read declaring it lost.
                submitted = job.dispatched_at
                age = (datetime.now(timezone.utc) - submitted.replace(tzinfo=timezone.utc)).total_seconds() if submitted else 0
                if job.state != JOB_DISPATCHING or age >= 30:
                    try:
                        fence = json.loads(job.quiesce_fence_json) if job.quiesce_fence_json else None
                        rec = executor.reconcile(job_id=job.id, snapshot=snapshot, fence=fence,
                            target_host=snapshot["execution"].get("target_host", ""),
                            status_file_path=job.status_file_path or "", log_file_path=job.log_file_path or "",
                            output_tail_bytes=snapshot["execution"]["max_output_bytes"])
                        if rec.state == JOB_ACCEPTED:
                            self.store.mark_accepted(job.id)
                        elif rec.state == JOB_RUNNING:
                            self.store.mark_running(job.id, started_at=_timestamp(rec.started_at))
                        elif rec.state in JOB_TERMINAL_STATES:
                            self.store.mark_terminal(job.id, state=rec.state, exit_code=rec.exit_code,
                                output_tail=rec.output_tail, error_text=rec.error,
                                started_at=_timestamp(rec.started_at), finished_at=_timestamp(rec.finished_at))
                    except (ExecutorError, ValueError) as exc:
                        logger.warning("ops reconcile unavailable for %s: %s", job.id, exc)
                    self.store.touch_reconcile(job.id)
        fresh = self.store.get_job(job_id)
        result = attach_members(self.store.engine, [fresh.to_api(include_output=include_output)])[0]
        # When this status was read (TASK-1169 S1d); last_reconcile_at is when
        # the reconciler last looked at the job.
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        if include_output and result.get("output_tail"):
            result["output_tail"] = result["output_tail"].encode()[-min(output_tail_bytes, 1048576):].decode(errors="replace")
        return result

    def cancel_quiesce_job(self, job_id: str, *, actor: str, reason: str | None = None):
        """Operator cancel of a QUIESCING job before dispatch (TASK-1233 (f)).

        Returns ``(outcome, job_api)``: outcome is cancelled | already |
        not_cancellable | missing (job_api None). The cancelled job is terminal,
        so its receipt goes out on the normal paths (approval continuation, or
        the app's receipt loop for a direct job with a callback)."""
        outcome, _state = cancel_quiesce_job(self.store.engine, job_id, actor=actor, reason=reason)
        if outcome == "missing":
            return outcome, None
        job = self.store.get_job(job_id)
        return outcome, attach_members(self.store.engine, [job.to_api(include_output=False)])[0]

    def reconcile_active_jobs(self, *, limit=100, offset=0):
        """Recovery pump for independent reconciler, safe for multiple processes.

        Snapshot IDs before transitions; caller should cycle pages (or use the
        standalone reconciler) so long-running oldest jobs cannot starve others.
        """
        jobs = self.store.find_active_jobs(limit=limit, offset=offset)
        for job in jobs:
            self.get_job_status(job.id)
        if self.quiesce_driver is not None:
            self.quiesce_driver.release_terminal()
        return len(jobs)
