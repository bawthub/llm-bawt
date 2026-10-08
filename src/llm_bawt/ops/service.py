"""Catalog validation, immutable approval snapshots and durable job orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from .executor import DockerExecutor, Executor, ExecutorError, validate_spec
from .image_deploy import DEPLOY_ACTION, IMAGE_ACTIONS
from .release_spec import is_release_spec
from .release_store import ReleaseStore
from .release import (GitHubReleaseVerifier, ReleaseVerificationError, ReleaseVerifier,
                      bind_image_invocation, binding_matches_args)
from .models import (JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, JOB_LOST, JOB_QUEUED,
                     JOB_RUNNING, JOB_TERMINAL_STATES, OpsOperation)
from .store import IdempotencyConflict, OpsStore
from .validation import ArgValidationError, canonical_json, validate_args

logger = logging.getLogger(__name__)


class OpsDispatchError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _display_args(args, schema_json):
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


class OpsService:
    def __init__(self, store: OpsStore, *, executor: Executor | None = None,
                 release_verifier: ReleaseVerifier | None = None,
                 release_store: ReleaseStore | None = None):
        self.store = store
        self.releases = release_store or ReleaseStore(None, engine=store.engine)
        self._release_verifier = release_verifier or GitHubReleaseVerifier()
        default = executor or DockerExecutor()
        self._executors = {default.kind(): default}

    def register_executor(self, executor):
        self._executors[executor.kind()] = executor

    def _resolve_executor(self, kind):
        executor = self._executors.get(kind)
        if executor is None:
            raise OpsDispatchError("executor_kind_unknown", f"no executor registered for {kind!r}")
        return executor

    def list_operations_for_agent(self, *, include_disabled=False):
        return [row.to_agent_summary() for row in self.store.list_operations(include_disabled=include_disabled)]

    def get_operation(self, slug):
        return self.store.get_operation_by_slug(slug)

    def prepare_invocation(self, operation_slug: str, args: dict | None) -> dict:
        """Trusted approval integration: persist this JSON BEFORE asking approval.

        Detached JSON snapshot; callers must never accept it from tool arguments.
        dispatch_job verifies the snapshot and original input, then executes this
        exact revision even if defaults/spec/config have subsequently changed.
        Current enabled/deleted state remains an emergency execution interlock.
        """
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
        if spec["action"] in IMAGE_ACTIONS:
            # TASK-997: the approver sees and the worker executes ONE verified
            # release (or recorded last-known-good) plus an image CAS.
            try:
                snapshot.update(bind_image_invocation(spec=spec, args=merged, store=self.store,
                                                      releases=self.releases, executor=executor,
                                                      verifier=self._release_verifier))
            except (ReleaseVerificationError, ExecutorError) as exc:
                raise OpsDispatchError("release_unverified", str(exc)) from exc
        else:
            # TASK-1002: never ask a human to approve a container this
            # executor's Docker daemon does not have.
            try:
                executor.check_target(spec, merged)
                if is_release_spec(spec):
                    snapshot["release_source"] = executor.approval_source(spec, merged)
            except ExecutorError as exc:
                raise OpsDispatchError("target_unavailable", str(exc)) from exc
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
            if not binding_matches_args(detached):
                raise ValueError("snapshot release/rollback binding mismatch")
            if is_release_spec(detached["spec"]):
                source = detached.get("release_source") or {}
                if (set(source) != {"expected_sha"}
                        or not isinstance(source["expected_sha"], str)
                        or len(source["expected_sha"]) != 40
                        or any(char not in "0123456789abcdef" for char in source["expected_sha"])):
                    raise ValueError("snapshot approved release source is invalid")
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
                     caller_session_key=None, caller_backend=None, approval_request_id=None) -> dict:
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
        snapshot = snapshot or self.prepare_invocation(operation_slug, supplied)
        current = self.store.get_operation_by_slug(operation_slug)
        if current is None or not current.enabled or current.soft_deleted_at is not None:
            raise OpsDispatchError("operation_disabled", "operation is currently disabled or deleted")
        if current.id != snapshot["operation"]["id"]:
            raise OpsDispatchError("snapshot_invalid", "operation identity changed")
        op = OpsOperation(**{k: v for k, v in snapshot["operation"].items()
                             if k not in {"created_at", "updated_at", "soft_deleted_at"}})
        try:
            job = self.store.create_job(operation=op, args_json=canonical_json(snapshot["resolved_args"]),
                display_args_json=canonical_json(_display_args(snapshot["resolved_args"], op.args_schema_json)),
                idempotency_key=key, invocation_snapshot_json=canonical_json(snapshot), request_payload_json=payload,
                caller_actor=caller_actor, caller_bot_id=caller_bot_id, caller_user_id=caller_user_id,
                caller_turn_id=caller_turn_id, caller_session_key=caller_session_key, caller_backend=caller_backend,
                approval_request_id=approval_request_id)
        except IdempotencyConflict as exc:
            raise OpsDispatchError("idempotency_conflict", str(exc)) from exc
        self._dispatch_queued(job)
        result = self.store.get_job(job.id).to_api()
        if op.executor_kind == "release":
            release = self.releases.get_by_parent_job(job.id)
            result["release_id"] = release.id if release else None
        return result

    def _dispatch_queued(self, job):
        if job.state != JOB_QUEUED:
            return
        if not job.invocation_snapshot_json:
            self.store.mark_terminal(job.id, state=JOB_LOST, error_text="legacy job has no immutable execution snapshot; not replayed")
            return
        snapshot = json.loads(job.invocation_snapshot_json)
        current = self.store.get_operation_by_slug(job.operation_slug)
        if current is None or not current.enabled or current.soft_deleted_at is not None:
            self.store.mark_terminal(job.id, state=JOB_FAILED, error_text="operation disabled before claim; no action submitted")
            return
        executor = self._resolve_executor(snapshot["execution"]["executor_kind"])
        try:
            executor.preflight(snapshot)
        except ExecutorError as exc:
            # No side effect can have been submitted by preflight. Keep QUEUED
            # for an operator fixing prerequisites, with a visible reason.
            self.store.note_queued_error(job.id, str(exc))
            return
        if not self.store.claim_job(job.id, max_concurrent=snapshot["execution"].get("max_concurrent")):
            return
        try:
            # No available() against *current* executor config here: execution
            # settings come exclusively from the persisted snapshot.
            result = executor.dispatch(job_id=job.id, snapshot=snapshot,
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

    def get_job_status(self, job_id, *, output_tail_bytes=None, reconcile_if_active=True):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        include_output = output_tail_bytes is not None and output_tail_bytes > 0
        if job.state not in JOB_TERMINAL_STATES and reconcile_if_active:
            if job.state == JOB_QUEUED:
                self._dispatch_queued(job)
                job = self.store.get_job(job_id)
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
                        rec = executor.reconcile(job_id=job.id, snapshot=snapshot,
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
        result = fresh.to_api(include_output=include_output)
        if include_output and result.get("output_tail"):
            result["output_tail"] = result["output_tail"].encode()[-min(output_tail_bytes, 1048576):].decode(errors="replace")
        return result

    def reconcile_active_jobs(self, *, limit=100, offset=0):
        """Recovery pump for independent reconciler, safe for multiple processes.

        Snapshot IDs before transitions; caller should cycle pages (or use the
        standalone reconciler) so long-running oldest jobs cannot starve others.
        """
        jobs = self.store.find_active_jobs(limit=limit, offset=offset)
        for job in jobs:
            self.get_job_status(job.id)
        return len(jobs)
