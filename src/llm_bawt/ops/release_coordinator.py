"""Durable BawtHub release state machine (TASK-1030).

One ``bawthub.release-prod`` ops job owns one ``ops_release_runs`` row. Every
status read of that job (MCP ``ops_job_status``, the continuation outbox, the
HTTP API, the independent ops reconciler) calls :meth:`ReleaseCoordinator.pump`
through the release executor. A pump is cheap when nothing is due
(``next_check_at`` throttle) and claim-fenced when something is, so any number
of readers in any number of processes advance exactly one release safely.

Invariants (each has a regression test in tests/test_release_coordinator.py):

* Preflight resolves *remote* branch heads only. A bot's or echo's local clone
  is never consulted (the v0.1.62 incident).
* Intent is recorded before every external side effect (dispatch, re-run,
  approval request, child deploy). A crash replays reads, never mutations: a
  workflow is dispatched at most once per release, a partial run is recovered
  with "Re-run failed jobs" on the SAME run (no second version bump), and the
  child deploy is keyed by a deterministic idempotency key.
* An optional (``llm_bawt_mode=auto``) llm-bawt tag failure yields a deployable
  ``complete_with_warning`` receipt, never a stranded image.
* Deploy is a separate human decision: a server-owned orchestration approval
  carrying the verified deploy snapshot. The first decision wins.
* Anything ambiguous ends in ``lost_requires_inspection``; nothing is retried
  blindly past its phase deadline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from ..approval_models import REQ_APPROVED, REQ_EXPIRED, REQ_PENDING
from .executor import ExecutorError, ReconcileResult
from .github_workflow import GitHubWorkflowError, GitHubWorkflowGateway
from .image_deploy import parse_deployment
from .models import (JOB_CANCELLED, JOB_FAILED, JOB_LOST, JOB_QUEUED, JOB_RUNNING, JOB_SUCCEEDED,
                     JOB_TERMINAL_STATES)
from .release import DEPLOYABLE_RECEIPT_STATUSES, RECEIPT_SCHEMA
from .release_models import (RELEASE_AWAITING_DEPLOY_APPROVAL, RELEASE_BUILD_COMPLETE,
                             RELEASE_BUILD_FAILED_SAFE, RELEASE_BUILD_PARTIAL, RELEASE_BUILDING,
                             RELEASE_DEPLOY_DECLINED, RELEASE_DEPLOY_FAILED_RESTORED, RELEASE_DEPLOYED,
                             RELEASE_DEPLOYING, RELEASE_DISPATCHING_BUILD, RELEASE_LOST,
                             RELEASE_PREFLIGHT, RELEASE_TERMINAL_STATES, ReleaseRun)
from .release_spec import workflow_file
from .release_store import ReleaseStore

logger = logging.getLogger(__name__)

RELEASE_OUTPUT_SCHEMA = "llm-bawt.ops.release/v1"
DEPLOY_TOOL_NAME = "ops_release_deploy"

PREFLIGHT_WINDOW = timedelta(minutes=15)
CORRELATION_WINDOW = timedelta(minutes=5)
CORRELATION_SLACK = timedelta(seconds=60)
BUILD_WINDOW = timedelta(hours=8)
RECEIPT_GRACE = timedelta(minutes=2)
RERUN_REGISTRATION_WINDOW = timedelta(minutes=10)
DEPLOY_PREP_WINDOW = timedelta(minutes=30)
DEPLOY_APPROVAL_WINDOW = timedelta(hours=24)
MAX_RERUNS = 2
CLAIM_LEASE_SECONDS = 120
MAX_STEPS_PER_PUMP = 6

POLL_CORRELATE = 5
POLL_BUILD = 20
POLL_APPROVAL = 5
POLL_DEPLOY = 5
BACKOFF_TRANSIENT = 30
BACKOFF_DEPLOY_PREP = 60

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
# Definitive refusals from OpsService.dispatch_job: nothing was created.
_DEPLOY_REFUSALS = {"operation_disabled", "operation_not_found", "snapshot_invalid", "args_invalid"}
_BINDING_KEYS = ("workflow_run_id", "workflow_run_attempt", "source_sha", "version", "tag", "digest",
                 "image_repository")

_STATE_MESSAGES = {
    RELEASE_PREFLIGHT: "resolving remote branch heads",
    RELEASE_DISPATCHING_BUILD: "release workflow dispatched; correlating the GitHub run",
    RELEASE_BUILDING: "GitHub release workflow running",
    RELEASE_BUILD_PARTIAL: "release run partial; re-running failed jobs on the same run",
    RELEASE_BUILD_COMPLETE: "build verified; preparing the deploy approval",
    RELEASE_AWAITING_DEPLOY_APPROVAL: "build verified; waiting for the deploy approval in BawtHub Approvals",
    RELEASE_DEPLOYING: "deploy approved; production image deploy running",
    RELEASE_DEPLOYED: "deployed and verified",
    RELEASE_BUILD_FAILED_SAFE: "release failed before anything was pushed; a new release is safe",
    RELEASE_DEPLOY_FAILED_RESTORED: "deploy failed; production remains on the previous image",
    RELEASE_DEPLOY_DECLINED: "build verified but not deployed",
    RELEASE_LOST: "outcome uncertain; inspect the run before any new release",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _parse_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return _aware(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError:
        return None


def _iso(value: datetime | None) -> str | None:
    value = _aware(value)
    return value.isoformat() if value else None


def _loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class _ClaimLost(Exception):
    """Another pump owns the release now; stop without touching it."""


class ReleaseCoordinator:
    """Advances release rows; every step is a claim-fenced CAS transition."""

    def __init__(
        self,
        *,
        releases: ReleaseStore,
        ops: Any,
        gateway: GitHubWorkflowGateway,
        approvals: Callable[[], Any],
        publisher: Callable[[dict], None] | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.releases = releases
        self.ops = ops
        self.gateway = gateway
        self._approvals = approvals
        self._publisher = publisher
        self._clock = clock
        self._handlers = {
            RELEASE_PREFLIGHT: self._preflight,
            RELEASE_DISPATCHING_BUILD: self._correlate,
            RELEASE_BUILDING: self._building,
            RELEASE_BUILD_PARTIAL: self._partial,
            RELEASE_BUILD_COMPLETE: self._request_deploy,
            RELEASE_AWAITING_DEPLOY_APPROVAL: self._await_approval,
            RELEASE_DEPLOYING: self._deploying,
        }

    # ── entry points ─────────────────────────────────────────────────────
    def create_for_job(self, job_id: str, snapshot: dict) -> ReleaseRun:
        """Idempotent; no remote side effect (safe to repeat on recovery)."""
        spec, args = snapshot["spec"], snapshot["resolved_args"]
        return self.releases.create(
            parent_job_id=job_id,
            release_task=args["release_task"],
            bump=args.get("bump", "patch"),
            llm_bawt_mode=args.get("llm_bawt_mode", "auto"),
            github_repository=spec["github_repository"],
            workflow_path=spec["workflow_path"],
            canonical_branch=spec["canonical_branch"],
            llm_bawt_repository=spec["llm_bawt_repository"],
            llm_bawt_branch=spec["llm_bawt_branch"],
        )

    def pump(self, job_id: str, snapshot: dict) -> ReconcileResult:
        row = self.releases.get_by_parent_job(job_id) or self.create_for_job(job_id, snapshot)
        if row.state in RELEASE_TERMINAL_STATES:
            return self.result(row)
        now = self._clock()
        due = _aware(row.next_check_at)
        if due is not None and due > now:
            return self.result(row)
        claimed = self.releases.claim(row.id, lease_seconds=CLAIM_LEASE_SECONDS)
        if claimed is None:
            return self.result(self.releases.get(row.id) or row)
        token = claimed.claim_token
        delay: float = BACKOFF_TRANSIENT
        row = claimed
        owned = True
        try:
            for _ in range(MAX_STEPS_PER_PUMP):
                delay = self._step(row, token, snapshot)
                fresh = self.releases.get(row.id)
                if fresh is None or fresh.state in RELEASE_TERMINAL_STATES or delay > 0:
                    break
                if not self.releases.renew_claim(row.id, token, lease_seconds=CLAIM_LEASE_SECONDS):
                    owned = False
                    break
                row = fresh
        except _ClaimLost:
            owned = False
        finally:
            if owned:
                self.releases.release_claim(
                    row.id, token, next_check_at=self._clock() + timedelta(seconds=max(delay, 0))
                )
        return self.result(self.releases.get(row.id) or row)

    def _step(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        handler = self._handlers.get(row.state)
        if handler is None:
            return BACKOFF_TRANSIENT
        try:
            return handler(row, token, snapshot)
        except GitHubWorkflowError as exc:
            # Reads after dispatch: GitHub being unreachable/forbidden does not
            # change what already happened remotely. Retry until the phase
            # deadline, then declare the outcome unknown.
            deadline = _aware(row.phase_deadline_at)
            if deadline is not None and self._clock() >= deadline:
                return self._end(row, token, RELEASE_LOST, f"github_{exc.code}"[:64], str(exc))
            self._record_error(row, token, exc.code, str(exc))
            return BACKOFF_TRANSIENT

    # ── preflight → dispatch ─────────────────────────────────────────────
    def _preflight(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        now = self._clock()
        within = now < _aware(row.created_at) + PREFLIGHT_WINDOW
        try:
            expected_sha = self.gateway.resolve_branch_head(row.github_repository, row.canonical_branch)
        except GitHubWorkflowError as exc:
            if exc.retryable and within:
                self._record_error(row, token, exc.code, str(exc))
                return BACKOFF_TRANSIENT
            return self._end(row, token, RELEASE_BUILD_FAILED_SAFE, f"preflight_{exc.code}"[:64], str(exc))
        approved = snapshot.get("release_source") or {}
        if expected_sha != approved.get("expected_sha"):
            return self._end(row, token, RELEASE_BUILD_FAILED_SAFE, "approved_source_moved",
                             "BawtHub remote branch moved after build approval; request a fresh approval")
        mode, llm_sha, warnings = row.llm_bawt_mode, None, []
        if mode != "off":
            try:
                llm_sha = self.gateway.resolve_branch_head(row.llm_bawt_repository, row.llm_bawt_branch)
            except GitHubWorkflowError as exc:
                if exc.retryable and within:
                    self._record_error(row, token, exc.code, str(exc))
                    return BACKOFF_TRANSIENT
                return self._end(row, token, RELEASE_BUILD_FAILED_SAFE,
                                 f"preflight_llm_bawt_{exc.code}"[:64], str(exc))
            if llm_sha != approved.get("llm_bawt_expected_sha"):
                return self._end(row, token, RELEASE_BUILD_FAILED_SAFE, "approved_llm_bawt_source_moved",
                                 "llm-bawt remote branch moved after build approval; request a fresh approval")
        inputs = {
            "expected_sha": expected_sha,
            "bump": row.bump,
            "release_task": row.release_task,
            "llm_bawt_mode": mode,
            "llm_bawt_expected_sha": llm_sha or "",
            "release_request_id": row.release_request_id,
        }
        spec = snapshot["spec"]
        plan = {
            "expected_sha": expected_sha,
            "llm_bawt_expected_sha": llm_sha,
            "inputs": inputs,
            "warnings": warnings,
            "image_repository": spec["image_repository"],
            "deploy_operation": spec["deploy_operation"],
            "workflow_file": workflow_file(row.workflow_path),
        }
        # Intent first: from here on a crash correlates, it never redispatches.
        row = self._move(row, token, RELEASE_DISPATCHING_BUILD, "release.dispatch_intent",
                         detail={"expected_sha": expected_sha, "llm_bawt_expected_sha": llm_sha,
                                 "llm_bawt_mode": mode, "warnings": warnings},
                         llm_bawt_sha=llm_sha, release_plan_json=plan, dispatch_started_at=now,
                         phase_deadline_at=now + CORRELATION_WINDOW, error_code=None, error_text=None)
        try:
            self.gateway.dispatch_workflow(row.github_repository, plan["workflow_file"],
                                           row.canonical_branch, inputs)
        except GitHubWorkflowError as exc:
            if not exc.retryable:
                # GitHub refused the dispatch (4xx / no credential): nothing started.
                return self._end(row, token, RELEASE_BUILD_FAILED_SAFE, f"dispatch_{exc.code}"[:64], str(exc))
            self._note(row, token, "release.dispatch_uncertain", {"code": exc.code, "error": str(exc)},
                       error_code="dispatch_uncertain", error_text=str(exc))
            return POLL_CORRELATE
        self._note(row, token, "release.dispatched", {"inputs": inputs})
        return POLL_CORRELATE

    def _correlate(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        plan = _loads(row.release_plan_json)
        started = _aware(row.dispatch_started_at) or _aware(row.created_at)
        runs = self.gateway.find_correlated_runs(
            row.github_repository, plan.get("workflow_file") or workflow_file(row.workflow_path),
            row.canonical_branch, row.release_request_id, created_after=started - CORRELATION_SLACK)
        if len(runs) > 1:
            return self._end(row, token, RELEASE_LOST, "correlation_ambiguous",
                             f"{len(runs)} workflow runs carry {row.release_request_id}",
                             detail={"run_ids": [str(r.get("id")) for r in runs][:10]})
        if len(runs) == 1:
            run = runs[0]
            self._move(row, token, RELEASE_BUILDING, "release.run_correlated",
                       detail={"run_id": str(run.get("id")), "url": run.get("html_url")},
                       github_run_id=str(run["id"]), github_run_attempt=int(run.get("run_attempt") or 1),
                       github_run_url=run.get("html_url") or None, phase_deadline_at=None,
                       error_code=None, error_text=None)
            return 0
        if self._clock() >= _aware(row.phase_deadline_at):
            return self._end(row, token, RELEASE_LOST, "correlation_timeout",
                             "no workflow run carrying this release id appeared after dispatch")
        return POLL_CORRELATE

    # ── building / partial ───────────────────────────────────────────────
    def _building(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        now = self._clock()
        run = self.gateway.get_run(row.github_repository, row.github_run_id)
        attempt = int(run.get("run_attempt") or 1)
        expected = int(row.github_run_attempt or 1)
        if attempt < expected:
            return self._await_rerun(row, token, run)
        if attempt > expected:
            row = self._note(row, token, "release.attempt_adopted",
                             {"from": expected, "to": attempt}, github_run_attempt=attempt)
        if run.get("status") != "completed":
            run_started = _parse_time(run.get("run_started_at")) or _parse_time(run.get("created_at"))
            if run_started is not None and now - run_started > BUILD_WINDOW:
                return self._end(row, token, RELEASE_LOST, "build_timeout",
                                 f"run attempt {attempt} not completed within {BUILD_WINDOW}")
            return POLL_BUILD
        conclusion = run.get("conclusion")
        try:
            receipt = self.gateway.download_release_receipt(row.github_repository, row.github_run_id, attempt)
        except GitHubWorkflowError as exc:
            if exc.retryable:
                raise
            if exc.code != "receipt_missing":
                return self._end(row, token, RELEASE_LOST, exc.code[:64], str(exc))
            completed = _parse_time(run.get("updated_at"))
            if completed is not None and now - completed < RECEIPT_GRACE:
                return POLL_BUILD  # artifact listing can lag a just-completed run
            if conclusion == "success":
                return self._end(row, token, RELEASE_LOST, "receipt_missing",
                                 "successful run published no release receipt")
            # The receipt job never ran: a same-run re-run can still finish it.
            return self._to_partial(row, token, conclusion, "no receipt for this attempt")
        plan = _loads(row.release_plan_json)
        problem = self._receipt_identity_problem(row, plan, receipt, attempt)
        if problem:
            return self._end(row, token, RELEASE_LOST, "receipt_identity", problem)
        status = receipt.get("status")
        if status in DEPLOYABLE_RECEIPT_STATUSES:
            if conclusion != "success":
                return self._end(row, token, RELEASE_LOST, "receipt_conclusion_mismatch",
                                 f"receipt {status} but run concluded {conclusion}")
            problem = self._receipt_structure_problem(receipt, plan)
            if problem:
                return self._end(row, token, RELEASE_LOST, "receipt_invalid", problem)
            warnings = [str(w) for w in [*(plan.get("warnings") or []), *(receipt.get("warnings") or [])] if w]
            self._move(row, token, RELEASE_BUILD_COMPLETE, "release.build_verified",
                       detail={"status": status, "version": receipt["version"], "digest": receipt["digest"],
                               "warnings": warnings[:10]},
                       source_sha=receipt["source_sha"], version=receipt["version"], tag=receipt["tag"],
                       digest=receipt["digest"], image_repository=receipt["image_repository"],
                       receipt_json=receipt, receipt_verified_at=now, deployable=True,
                       warning_text="; ".join(warnings[:10]) or None,
                       phase_deadline_at=now + DEPLOY_PREP_WINDOW, error_code=None, error_text=None)
            return 0
        if status == "partial":
            return self._to_partial(row, token, conclusion, "receipt status partial", receipt=receipt)
        if status == "failed_safe":
            return self._end(row, token, RELEASE_BUILD_FAILED_SAFE, "build_failed_safe",
                             "release workflow failed before anything was pushed", receipt_json=receipt)
        return self._end(row, token, RELEASE_LOST, "receipt_status_unknown", f"receipt status {status!r}")

    def _await_rerun(self, row: ReleaseRun, token: str, run: dict) -> float:
        """A requested re-run has not registered its new attempt yet."""
        if self._clock() >= _aware(row.phase_deadline_at):
            return self._end(row, token, RELEASE_LOST, "rerun_not_registered",
                             f"attempt {row.github_run_attempt} never appeared on run {row.github_run_id}")
        # An HTTP timeout can conceal an accepted re-run, and GitHub's run
        # attempt counter may lag the request. Never repeat the mutation merely
        # because the old attempt is still visible; time out to inspection.
        return POLL_BUILD

    def _to_partial(self, row, token, conclusion, reason, *, receipt=None) -> float:
        values = {"receipt_json": receipt} if receipt is not None else {}
        self._move(row, token, RELEASE_BUILD_PARTIAL, "release.build_partial",
                   detail={"conclusion": conclusion, "reason": reason, "attempt": row.github_run_attempt},
                   error_code="build_partial", error_text=reason, **values)
        return 0

    def _partial(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        if int(row.rerun_count or 0) >= MAX_RERUNS:
            return self._end(row, token, RELEASE_LOST, "partial_rerun_exhausted",
                             f"run {row.github_run_id} still partial after {row.rerun_count} re-runs")
        now = self._clock()
        row = self._move(row, token, RELEASE_BUILDING, "release.rerun_intent",
                         detail={"attempt": int(row.github_run_attempt or 1) + 1},
                         rerun_count=int(row.rerun_count or 0) + 1,
                         github_run_attempt=int(row.github_run_attempt or 1) + 1,
                         phase_deadline_at=now + RERUN_REGISTRATION_WINDOW)
        return self._request_rerun(row, token)

    def _request_rerun(self, row: ReleaseRun, token: str) -> float:
        try:
            self.gateway.rerun_failed_jobs(row.github_repository, row.github_run_id)
        except GitHubWorkflowError as exc:
            if not exc.retryable:
                return self._end(row, token, RELEASE_LOST, "rerun_refused", str(exc))
            self._note(row, token, "release.rerun_uncertain", {"code": exc.code, "error": str(exc)},
                       error_code="rerun_uncertain", error_text=str(exc))
            return BACKOFF_DEPLOY_PREP
        self._note(row, token, "release.rerun_requested", {"attempt": row.github_run_attempt},
                   error_code=None, error_text=None)
        return POLL_BUILD

    def _receipt_identity_problem(self, row, plan, receipt, attempt) -> str | None:
        expected = {
            "schema": RECEIPT_SCHEMA,
            "repository": row.github_repository,
            "workflow_run_id": str(row.github_run_id),
            "workflow_run_attempt": str(attempt),
            "release_request_id": row.release_request_id,
        }
        for key, want in expected.items():
            got = receipt.get(key)
            got = str(got) if key in ("workflow_run_id", "workflow_run_attempt") and got is not None else got
            if got != want:
                return f"receipt {key} is {got!r}, expected {want!r}"
        base = receipt.get("base_sha")
        # A preflight that failed (e.g. main moved) never published base_sha.
        if base != plan.get("expected_sha") and not (receipt.get("status") == "failed_safe" and not base):
            return f"receipt base_sha is {base!r}, expected {plan.get('expected_sha')!r}"
        return None

    @staticmethod
    def _receipt_structure_problem(receipt: dict, plan: dict) -> str | None:
        version, digest = str(receipt.get("version") or ""), str(receipt.get("digest") or "")
        if not _VERSION.fullmatch(version):
            return f"version {version!r} is not semver"
        if receipt.get("tag") != f"v{version}":
            return f"tag {receipt.get('tag')!r} does not match version"
        if not _DIGEST.fullmatch(digest):
            return "digest is not sha256:<64 hex>"
        if not _SHA.fullmatch(str(receipt.get("source_sha") or "")):
            return "source_sha is not a full SHA"
        if receipt.get("image_repository") != plan.get("image_repository"):
            return f"image_repository {receipt.get('image_repository')!r} is not the pinned repository"
        ref = receipt.get("image_ref")
        if ref and ref != f"{plan.get('image_repository')}@{digest}":
            return "image_ref does not match repository@digest"
        mode = (plan.get("inputs") or {}).get("llm_bawt_mode")
        if mode and receipt.get("llm_bawt_mode") != mode:
            return f"receipt llm_bawt_mode {receipt.get('llm_bawt_mode')!r}, dispatched {mode!r}"
        return None

    # ── deploy approval ──────────────────────────────────────────────────
    def _deploy_args(self, row: ReleaseRun) -> dict:
        return {"release_run_id": row.id}

    @staticmethod
    def _binding_problem(snapshot: dict, binding: dict) -> str | None:
        release = snapshot.get("release") or {}
        for key in _BINDING_KEYS:
            if str(release.get(key) or "") != str(binding.get(key) or ""):
                return f"deploy snapshot {key} {release.get(key)!r} != verified release {binding.get(key)!r}"
        return None

    def _request_deploy(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        from .service import OpsDispatchError

        plan = _loads(row.release_plan_json)
        slug = plan.get("deploy_operation") or snapshot["spec"]["deploy_operation"]
        try:
            binding = self.releases.verified_binding(row.id)
        except ValueError as exc:
            return self._end(row, token, RELEASE_LOST, "binding_invalid", str(exc))
        approvals = self._approvals()
        request_id = f"release-deploy-{row.id}"
        stored = approvals.get_request(request_id)
        if stored is not None:
            deploy_snapshot, created = _loads(stored.operations_snapshot_json), False
        else:
            args = self._deploy_args(row)
            try:
                deploy_snapshot = self.ops.prepare_invocation(slug, args)
            except OpsDispatchError as exc:
                if exc.code in ("operation_disabled", "operation_not_found"):
                    return self._end(row, token, RELEASE_DEPLOY_DECLINED, "deploy_interlocked", str(exc))
                if self._clock() >= _aware(row.phase_deadline_at):
                    return self._end(row, token, RELEASE_DEPLOY_DECLINED, "deploy_unverifiable", str(exc))
                self._record_error(row, token, exc.code, str(exc))
                return BACKOFF_DEPLOY_PREP
            problem = self._binding_problem(deploy_snapshot, binding)
            if problem:
                return self._end(row, token, RELEASE_LOST, "deploy_binding_mismatch", problem)
            parent = self.ops.store.get_job(row.parent_job_id)
            stored, created = approvals.record_orchestration_request(
                request_id=request_id,
                bot_id=(parent and parent.caller_bot_id) or "ops",
                user_id=(parent and parent.caller_user_id) or "operator",
                turn_id=(parent and parent.caller_turn_id) or f"release:{row.id}",
                backend=(parent and parent.caller_backend) or "ops-release",
                tool_name=DEPLOY_TOOL_NAME,
                tool_arguments={"operation": slug, "args": args, "release_run_id": row.id},
                subject=f"operation={slug} version={row.version} digest={row.digest}",
                grant_key=hashlib.sha256(f"release-deploy:{row.id}".encode()).hexdigest(),
                severity="high",
                prompt=self._approval_prompt(row, slug, deploy_snapshot),
                operations_snapshot=deploy_snapshot,
                caller_context={"release_run_id": row.id, "parent_job_id": row.parent_job_id,
                                "release_task": row.release_task},
                session_key=parent.caller_session_key if parent else None,
            )
            deploy_snapshot = _loads(stored.operations_snapshot_json)
        # The stored request is authoritative: it is exactly what a human approves.
        problem = self._binding_problem(deploy_snapshot, binding)
        if problem or deploy_snapshot.get("operation", {}).get("slug") != slug:
            return self._end(row, token, RELEASE_LOST, "deploy_binding_mismatch",
                             problem or "stored deploy request targets another operation")
        now = self._clock()
        self._move(row, token, RELEASE_AWAITING_DEPLOY_APPROVAL, "release.deploy_approval_requested",
                   detail={"approval_request_id": request_id, "created": created},
                   deploy_approval_request_id=request_id, phase_deadline_at=now + DEPLOY_APPROVAL_WINDOW,
                   error_code=None, error_text=None)
        parent = self.ops.store.get_job(row.parent_job_id)
        if created and parent is not None and parent.caller_bot_id and parent.caller_user_id:
            self._publish_approval(stored)
        return POLL_APPROVAL

    @staticmethod
    def _approval_prompt(row: ReleaseRun, slug: str, snapshot: dict) -> str:
        release = snapshot.get("release") or {}
        lines = [
            f"Deploy BawtHub {row.version} to production ({slug}) for {row.release_task}.",
            f"Image: {row.image_repository}@{row.digest}",
            f"Source: {row.source_sha} (tag {row.tag})",
            f"Build: {row.github_run_url or row.github_run_id} (attempt {row.github_run_attempt})",
            f"Replaces running image: {release.get('expected_current_image_id', 'unknown')}",
        ]
        if row.warning_text:
            lines.append(f"Warnings: {row.warning_text}")
        lines.append("The previous container is restored automatically if health or release checks fail.")
        return "\n".join(lines)

    def _publish_approval(self, request) -> None:
        payload = {
            "_type": "tool_approval_required",
            "request_id": request.id,
            "tool_use_id": None,
            "turn_id": request.turn_id,
            "trigger_message_id": None,
            "bot_id": request.bot_id,
            "user_id": request.user_id,
            "tool_name": request.tool_name,
            "arguments": _loads(request.tool_arguments_json),
            "subject": request.subject,
            "label": "Deploy BawtHub release",
            "prompt": request.prompt,
            "severity": request.severity,
            "policy_id": None,
            "session_key": request.session_key or "",
            "provider": request.backend,
            "request_kind": request.request_kind,
            "continuation_capable": False,
        }
        try:
            if self._publisher is not None:
                self._publisher(payload)
        except Exception:  # noqa: BLE001 - the DB row is authoritative; fanout is best-effort
            logger.warning("release deploy approval committed but live publish failed id=%s",
                           request.id, exc_info=True)

    def _await_approval(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        approvals = self._approvals()
        request = approvals.get_request(row.deploy_approval_request_id)
        if request is None:
            return self._end(row, token, RELEASE_LOST, "approval_missing",
                             f"deploy approval {row.deploy_approval_request_id} no longer exists")
        if request.status == REQ_PENDING and self._clock() >= _aware(row.phase_deadline_at):
            request = approvals.resolve_request(
                request.id, status=REQ_EXPIRED, resolved_by="system:release-timeout",
                message="deploy approval window elapsed", continuation_owner="none") or request
        if request.status == REQ_PENDING:
            return POLL_APPROVAL
        if request.status == REQ_APPROVED:
            now = self._clock()
            self._move(row, token, RELEASE_DEPLOYING, "release.deploy_approved",
                       detail={"resolved_by": request.resolved_by},
                       phase_deadline_at=now + DEPLOY_PREP_WINDOW)
            return 0
        return self._end(row, token, RELEASE_DEPLOY_DECLINED, f"deploy_{request.status}"[:64],
                         request.resolution_message or f"deploy approval {request.status}",
                         detail={"resolved_by": request.resolved_by})

    # ── deploy ───────────────────────────────────────────────────────────
    def _deploying(self, row: ReleaseRun, token: str, snapshot: dict) -> float:
        from .service import OpsDispatchError

        request = self._approvals().get_request(row.deploy_approval_request_id)
        if request is None or request.status != REQ_APPROVED:
            return self._end(row, token, RELEASE_LOST, "approval_missing", "approved deploy request vanished")
        approved = _loads(request.operations_snapshot_json)
        key = f"release-deploy:{row.id}"
        store = self.ops.store
        job = store.get_job(row.deploy_job_id) if row.deploy_job_id else store.get_job_by_key(key)
        if job is None:
            parent = store.get_job(row.parent_job_id)
            try:
                self.ops.dispatch_job(
                    operation_slug=approved["operation"]["slug"], args=approved["input_args"],
                    idempotency_key=key, approved_snapshot=approved, caller_actor=f"release:{row.id}",
                    caller_bot_id=parent.caller_bot_id if parent else None,
                    caller_user_id=parent.caller_user_id if parent else None,
                    caller_turn_id=parent.caller_turn_id if parent else None,
                    caller_session_key=parent.caller_session_key if parent else None,
                    caller_backend="ops-release", approval_request_id=request.id)
            except OpsDispatchError as exc:
                if store.get_job_by_key(key) is None:
                    if exc.code in _DEPLOY_REFUSALS:
                        return self._end(row, token, RELEASE_DEPLOY_DECLINED, f"deploy_{exc.code}"[:64], str(exc))
                    if exc.code == "idempotency_conflict":
                        return self._end(row, token, RELEASE_LOST, "deploy_key_conflict", str(exc))
                    if self._clock() >= _aware(row.phase_deadline_at):
                        return self._end(row, token, RELEASE_LOST, "deploy_dispatch_failed", str(exc))
                    self._record_error(row, token, exc.code, str(exc))
                    return BACKOFF_TRANSIENT
            job = store.get_job_by_key(key)
            if job is None:
                return BACKOFF_TRANSIENT
        if row.deploy_job_id != job.id:
            row = self._note(row, token, "release.deploy_dispatched", {"deploy_job_id": job.id},
                             deploy_job_id=job.id, error_code=None, error_text=None)
        if job.state not in JOB_TERMINAL_STATES:
            self.ops.get_job_status(job.id, reconcile_if_active=True)
            job = store.get_job(job.id)
        if job.state not in JOB_TERMINAL_STATES:
            if job.state == JOB_QUEUED and job.error_text and job.error_text != row.error_text:
                self._note(row, token, "release.deploy_queued", {"error": job.error_text},
                           error_code="deploy_queued", error_text=job.error_text)
            return POLL_DEPLOY
        return self._finish_deploy(row, token, job, approved)

    def _finish_deploy(self, row: ReleaseRun, token: str, job, approved: dict) -> float:
        record = parse_deployment(job.output_tail)
        if job.state == JOB_SUCCEEDED and job.exit_code == 0:
            problem = self._deployment_problem(row, record, approved)
            if problem == "":
                if self._clock() >= _aware(row.phase_deadline_at):
                    return self._end(row, token, RELEASE_LOST, "deploy_inspection_timeout",
                                     "deploy succeeded but the production image could not be inspected")
                return BACKOFF_TRANSIENT  # target not inspectable right now; verify again
            if problem:
                return self._end(row, token, RELEASE_LOST, "deploy_verification_failed", problem)
            return self._end(row, token, RELEASE_DEPLOYED, None, None,
                             detail={"deploy_job_id": job.id, "image_id": record["deployed"]["image_id"]})
        if job.state in (JOB_FAILED, JOB_CANCELLED) and job.dispatched_at is None:
            return self._end(row, token, RELEASE_DEPLOY_DECLINED, "deploy_interlocked",
                             job.error_text or "deploy job never dispatched")
        if job.state == JOB_FAILED:
            # The fixed worker reports failure only for a known outcome, but
            # "restored" is true ONLY if it verified the old target healthy.
            if not record or record.get("action") != "deploy":
                return self._end(row, token, RELEASE_LOST, "deploy_receipt_missing",
                                 "failed deploy has no validated worker outcome")
            restored = record.get("restored") is True
            if restored:
                return self._end(row, token, RELEASE_DEPLOY_FAILED_RESTORED, "deploy_restored",
                                 record.get("error") or job.error_text or "deploy failed",
                                 detail={"deploy_job_id": job.id, "restored": True})
            if record.get("phase") not in {"init", "inspect", "verify-image", "create"}:
                return self._end(row, token, RELEASE_LOST, "deploy_outcome_unknown",
                                 "deploy failed after production may have changed without verified restoration")
            return self._end(row, token, RELEASE_DEPLOY_DECLINED, "deploy_refused_unchanged",
                             record.get("error") or job.error_text or "deploy refused before target mutation",
                             detail={"deploy_job_id": job.id, "restored": False})
        return self._end(row, token, RELEASE_LOST, f"deploy_{job.state}"[:64],
                         job.error_text or f"deploy job {job.state}", detail={"deploy_job_id": job.id})

    def _deployment_problem(self, row: ReleaseRun, record: dict | None, approved: dict) -> str | None:
        """None = verified; "" = retry later; text = verification failure."""
        if not record or record.get("phase") != "done" or record.get("action") != "deploy":
            return "deploy job has no complete deployment record"
        deployed = record.get("deployed") or {}
        if deployed.get("image_ref") != f"{row.image_repository}@{row.digest}":
            return "deployed image reference differs from the verified digest"
        release = deployed.get("release") or {}
        expected = {"version": row.version, "source_sha": row.source_sha,
                    "workflow_run_id": str(row.github_run_id)}
        for key, want in expected.items():
            if str(release.get(key) or "") != want:
                return f"deployed {key} {release.get(key)!r}, expected {want!r}"
        health = record.get("health") or {}
        baked = health.get("release") or {}
        if (health.get("status") != "healthy" or baked.get("version") != row.version
                or baked.get("sourceSha") != row.source_sha
                or str(baked.get("workflowRunId")) != str(row.github_run_id)):
            return "worker did not attest healthy production with the approved baked release identity"
        try:
            executor = self.ops._resolve_executor(approved["execution"]["executor_kind"])
            current = executor.inspect_target_image(approved["spec"])
            health_now = executor.inspect_target_health(approved["spec"])
        except ExecutorError:
            return ""
        if current != deployed.get("image_id"):
            return f"target runs {current}, deploy installed {deployed.get('image_id')}"
        if health_now != "healthy":
            return f"target health is {health_now!r} after the worker completed"
        return None

    # ── persistence helpers ──────────────────────────────────────────────
    def _move(self, row, token, to_state, event, *, detail=None, **values) -> ReleaseRun:
        moved = self.releases.transition(row.id, from_states=row.state, to_state=to_state, claim_token=token,
                                         event_type=event, detail=detail, **values)
        if moved is None:
            raise _ClaimLost()
        if to_state != row.state and self._publisher is not None:
            try:
                parent = self.ops.store.get_job(row.parent_job_id)
                if parent and parent.caller_bot_id and parent.caller_user_id:
                    self._publisher({"_type": "ops_release_transition", "bot_id": parent.caller_bot_id,
                                     "user_id": parent.caller_user_id, "release_id": moved.id,
                                     "parent_job_id": moved.parent_job_id, "state": moved.state,
                                     "event_seq": moved.event_seq, "at": _iso(moved.updated_at)})
            except Exception:  # noqa: BLE001 - append-only DB event is canonical
                logger.warning("release event persisted but live publish failed id=%s", moved.id,
                               exc_info=True)
        return moved

    def _note(self, row, token, event, detail=None, **values) -> ReleaseRun:
        return self._move(row, token, row.state, event, detail=detail, **values)

    def _record_error(self, row, token, code, text) -> None:
        """One event per distinct error, not one per retry."""
        if row.error_code != code or row.error_text != text:
            self._note(row, token, "release.retrying", {"code": code, "error": text},
                       error_code=str(code)[:64], error_text=text)

    def _end(self, row, token, state, code, text, *, detail=None, **values) -> float:
        info = {"code": code, "error": text, **(detail or {})}
        self._move(row, token, state, f"release.{state}", detail=info, error_code=code, error_text=text,
                   next_check_at=None, **values)
        return 0

    # ── job-facing result ────────────────────────────────────────────────
    @staticmethod
    def summary(row: ReleaseRun) -> dict:
        return {
            "schema": RELEASE_OUTPUT_SCHEMA,
            "release_id": row.id,
            "release_task": row.release_task,
            "state": row.state,
            "message": _STATE_MESSAGES.get(row.state, row.state),
            "version": row.version,
            "tag": row.tag,
            "digest": row.digest,
            "image_repository": row.image_repository,
            "source_sha": row.source_sha,
            "llm_bawt_sha": row.llm_bawt_sha,
            "github_run_id": row.github_run_id,
            "github_run_attempt": row.github_run_attempt,
            "github_run_url": row.github_run_url,
            "rerun_count": int(row.rerun_count or 0),
            "warning": row.warning_text,
            "deploy_approval_request_id": row.deploy_approval_request_id,
            "deploy_job_id": row.deploy_job_id,
            "error_code": row.error_code,
            "error_text": row.error_text,
        }

    def result(self, row: ReleaseRun) -> ReconcileResult:
        output = json.dumps(self.summary(row), sort_keys=True)
        started, finished = _iso(row.created_at), _iso(row.finished_at)
        if row.state == RELEASE_DEPLOYED:
            return ReconcileResult(JOB_SUCCEEDED, 0, output, None, started, finished)
        if row.state in (RELEASE_BUILD_FAILED_SAFE, RELEASE_DEPLOY_FAILED_RESTORED):
            return ReconcileResult(JOB_FAILED, 1, output, row.error_text or row.state, started, finished)
        if row.state == RELEASE_DEPLOY_DECLINED:
            return ReconcileResult(JOB_CANCELLED, None, output, row.error_text or row.state, started, finished)
        if row.state == RELEASE_LOST:
            return ReconcileResult(JOB_LOST, None, output, row.error_text or row.state, started, finished)
        return ReconcileResult(JOB_RUNNING, None, output, None, started, None)


__all__ = ["ReleaseCoordinator", "RELEASE_OUTPUT_SCHEMA", "DEPLOY_TOOL_NAME"]
