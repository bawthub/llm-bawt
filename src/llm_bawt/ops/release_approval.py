"""Exact-ID recovery of an existing, coordinator-owned release deploy gate.

This is read/decision validation only. It never prepares an invocation, creates a
job, grants a tool or resumes an agent. Listing and approver fanout remain MCP-only.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session, select

from ..approval_models import CONT_NOT_NEEDED, EXEC_NOT_APPLICABLE, KIND_ORCHESTRATION, REQ_PENDING
from ..approval_orchestration_store import ORCHESTRATION_GRANT_STATE
from .models import JOB_RUNNING, OpsJob
from .release_models import RELEASE_AWAITING_DEPLOY_APPROVAL, ReleaseRun
from .service import OpsDispatchError

_REQUEST_ID = re.compile(r"release-deploy-([0-9a-f]{32})")
DEPLOY_TOOL_NAME = "ops_release_deploy"


class InvalidReleaseApproval(ValueError):
    """The card is not the active release's exact durable deploy gate."""


@dataclass(frozen=True)
class ReleaseApproval:
    snapshot: dict
    operator_request: bool


def is_release_deploy_card(row) -> bool:
    """Candidate only; this shape alone confers no visibility or authority."""
    return (row is not None and row.request_kind == KIND_ORCHESTRATION
            and row.tool_name == DEPLOY_TOOL_NAME)


def _object(raw: str | None) -> dict:
    value = json.loads(raw or "null")
    if not isinstance(value, dict):
        raise InvalidReleaseApproval("missing or malformed stored object")
    return value


def validate_release_approval(row, *, ops, session: Session | None = None, lock: bool = True) -> ReleaseApproval:
    """Validate stored identity, ownership and snapshots without external effects.

    A decision passes its transaction's session, locking the release and parent
    until the audited decision commits. The normal GET is a read-only check.
    ``_verify_snapshot`` checks content hashes, schema/spec/args and the verified
    release binding; unlike ``prepare_invocation`` it performs no target reads.
    """
    if session is None:
        with Session(ops.store.engine) as reader:
            return validate_release_approval(row, ops=ops, session=reader, lock=False)
    try:
        match = _REQUEST_ID.fullmatch(row.id) if is_release_deploy_card(row) else None
        if match is None:
            raise InvalidReleaseApproval("not a deterministic release deploy card")
        release_id = match[1]
        release_query = select(ReleaseRun).where(ReleaseRun.id == release_id)
        release = session.exec(release_query.with_for_update() if lock else release_query).first()
        if (release is None or release.state != RELEASE_AWAITING_DEPLOY_APPROVAL
                or release.deploy_approval_request_id != row.id or release.deploy_job_id
                or release.id != hashlib.sha256(
                    f"bawthub-release:{release.parent_job_id}".encode()).hexdigest()[:32]):
            raise InvalidReleaseApproval("card is not the current release deploy gate")
        deadline = release.phase_deadline_at
        if deadline is None:
            raise InvalidReleaseApproval("release approval has no deadline")
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if row.status == REQ_PENDING and deadline <= datetime.now(timezone.utc):
            raise InvalidReleaseApproval("release approval window elapsed")
        parent_query = select(OpsJob).where(OpsJob.id == release.parent_job_id)
        parent = session.exec(parent_query.with_for_update() if lock else parent_query).first()
        if parent is None or parent.operation_slug != "bawthub.release-prod" or parent.state != JOB_RUNNING:
            raise InvalidReleaseApproval("release parent is missing, foreign or inactive")
        operator = parent.caller_backend == "http-operator" and parent.caller_bot_id is None
        if not operator and not parent.caller_bot_id:
            raise InvalidReleaseApproval("release has no stored requester")
        expected_owner = (parent.caller_bot_id or "ops", parent.caller_user_id or "operator",
                          parent.caller_turn_id or f"release:{release.id}",
                          parent.caller_backend or "ops-release", parent.caller_session_key or None)
        if ((row.bot_id, row.user_id, row.turn_id, row.backend, row.session_key) != expected_owner
                or row.continuation_owner != "none" or row.grant_state != ORCHESTRATION_GRANT_STATE
                or row.execution_state != EXEC_NOT_APPLICABLE or row.continuation_state != CONT_NOT_NEEDED
                or row.continuation_capable or row.execution_claim_token or row.continuation_claim_token
                or row.result_json is not None
                or row.grant_key != hashlib.sha256(f"release-deploy:{release.id}".encode()).hexdigest()
                or _object(row.caller_context_json) != {
                    "release_run_id": release.id, "parent_job_id": parent.id,
                    "release_task": release.release_task}):
            raise InvalidReleaseApproval("release requester or caller linkage differs")
        parent_snapshot = _object(parent.invocation_snapshot_json)
        parent_spec = parent_snapshot["spec"]
        parent_operation = parent_snapshot["operation"]
        if (parent_spec["action"] != "release_orchestrate"
                or parent_spec["github_repository"] != release.github_repository
                or parent_spec["workflow_path"] != release.workflow_path
                or parent_spec["canonical_branch"] != release.canonical_branch
                or parent_operation["slug"] != parent.operation_slug
                or parent_operation["version"] != parent.operation_version
                or parent_operation["script_hash"] != parent.operation_script_hash
                or parent_snapshot["resolved_args"] != _object(parent.args_json)
                or parent_snapshot["resolved_args"]["release_task"] != release.release_task
                or parent_snapshot["resolved_args"].get("bump", "patch") != release.bump):
            raise InvalidReleaseApproval("parent release snapshot differs")
        ops._verify_snapshot(parent_snapshot, parent.operation_slug, parent_snapshot["input_args"])
        plan = _object(release.release_plan_json)
        slug = parent_spec["deploy_operation"]
        if (plan["deploy_operation"] != slug or plan["image_repository"] != release.image_repository
                or plan["image_repository"] != parent_spec["image_repository"]
                or plan["expected_sha"] != parent_snapshot["release_source"]["expected_sha"]):
            raise InvalidReleaseApproval("stored release plan differs from parent")
        args = {"release_run_id": release.id}
        if _object(row.tool_arguments_json) != {"operation": slug, "args": args, "release_run_id": release.id}:
            raise InvalidReleaseApproval("deploy card arguments differ")
        snapshot = _object(row.operations_snapshot_json)
        if (snapshot["spec"]["action"] != "deploy_image"
                or snapshot["spec"].get("binding") == "ci_publish"
                or snapshot["release"]["release_run_id"] != release.id):
            raise InvalidReleaseApproval("not a verified release-run deploy snapshot")
        # Also checks the active, durable receipt through ReleaseStore. Do not
        # re-prepare: the existing snapshot is exactly what the approver decides.
        snapshot = ops._verify_snapshot(snapshot, slug, args)
        binding = ops.releases.verified_binding(release.id)
        if binding["trigger_sha"] != plan["expected_sha"]:
            raise InvalidReleaseApproval("verified receipt differs from the approved source")
        for key in ("workflow_run_id", "workflow_run_attempt", "source_sha", "version",
                    "tag", "digest", "image_repository", "trigger_sha", "image_ref"):
            if str(snapshot["release"].get(key)) != str(binding.get(key)):
                raise InvalidReleaseApproval(f"stored deploy binding differs: {key}")
        for key in ("github_repository", "workflow_path", "image_repository"):
            if snapshot["spec"].get(key) != binding[key]:
                raise InvalidReleaseApproval(f"stored deploy target differs: {key}")
        return ReleaseApproval(snapshot=snapshot, operator_request=operator)
    except (TypeError, KeyError, ValueError, AttributeError, OpsDispatchError) as exc:
        if isinstance(exc, InvalidReleaseApproval):
            raise
        raise InvalidReleaseApproval("malformed or unverified release deploy card") from exc
