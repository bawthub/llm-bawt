"""Prompt-registry seeds for approver notices (TASK-1661).

Seed data only: registered in :mod:`llm_bawt.prompt_definitions` and resolved
per approver bot at send time (``approval_fanout.notice_text``), so Nick can
edit them in BawtHub's Prompt Registry.
"""
from __future__ import annotations

DEFAULT_APPROVER_REQUEST = (
    "[Approval requested: {requested_by} asks to run the ops operation {operation} "
    "(\"{title}\", risk {risk_level}, severity {severity}) with arguments {args} "
    "(sensitive values masked). Request id: {request_id}.]\n"
    "You hold the approve_ops grant, so you may decide it. Read it with "
    "ops_approvals_get(request_id) and approve or deny it with ops_approval_resolve, "
    "giving a reason. Decide only on what you can verify; if it is not yours to "
    "judge, leave it for the user. Someone else may decide first."
)
DEFAULT_APPROVER_RESOLVED = (
    "[Approval decided: request {request_id} from {requested_by} to run "
    "{operation} is now {status}, decided by {resolved_by}.]\n"
    "Nothing is left for you to decide on it. Drop any pending work about deciding it."
)

__all__ = ["DEFAULT_APPROVER_REQUEST", "DEFAULT_APPROVER_RESOLVED"]
