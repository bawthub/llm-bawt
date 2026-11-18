"""Compatibility facade for approval policies, models, and request lifecycle.

Implementation is split along model/policy-store/request-store seams.
Existing callers may continue importing public symbols from this module.
"""
from agent_bridge.approval import PolicyBundle as PolicyBundle, compute_etag as compute_etag
from .approval_models import *  # noqa: F403
from .approval_models import _new_id, _utcnow, _as_aware_utc  # noqa: F401
from .approval_policy_store import ToolApprovalPolicyStore as ToolApprovalPolicyStore
from .approval_defaults import _DEFAULT_POLICIES, _OPS_DEFAULT_POLICIES  # noqa: F401
