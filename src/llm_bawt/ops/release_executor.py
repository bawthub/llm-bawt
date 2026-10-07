"""Release executor and application/reconciler wiring for TASK-1030.

Dispatch records an intent only. The independent reconciler or a status read
advances the same durable state machine; no thread or local checkout is needed.
"""
from __future__ import annotations

import json
import logging

from .executor import DispatchResult, Executor, ExecutorError, ReconcileResult
from .github_workflow import HttpGitHubWorkflowGateway
from .release_coordinator import ReleaseCoordinator
from .release_spec import RELEASE_EXECUTOR_KIND
from .release_store import ReleaseStore
from .service import OpsService
from .store import OpsStore

logger = logging.getLogger(__name__)


class ReleaseExecutor(Executor):
    def __init__(self, coordinator: ReleaseCoordinator):
        self.coordinator = coordinator

    def kind(self) -> str:
        return RELEASE_EXECUTOR_KIND

    def available(self) -> bool:
        return self.coordinator.releases.engine is not None

    def check_target(self, spec: dict, args: dict) -> None:
        slug = spec["deploy_operation"]
        op = self.coordinator.ops.store.get_operation_by_slug(slug)
        if op is None or not op.enabled or op.soft_deleted_at is not None:
            raise ExecutorError(f"deploy operation {slug} is unavailable or disabled")

    def approval_source(self, spec: dict, args: dict) -> dict:
        """Freeze remote heads before the build approval is shown."""
        from .github_workflow import GitHubWorkflowError

        try:
            gateway = self.coordinator.gateway
            source = {"expected_sha": gateway.resolve_branch_head(
                spec["github_repository"], spec["canonical_branch"]),
                "llm_bawt_expected_sha": None}
            if args["llm_bawt_mode"] != "off":
                source["llm_bawt_expected_sha"] = gateway.resolve_branch_head(
                    spec["llm_bawt_repository"], spec["llm_bawt_branch"])
            return source
        except GitHubWorkflowError as exc:
            raise ExecutorError(f"remote release source unavailable: {exc}") from exc

    def dispatch(self, **kwargs) -> DispatchResult:
        try:
            row = self.coordinator.create_for_job(kwargs["job_id"], kwargs["snapshot"])
        except Exception as exc:
            raise ExecutorError(f"cannot persist release intent: {exc}") from exc
        return DispatchResult(host_unit_name=f"release:{row.id}")

    def reconcile(self, **kwargs) -> ReconcileResult:
        try:
            return self.coordinator.pump(kwargs["job_id"], kwargs["snapshot"])
        except Exception as exc:
            logger.exception("release reconcile failed for job %s", kwargs.get("job_id"))
            raise ExecutorError(f"release reconciliation unavailable: {exc}") from exc


def _redis_approval_publisher(redis_url: str):
    """Best-effort live card; the persisted approval row remains authoritative."""
    def publish(payload: dict) -> None:
        if not redis_url:
            return
        import redis
        from agent_bridge.publisher import UNIFIED_EVENTS_PREFIX, UNIFIED_STREAM_MAXLEN

        client = redis.Redis.from_url(redis_url, decode_responses=True,
                                      socket_timeout=1.0, socket_connect_timeout=1.0)
        try:
            client.xadd(f"{UNIFIED_EVENTS_PREFIX}{payload['bot_id']}:{payload['user_id']}",
                        {"payload": json.dumps(payload, ensure_ascii=False)},
                        maxlen=UNIFIED_STREAM_MAXLEN, approximate=True)
        finally:
            client.close()
    return publish


def build_ops_service(config, store: OpsStore | None = None, *, gateway=None,
                      approval_store=None, publisher=None) -> OpsService:
    """Register release execution in both the app and the independent pump."""
    from ..service.dependencies import get_tool_approval_policy_store

    service = OpsService(store or OpsStore(config))
    releases = ReleaseStore(config)
    coordinator = ReleaseCoordinator(
        releases=releases,
        ops=service,
        gateway=gateway or HttpGitHubWorkflowGateway(),
        approvals=lambda: approval_store or get_tool_approval_policy_store(config),
        publisher=publisher if publisher is not None else _redis_approval_publisher(config.REDIS_URL),
    )
    service.register_executor(ReleaseExecutor(coordinator))
    return service


__all__ = ["ReleaseExecutor", "build_ops_service"]
