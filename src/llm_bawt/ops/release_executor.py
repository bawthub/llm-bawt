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
        """Freeze the source and production baseline for one end-to-end approval."""
        from .executor import validate_spec
        from .github_workflow import GitHubWorkflowError

        try:
            head = self.coordinator.gateway.resolve_branch_head(
                spec["github_repository"], spec["canonical_branch"])
            operation = self.coordinator.ops.store.get_operation_by_slug(spec["deploy_operation"])
            if operation is None or not operation.enabled or operation.soft_deleted_at is not None:
                raise ExecutorError("production deploy operation is unavailable")
            target = validate_spec(operation.command_script)
            if target.get("action") != "deploy_image":
                raise ExecutorError("release deploy operation does not target an image deploy")
            executor = self.coordinator.ops._resolve_executor(operation.executor_kind)
            current = executor.inspect_target_image(target)
            if not current or not current.startswith("sha256:") or len(current) != 71:
                raise ExecutorError("production image identity is unavailable")
            return {"expected_sha": head, "deploy_authorization": "after_verified_build",
                    "expected_current_image_id": current, "deploy_operation_id": operation.id,
                    "deploy_operation_version": operation.version,
                    "deploy_operation_script_hash": operation.script_hash}
        except GitHubWorkflowError as exc:
            raise ExecutorError(f"remote release source unavailable: {exc}") from exc
        except ValueError as exc:
            raise ExecutorError(f"release deploy target invalid: {exc}") from exc

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
