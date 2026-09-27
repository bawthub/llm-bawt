"""Actual adapter policy/spec checks without external services."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent_bridge.approval import ApprovalPolicy, PolicyAction
from llm_bawt.media.gpu_handoff import STOP_OPERATIONS
from llm_bawt.media.gpu_handoff_adapter import LocalHandoffAdapter
from llm_bawt.media.gpu_handoff_store import HandoffConflict


class Ops:
    def __init__(self):
        self.calls = []
        self.rows = {slug: {
            "operation": {"slug": slug}, "resolved_args": {}, "input_args": {},
            "spec": {"action": "stop", "compose_project": "bawthub", "compose_service": service},
            "execution": {"executor_kind": "docker", "target_host": ""},
            "snapshot_hash": service,
        } for slug, service in zip(STOP_OPERATIONS, ("stt", "tts"), strict=True)}

    def prepare_invocation(self, slug, args):
        assert args == {}
        return deepcopy(self.rows[slug])

    def dispatch_job(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "job"}


def make_adapter(action=PolicyAction.REQUIRE_APPROVAL):
    ops = Ops()
    bundle = SimpleNamespace(policies=[ApprovalPolicy(id="ops", tool_name="ops_run", action=action)])
    policies = SimpleNamespace(compile_bundle=lambda: bundle)
    return LocalHandoffAdapter(config=None, http=None, video=None, ops=ops, policies=policies), ops


def test_snapshot_dispatch_is_scoped_and_idempotent():
    adapter, ops = make_adapter()

    async def run():
        snapshots = await adapter.prepare_operations()
        await adapter.check_operations(snapshots)
        assert not ops.calls  # Preparing an offer never means dispatching.
        await adapter.dispatch(snapshots[0], user="nick", generation=2)
        assert ops.calls[0]["approved_snapshot"] == snapshots[0]
        assert ops.calls[0]["operation_slug"] == STOP_OPERATIONS[0]
        assert ops.calls[0]["args"] == {}
        assert ops.calls[0]["idempotency_key"] == f"gpu-handoff:2:{STOP_OPERATIONS[0]}"
        assert ops.calls[0]["caller_user_id"] == "nick"
        assert ops.calls[0]["caller_backend"] == "http-operator"
    asyncio.run(run())


def test_policy_deny_blocks_scoped_confirmation():
    adapter, ops = make_adapter(PolicyAction.DENY)

    async def run():
        with pytest.raises(HandoffConflict, match="denied"):
            await adapter.check_operations(await adapter.prepare_operations())
    asyncio.run(run())
    assert not ops.calls


@pytest.mark.parametrize("change", ["action", "service", "host", "args", "revision"])
def test_changed_or_broadened_catalog_requires_fresh_offer(change):
    adapter, ops = make_adapter()

    async def run():
        snapshots = await adapter.prepare_operations()
        if change == "revision":
            ops.rows[STOP_OPERATIONS[0]]["snapshot_hash"] = "changed"
        if change == "action":
            snapshots[0]["spec"]["action"] = "restart"
        if change == "service":
            snapshots[0]["spec"]["compose_service"] = "backend"
        if change == "host":
            snapshots[0]["execution"]["target_host"] = "other"
        if change == "args":
            snapshots[0]["resolved_args"] = {"container": "other"}
        with pytest.raises(HandoffConflict):
            await adapter.check_operations(snapshots)
    asyncio.run(run())
    assert not ops.calls


class _Container:
    def __init__(self, service):
        self.id = f"{service}-id"
        self.attrs = {"State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}}

    def reload(self):
        pass


class _PlainDockerClient:
    """Mirrors docker-py 7.x: no __enter__/__exit__, only close()."""

    def __init__(self):
        self.closed = False
        self.containers = SimpleNamespace(list=self._list)

    def _list(self, all, filters):
        service = filters["label"][1].split("=", 1)[1]
        return [_Container(service)]

    def close(self):
        self.closed = True


def test_service_observation_works_with_non_context_manager_client():
    client = _PlainDockerClient()
    adapter = LocalHandoffAdapter(config=None, http=None, video=None, ops=None, policies=None,
                                  docker_factory=lambda: client)
    services = adapter._services()
    assert services["stt"] == {"id": "stt-id", "running": True, "status": "running", "health": "healthy"}
    assert services["tts"]["id"] == "tts-id"
    assert client.closed
