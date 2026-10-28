"""Actual adapter policy/spec checks without external services."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
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


STARTED = "2026-10-28T06:00:00.123456+00:00"


def _instance(**overrides):
    instance = {"instanceId": "c" * 64, "name": "bawthub-tts-1", "state": "running",
                "health": "healthy", "startedAt": STARTED, "restartCount": 0}
    instance.update(overrides)
    return instance


def _body(name, instances, **overrides):
    body = {"schemaVersion": 1, "observedAt": STARTED, "fresh": True, "reason": None,
            "unlabeledCount": 0,
            "service": {"id": f"bawthub.{name}", "project": "bawthub", "service": name,
                        "instances": instances, "capabilities": []}}
    body.update(overrides)
    return body


class _Http:
    """The reconciler's /v1/health/services/{id}, one canned answer per service."""

    def __init__(self, answers):
        self.answers, self.urls = answers, []

    async def get(self, url, timeout):
        self.urls.append(url)
        answer = self.answers[url.rsplit(".", 1)[1]]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        if isinstance(body, str):
            return httpx.Response(status, content=body.encode())
        return httpx.Response(status, json=body)


def _observe(answers):
    http = _Http(answers)
    adapter = LocalHandoffAdapter(config=SimpleNamespace(OPS_HEALTH_URL="http://ops:8650/"),
                                  http=http, video=None, ops=None, policies=None)
    return asyncio.run(adapter._services()), http


def _healthy(service):
    return 200, _body(service, [_instance(instanceId=service * 32)])


def test_one_fresh_instance_per_service_is_the_observation():
    services, http = _observe({"stt": _healthy("stt"), "tts": _healthy("tts")})
    assert http.urls == ["http://ops:8650/v1/health/services/bawthub.stt",
                         "http://ops:8650/v1/health/services/bawthub.tts"]
    assert services["stt"] == {"id": "stt" * 32, "running": True, "status": "running",
                               "health": "healthy", "started_at": STARTED, "restart_count": 0}
    assert services["tts"]["id"] == "tts" * 32


def test_exited_and_created_instances_are_observed_as_not_running():
    for state in ("exited", "created"):
        answer = (200, _body("tts", [_instance(state=state, health=None, restartCount=3)]))
        services, _ = _observe({"stt": _healthy("stt"), "tts": answer})
        assert services["tts"]["running"] is False and services["tts"]["status"] == state
        assert services["tts"]["restart_count"] == 3


def _refuses(tts_answer, match):
    with pytest.raises(HandoffConflict, match=match):
        _observe({"stt": _healthy("stt"), "tts": tts_answer})


@pytest.mark.parametrize("instances, match", [
    ([], "0 instances"),
    ([_instance(), _instance(instanceId="d" * 64)], "2 instances"),
    (None, "no current inventory"),
    ({"instanceId": "x"}, "malformed"),
])
def test_never_picks_an_instance(instances, match):
    _refuses((200, _body("tts", instances)), match)


@pytest.mark.parametrize("answer, match", [
    ((503, _body("tts", None, fresh=False, reason="observation stale")), "not fresh: observation stale"),
    ((200, _body("tts", [_instance()], fresh=False)), "not fresh"),
    ((503, _body("tts", None, service=None, fresh=False)), "not fresh"),
    ((404, {"error": "not found"}), "not a catalogued"),
    ((200, _body("tts", [_instance()], schemaVersion=2)), "unsupported schema"),
    ((200, "not json"), "malformed"),
    ((200, ["a", "list"]), "unsupported schema"),
    ((200, _body("stt", [_instance()])), "another service"),
    (httpx.ConnectError("refused"), "unavailable"),
    (httpx.ReadTimeout("slow"), "unavailable"),
])
def test_unavailable_or_foreign_answers_refuse(answer, match):
    _refuses(answer, match)


@pytest.mark.parametrize("field, value", [
    ("instanceId", None), ("instanceId", 7),
    ("state", "running-ish"), ("state", None),
    ("health", "fine"), ("health", True),
    ("startedAt", "yesterday"), ("startedAt", "2026-10-28T06:00:00"), ("startedAt", 5),
    ("restartCount", -1), ("restartCount", True), ("restartCount", 1.0), ("restartCount", "0"),
])
def test_ill_typed_instance_fields_refuse(field, value):
    _refuses((200, _body("tts", [_instance(**{field: value})])), "malformed")


def test_missing_health_key_refuses():
    instance = _instance()
    del instance["health"]
    _refuses((200, _body("tts", [instance])), "malformed")


def test_absent_restart_count_and_start_are_observed_as_unknown():
    services, _ = _observe({"stt": _healthy("stt"), "tts": (200, _body(
        "tts", [_instance(restartCount=None, startedAt=None)]))})
    assert services["tts"]["restart_count"] is None and services["tts"]["started_at"] is None
