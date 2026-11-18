"""The real rootless control listener persists cards, never executes requests."""
from __future__ import annotations

import httpx
import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
from llm_bawt.ops import OpsDispatchError, OpsService
from llm_bawt.ops.control_client import OpsControlClient
from llm_bawt.ops.control_http import OpsControlServer
from llm_bawt.ops.control_protocol import PROTOCOL_FIELD, REQUEST_PATH
from llm_bawt.ops.invocation_attestation import mint
from tests.test_ops_invocation_attestation import CALLER, rootless  # noqa: F401
from tests.test_ops_service import _make
from tests.test_ops_store import _op_data, _store


@pytest.mark.usefixtures("rootless")
def test_rootless_request_creates_durable_card_and_retry_never_reprepares():
    store = _store()
    operation = store.create_operation(_op_data())
    service, executor = _make(store)
    service.approval_store = ToolApprovalPolicyStore(None, engine=store.engine)
    server = OpsControlServer(lambda: service, ("127.0.0.1", 0))
    server.start()
    try:
        client = OpsControlClient("http://127.0.0.1:%d" % server.address[1])
        app = OpsService(store, admission_only=True, preparer=client)
        arguments = {"operation": operation.slug, "args": {}}
        envelope = mint(arguments, CALLER, canonical_invocation_hash("ops_run", arguments))
        result = app.request_invocation(envelope)
        assert result["status"] == "approval_required" and result["created"]
        row = service.approval_store.get_request(result["approval_request_id"])
        assert row.prompt == result["card"]["prompt"] and row.policy_id is None
        store.update_operation(operation.id, {"enabled": False})
        retry = app.request_invocation(envelope)
        assert retry["approval_request_id"] == row.id and not retry["created"]
        assert not store.list_jobs() and not executor.dispatch_calls
    finally:
        server.close()


@pytest.mark.parametrize("status", [404, 500, 504])
def test_request_protocol_failure_preserves_unknown_write_outcome(status):
    client = OpsControlClient("http://ops.invalid", token_source=lambda: "fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(
            status, json={PROTOCOL_FIELD: 1, "code": "ops_timeout"})))
    with pytest.raises(OpsDispatchError) as error:
        client.request_invocation({})
    assert error.value.code == "ops_request_unknown"
    assert "may exist" in str(error.value) and "nothing was created" not in str(error.value)


def test_request_read_timeout_is_not_retried():
    calls = []

    def timeout(request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout("synthetic", request=request)

    client = OpsControlClient("http://ops.invalid", token_source=lambda: "fixture",
                              transport=httpx.MockTransport(timeout))
    with pytest.raises(OpsDispatchError) as error:
        client.request_invocation({})
    assert error.value.code == "ops_request_unknown" and calls == [REQUEST_PATH]
