"""Pinned credential gateway forwards only the active run's scoped token request."""

import json
from io import BytesIO
from unittest.mock import Mock

import httpx
import pytest

from llm_bawt.preview import credential_gateway as gw
from llm_bawt.service import preview_credential as broker

NOW = 2_000_000_000
RUN = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(broker.crypto, "derive_key", lambda _: b"fixture-key")
    grant = broker.PreviewGrant(RUN, "claude", 7, NOW + 1800)
    path = tmp_path / "capability.json"
    config = gw.GatewayConfig(path, "http://broker.test:8642", "claude", RUN)

    def publish(run=grant, now=NOW):
        path.write_text(json.dumps({"run_id": run.run_id, "provider": run.provider,
                                   "endpoint_id": run.endpoint_id, "expires_at": run.expires_at,
                                   "capability": broker.issue_capability(run, now=now)}))

    publish()
    return config, path, grant, publish


def test_forward_fixed_endpoint_and_no_caller_headers(gateway):
    config, _, _, _ = gateway
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"access_token": "fixture-token", "state": "ok",
                                          "expires_at": (NOW + 1200) * 1000})

    body = gw.fetch_credential(config, now=NOW, transport=httpx.MockTransport(respond))
    assert json.loads(body) == {"access_token": "fixture-token", "state": "ok",
                                "expires_at": (NOW + 1200) * 1000}
    assert len(requests) == 1
    assert str(requests[0].url) == "http://broker.test:8642/v1/preview/credential"
    assert requests[0].headers["X-Preview-Broker-Token"]
    assert "X-Bridge-Token" not in requests[0].headers
    assert "Authorization" not in requests[0].headers


@pytest.mark.parametrize("damage", ["remove", "bad_json", "wrong_run", "wrong_provider",
                                   "wrong_endpoint", "expired", "bad_capability"])
def test_bad_or_revoked_capability_never_reaches_broker(gateway, damage):
    config, path, _, _ = gateway
    data = json.loads(path.read_text())
    if damage == "remove":
        path.unlink()
    elif damage == "bad_json":
        path.write_text("[")
    elif damage == "wrong_run":
        data["run_id"] = "00000000-0000-4000-8000-000000000002"
    elif damage == "wrong_provider":
        data["provider"] = "openai_chatgpt"
    elif damage == "wrong_endpoint":
        data["endpoint_id"] = 8
    elif damage == "expired":
        data["expires_at"] = NOW - 1
    else:
        data["capability"] = "invalid"
    if damage not in {"remove", "bad_json"}:
        path.write_text(json.dumps(data))
    transport = httpx.MockTransport(Mock(side_effect=AssertionError("broker reached")))
    with pytest.raises(gw.GatewayUnavailable):
        gw.fetch_credential(config, now=NOW, transport=transport)


def test_capability_file_read_each_request_and_rotation(gateway):
    config, path, grant, publish = gateway
    seen = []

    def respond(request):
        seen.append(request.headers["X-Preview-Broker-Token"])
        return httpx.Response(200, json={"access_token": "fixture", "state": "ok",
                                          "expires_at": (NOW + 1200) * 1000})

    transport = httpx.MockTransport(respond)
    gw.fetch_credential(config, now=NOW, transport=transport)
    publish(now=NOW + 5)
    gw.fetch_credential(config, now=NOW + 5, transport=transport)
    assert seen[0] != seen[1]
    path.unlink()
    with pytest.raises(gw.GatewayUnavailable):
        gw.fetch_credential(config, now=NOW + 5, transport=transport)
    assert len(seen) == 2


@pytest.mark.parametrize("status,body", [
    (503, {"detail": "prod error"}),
    (302, {"access_token": "fixture", "state": "ok", "expires_at": (NOW + 1200) * 1000}),
    (200, {"access_token": "fixture", "state": "ok", "expires_at": (NOW + 1200) * 1000,
           "refresh_token": "not allowed"}),
    (200, {"access_token": "fixture", "state": "ok", "expires_at": NOW + 1200}),
    (200, {"access_token": "fixture", "state": "ok", "expires_at": (NOW + 30) * 1000}),
])
def test_broker_failures_never_expose_token_or_fallback(gateway, status, body):
    config, _, _, _ = gateway
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    with pytest.raises(gw.GatewayUnavailable) as error:
        gw.fetch_credential(config, now=NOW, transport=transport)
    assert "fixture" not in str(error.value)
    assert "prod error" not in str(error.value)


def test_chatgpt_shape_and_account_is_required(gateway):
    config, _, grant, publish = gateway
    chatgpt = broker.PreviewGrant(grant.run_id, "openai_chatgpt", 8, grant.expires_at)
    config = gw.GatewayConfig(config.capability_file, config.broker_origin,
                              "openai_chatgpt", config.run_id)
    publish(run=chatgpt)
    ok = {"access_token": "fixture", "state": "ok", "account_id": "account",
          "expires_at": NOW + 1200}
    assert json.loads(gw.fetch_credential(config, now=NOW,
                         transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ok)))) == ok
    ok["expires_at"] = float(NOW + 1200)  # broker JWT expiration is numeric seconds
    assert json.loads(gw.fetch_credential(config, now=NOW,
                         transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ok)))) == ok
    del ok["account_id"]
    with pytest.raises(gw.GatewayUnavailable):
        gw.fetch_credential(config, now=NOW,
                            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ok)))


@pytest.mark.parametrize("origin", ["http://broker.test/other", "http://u:p@broker.test",
                                     "file:///tmp/token", "http://broker.test?url=prod",
                                     "http://broker.test/#fragment", "broker.test"])
def test_invalid_broker_origin_rejected(origin):
    with pytest.raises(gw.GatewayUnavailable):
        gw.GatewayConfig.from_env({gw.CAPABILITY_FILE_ENV: "/tmp/fixture",
                                   gw.BROKER_ORIGIN_ENV: origin,
                                   gw.PROVIDER_ENV: "claude", gw.RUN_ID_ENV: RUN})


def test_only_single_exact_route_allowed(gateway, monkeypatch):
    config, _, _, _ = gateway
    calls = Mock(return_value=b"{}")
    monkeypatch.setattr(gw, "fetch_credential", calls)
    handler = object.__new__(gw.CredentialHandler)
    handler.config = config
    responses = []
    handler._reply = lambda status, body=b"": responses.append((status, body))
    for path in ["/v1/preview/credential", "/v1/providers/claude/bundle",
                 "/v1/providers/openai_chatgpt/token", "/v1/providers/claude/token?force=true",
                 "/v1/providers/claude/token/", "http://prod.test/v1/providers/claude/token"]:
        handler.path = path
        handler.do_GET()
    assert responses == [(404, b"")] * 6
    calls.assert_not_called()
    handler.path = "/v1/providers/claude/token"
    handler.do_GET()
    assert responses[-1] == (200, b"{}")
    calls.assert_called_once_with(config)


def test_rate_limit_denies_without_broker_request(gateway, monkeypatch):
    config, _, _, _ = gateway
    calls = Mock(return_value=b"{}")
    monkeypatch.setattr(gw, "fetch_credential", calls)
    handler_type = type("TestHandler", (gw.CredentialHandler,),
                        {"_rate_bucket_start": 1.0, "_rate_count": 0,
                         "_rate_per_minute": 2, "config": config})
    handler = object.__new__(handler_type)
    handler.path = "/v1/providers/claude/token"
    responses = []
    handler._reply = lambda status, body=b"": responses.append(status)
    assert handler._allow(now=1.0)
    assert handler._allow(now=1.0)
    assert not handler._allow(now=1.0)
    assert handler._allow(now=61.0)
    # Test the actual route refusal without consuming its allowance.
    handler_type._rate_count = 2
    handler_type._rate_bucket_start = gw.time.monotonic()
    handler.do_GET()
    assert responses == [429]
    calls.assert_not_called()


def test_health_checks_bearer_without_calling_broker(gateway, monkeypatch):
    config, path, _, _ = gateway
    calls = Mock(side_effect=AssertionError("broker reached"))
    monkeypatch.setattr(gw, "fetch_credential", calls)
    monkeypatch.setattr(gw.time, "time", lambda: NOW)
    handler = object.__new__(gw.CredentialHandler)
    handler.config = config
    handler.path = "/health"
    responses = []
    handler._reply = lambda status, body=b"": responses.append((status, body))
    handler.do_GET()
    path.unlink()
    handler.do_GET()
    assert responses == [(200, b'{"state":"ok"}'), (503, b"")]
    calls.assert_not_called()


def test_gateway_errors_have_json_without_broker_details():
    handler = object.__new__(gw.CredentialHandler)
    handler.wfile = BytesIO()
    statuses = []
    headers = {}
    handler.send_response = statuses.append
    handler.send_header = lambda key, value: headers.update({key.lower(): value})
    handler.end_headers = lambda: None
    handler._reply(503, b"upstream SECRET")
    assert statuses == [503]
    assert headers["cache-control"] == "no-store"
    assert json.loads(handler.wfile.getvalue()) == {"detail": "Preview credential unavailable"}
    assert b"SECRET" not in handler.wfile.getvalue()
