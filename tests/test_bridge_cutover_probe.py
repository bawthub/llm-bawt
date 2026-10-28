"""The bridge's ``broker`` cutover probe (TASK-1476 §3): fakes only."""
from __future__ import annotations

import io

import httpx
import pytest

from claude_code_bridge import cutover_probe as bp
from claude_code_bridge import provider_key_broker
from claude_code_bridge.provider_key_broker import BrokerKeyError
from llm_bawt.ops import probe_contract as pc

FAKE_TOKEN = "sk-ant-api-fake-broker-key-0123456789"


def _fetch(answer):
    asked = []

    def fetch(provider_id):
        asked.append(provider_id)
        if isinstance(answer, Exception):
            raise answer
        return answer
    fetch.asked = asked
    return fetch


def _run(answer):
    fetch = _fetch(answer)
    line = bp.run_probe(["broker"], fetch)
    assert FAKE_TOKEN not in line
    assert fetch.asked == ["anthropic-api"]
    parsed = pc.parse_line(line)
    return parsed["ok"], parsed["status"], parsed["code"]


def test_the_bridge_loads_the_same_contract_by_path():
    assert bp.pc.PROBE_IDS == pc.PROBE_IDS and bp.pc.CODES == pc.CODES
    assert bp.PROBE_ID in pc.PROBE_IDS


@pytest.mark.parametrize("answer,expected", [
    (httpx.Response(200, json={"access_token": FAKE_TOKEN}), (True, 200, "ok")),
    (httpx.Response(200, json={"access_token": ""}), (False, 200, "unexpected_response")),
    (httpx.Response(200, json=["x"]), (False, 200, "unexpected_response")),
    (httpx.Response(200, text="nope"), (False, 200, "unexpected_response")),
    (httpx.Response(503, json={}), (False, 503, "not_configured")),
    (httpx.Response(401, json={}), (False, 401, "unauthorized")),
    (httpx.Response(403, json={}), (False, 403, "unauthorized")),
    (httpx.Response(500, json={}), (False, 500, "http_error")),
    (httpx.ConnectError("refused"), (False, None, "unreachable")),
    (httpx.ReadTimeout("slow"), (False, None, "timeout")),
    (BrokerKeyError("LLM_BAWT_API_URL not set"), (False, None, "error")),
    (RuntimeError("boom"), (False, None, "error")),
])
def test_broker_outcomes(answer, expected):
    assert _run(answer) == expected


def test_the_default_fetch_is_the_broker_call(monkeypatch):
    seen = []

    def fake_get(url, headers, timeout):
        seen.append(url)
        return httpx.Response(200, json={"access_token": FAKE_TOKEN})

    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app.test:8642/")
    monkeypatch.setattr(provider_key_broker.httpx, "get", fake_get)
    monkeypatch.setattr(provider_key_broker.bridge_token, "headers", lambda: {})
    line = bp.run_probe(["broker"])
    assert pc.parse_line(line)["ok"] is True and FAKE_TOKEN not in line
    assert seen == ["http://app.test:8642/v1/providers/anthropic-api/token"]


def test_fetch_provider_api_key_still_wraps_transport_errors(monkeypatch):
    def fail(url, headers, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app.test")
    monkeypatch.setattr(provider_key_broker.httpx, "get", fail)
    monkeypatch.setattr(provider_key_broker.bridge_token, "headers", lambda: {})
    with pytest.raises(BrokerKeyError, match="unreachable"):
        provider_key_broker.fetch_provider_api_key("anthropic-api")


@pytest.mark.parametrize("argv,probe", [([], "records"), (["records"], "records"),
                                        (["zai_completion"], "zai_completion"), (["broker", "x"], "records"),
                                        (["nope"], "records")])
def test_anything_but_broker_is_one_error_line(argv, probe, monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert bp.main(argv) == 0
    assert out.getvalue().count("\n") == 1
    assert pc.parse_line(out.getvalue()) == {"probe": probe, "ok": False, "status": None, "code": "error"}
