"""Cutover credential probes (TASK-1476 §3): fakes only, no network, no DB."""
from __future__ import annotations

import io
import json

import httpx
import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError

from llm_bawt.env_import import SPECS
from llm_bawt.ops import probe_contract as pc
from llm_bawt.service import cutover_probe as cp
from llm_bawt.service.connection_secrets import ConnectionSecret, ConnectionUnavailable

FAKE_KEY = "sk-fake-probe-secret-0123456789"
FAKE_TOKEN = "f" * 64


class Upstream:
    """An httpx MockTransport handler: ``(method, path) -> response`` plus a request log."""

    def __init__(self, routes=None, default=None):
        self.routes = dict(routes or {})
        self.default = default
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get(request.url.path, self.default)
        if answer is None:
            raise AssertionError(f"unexpected request {request.url.path}")
        if isinstance(answer, Exception):
            raise answer
        return answer(request) if callable(answer) else answer


def _secret(kind, connection_id, meta=None, secret=None):
    return ConnectionSecret(kind=kind, id=connection_id, meta=dict(meta or {}), secret=dict(secret or {}))


def _stored(records):
    def read(kind, connection_id, spec):
        found = records.get((kind, connection_id))
        if isinstance(found, Exception):
            raise found
        return found
    return read


def _ctx(upstream=None, records=None, *, environ=None, derive=lambda: FAKE_TOKEN, s3_client=None):
    return cp.ProbeContext(
        http=httpx.Client(transport=httpx.MockTransport(upstream or Upstream())),
        read=_stored(records or {}),
        environ=environ or {},
        derive_crawl4ai_token=derive,
        s3_client=s3_client or (lambda cfg: pytest.fail("no S3 call expected")),
    )


def _run(probe_id, ctx):
    line = cp.run_probe(probe_id, lambda: ctx)
    assert FAKE_KEY not in line and FAKE_TOKEN not in line
    return pc.parse_line(line)


def _key(provider_id):
    return {("provider", provider_id): _secret("provider", provider_id, secret={"api_key": FAKE_KEY})}


def _result(parsed):
    return parsed["ok"], parsed["status"], parsed["code"]


def test_every_app_probe_is_served_and_only_broker_is_left_to_the_bridge():
    assert set(cp.PROBES) | cp.BRIDGE_PROBES == pc.PROBE_IDS
    assert not set(cp.PROBES) & cp.BRIDGE_PROBES


# --- api-key probes ---------------------------------------------------------------

def _json(status, payload):
    return httpx.Response(status, json=payload)


@pytest.mark.parametrize("answer,expected", [
    (_json(200, {"data": [{"id": "m"}]}), (True, 200, "ok")),
    (_json(200, {"data": []}), (False, 200, "unexpected_response")),
    (_json(200, {"nope": 1}), (False, 200, "unexpected_response")),
    (httpx.Response(200, text="<html>"), (False, 200, "unexpected_response")),
    (_json(401, {}), (False, 401, "unauthorized")),
    (_json(500, {}), (False, 500, "http_error")),
    (httpx.ConnectError("refused"), (False, None, "unreachable")),
    (httpx.ReadTimeout("slow"), (False, None, "timeout")),
])
def test_anthropic_models(answer, expected):
    upstream = Upstream({"/v1/models": answer})
    assert _result(_run("anthropic_models", _ctx(upstream, _key("anthropic-api")))) == expected
    sent = upstream.requests[0]
    assert sent.headers["x-api-key"] == FAKE_KEY and sent.method == "GET"


@pytest.mark.parametrize("answer,expected", [
    (_json(200, {"code": 200, "success": True, "data": {}}), (True, 200, "ok")),
    (_json(200, {"code": 401, "success": False}), (False, 200, "unauthorized")),
    (_json(200, {"code": 500, "success": False}), (False, 200, "unexpected_response")),
    # Valid key, no GLM Coding Plan: the key authenticated, so the probe passes.
    # z.ai has no dedicated code for this; the exact msg is the only signal.
    (_json(200, {"code": 500, "msg": "当前用户不存在coding plan", "success": False}),
     (True, 200, "ok")),
    # Mutants: the no-plan msg never masks a rejected key, another msg, or data.
    (_json(200, {"code": 401, "msg": "当前用户不存在coding plan", "success": False}),
     (False, 200, "unauthorized")),
    (_json(200, {"code": 500, "msg": "coding plan service unavailable", "success": False}),
     (False, 200, "unexpected_response")),
    (_json(200, {"code": 500, "msg": "当前用户不存在coding plan", "success": False, "data": {}}),
     (False, 200, "unexpected_response")),
    (_json(401, {"code": 500, "msg": "当前用户不存在coding plan", "success": False}),
     (False, 401, "unauthorized")),
    # Only the exact 200 body passes: not another 2xx, padded/case-changed msg,
    # data:null, an extra key, or a bool/str code.
    (_json(201, {"code": 500, "msg": "当前用户不存在coding plan", "success": False}),
     (False, 201, "unexpected_response")),
    (_json(200, {"code": 500, "msg": " 当前用户不存在coding plan ", "success": False}),
     (False, 200, "unexpected_response")),
    (_json(200, {"code": 500, "msg": "当前用户不存在CODING PLAN", "success": False}),
     (False, 200, "unexpected_response")),
    (_json(200, {"code": 500, "msg": "当前用户不存在coding plan", "success": False, "data": None}),
     (False, 200, "unexpected_response")),
    (_json(200, {"code": 500, "msg": "当前用户不存在coding plan", "success": False, "x": 1}),
     (False, 200, "unexpected_response")),
    (_json(200, {"code": "500", "msg": "当前用户不存在coding plan", "success": False}),
     (False, 200, "unexpected_response")),
    (httpx.Response(200, text="not json"), (False, 200, "unexpected_response")),
    (_json(400, {}), (False, 400, "unauthorized")),
    (_json(403, {}), (False, 403, "unauthorized")),
    (_json(502, {}), (False, 502, "http_error")),
])
def test_zai_completion(answer, expected):
    upstream = Upstream(default=answer)
    assert _result(_run("zai_completion", _ctx(upstream, _key("zai")))) == expected


@pytest.mark.parametrize("answer,expected", [
    (_json(200, {"data": []}), (True, 200, "ok")),
    (_json(401, {}), (False, 401, "unauthorized")),
    (_json(429, {}), (False, 429, "http_error")),
])
def test_kimi_completion(answer, expected):
    upstream = Upstream(default=answer)
    assert _result(_run("kimi_completion", _ctx(upstream, _key("kimi-coding")))) == expected
    assert upstream.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


@pytest.mark.parametrize("probe_id,provider_id,header", [
    ("brave_search", "brave", "x-subscription-token"),
    ("newsapi_search", "newsapi", None),
])
def test_search_keys(probe_id, provider_id, header):
    ok = Upstream(default=_json(200, {}))
    assert _result(_run(probe_id, _ctx(ok, _key(provider_id)))) == (True, 200, "ok")
    if header:
        assert ok.requests[0].headers[header] == FAKE_KEY
    rejected = 422 if provider_id == "brave" else 401
    bad = Upstream(default=_json(rejected, {}))
    assert _result(_run(probe_id, _ctx(bad, _key(provider_id)))) == (False, rejected, "unauthorized")


@pytest.mark.parametrize("probe_id,provider_id", [
    ("anthropic_models", "anthropic-api"), ("zai_completion", "zai"), ("kimi_completion", "kimi-coding"),
    ("brave_search", "brave"), ("newsapi_search", "newsapi"),
])
@pytest.mark.parametrize("stored,code", [
    (None, "not_configured"),
    (ConnectionUnavailable("not_connected"), "not_configured"),
    (ConnectionUnavailable("unreadable"), "not_configured"),
    (ConnectionUnavailable("store_unavailable"), "error"),
])
def test_key_probes_need_the_stored_record(probe_id, provider_id, stored, code):
    # No upstream route: a probe without a usable record must not call out.
    parsed = _run(probe_id, _ctx(Upstream(), {("provider", provider_id): stored}))
    assert _result(parsed) == (False, None, code)


def test_key_probes_never_read_env(monkeypatch):
    for spec in SPECS:
        for name in spec.env_names:
            monkeypatch.setenv(name, FAKE_KEY)
    assert _result(_run("zai_completion", _ctx(Upstream()))) == (False, None, "not_configured")


# --- media integrations -------------------------------------------------------------

def _integration(integration_id, url="http://media.test:8989/"):
    return {("integration", integration_id): _secret("integration", integration_id,
                                                     meta={"url": url}, secret={"api_key": FAKE_KEY})}


@pytest.mark.parametrize("probe_id,integration_id", [("sonarr_status", "sonarr"), ("radarr_status", "radarr")])
@pytest.mark.parametrize("answer,expected", [
    (_json(200, {"version": "4"}), (True, 200, "ok")),
    (_json(200, ["not", "a", "dict"]), (False, 200, "unexpected_response")),
    (_json(401, {}), (False, 401, "unauthorized")),
    (_json(404, {}), (False, 404, "http_error")),
])
def test_arr_status(probe_id, integration_id, answer, expected):
    upstream = Upstream({"/api/v3/system/status": answer})
    assert _result(_run(probe_id, _ctx(upstream, _integration(integration_id)))) == expected
    assert upstream.requests[0].headers["x-api-key"] == FAKE_KEY


@pytest.mark.parametrize("answer,expected", [
    (_json(200, {"queue": {"slots": []}}), (True, 200, "ok")),
    (_json(200, {"status": False, "error": "API Key Incorrect"}), (False, 200, "unauthorized")),
    (_json(200, {"version": "4"}), (False, 200, "unexpected_response")),
    (_json(500, {}), (False, 500, "http_error")),
])
def test_sabnzbd_status(answer, expected):
    upstream = Upstream({"/api": answer})
    assert _result(_run("sabnzbd_status", _ctx(upstream, _integration("sabnzbd")))) == expected
    assert upstream.requests[0].url.params["mode"] == "queue"


def test_a_stored_url_httpx_cannot_use_is_unreachable():
    parsed = _run("sonarr_status", _ctx(Upstream(), _integration("sonarr", url="http://[::1")))
    assert _result(parsed) == (False, None, "unreachable")


# --- s3 ------------------------------------------------------------------------------

def _object_store():
    return {("integration", "object-store"): _secret(
        "integration", "object-store",
        meta={"endpoint": "http://s3.test", "region": "garage", "bucket": "b"},
        secret={"access_key_id": "AK", "secret_access_key": FAKE_KEY})}


class FakeS3:
    def __init__(self, outcome):
        self.outcome = outcome
        self.cfg = None

    def __call__(self, cfg):
        self.cfg = cfg
        return self

    def head_bucket(self, Bucket):  # noqa: N803 - boto3's name
        assert Bucket == "b"
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _client_error(status):
    return ClientError({"Error": {"Code": str(status)}, "ResponseMetadata": {"HTTPStatusCode": status}}, "HeadBucket")


@pytest.mark.parametrize("outcome,expected", [
    ({"ResponseMetadata": {"HTTPStatusCode": 200}}, (True, 200, "ok")),
    (_client_error(403), (False, 403, "unauthorized")),
    (_client_error(404), (False, 404, "http_error")),
    (EndpointConnectionError(endpoint_url="http://s3.test"), (False, None, "unreachable")),
    (ConnectTimeoutError(endpoint_url="http://s3.test"), (False, None, "timeout")),
    # A malformed "success" is not a pass (Codex, TASK-1476 s3 review).
    ({}, (False, None, "http_error")),
    (None, (False, None, "http_error")),
    ({"ResponseMetadata": {}}, (False, None, "http_error")),
    ({"ResponseMetadata": {"HTTPStatusCode": "200"}}, (False, None, "http_error")),
    ({"ResponseMetadata": {"HTTPStatusCode": True}}, (False, None, "http_error")),
    ({"ResponseMetadata": {"HTTPStatusCode": 301}}, (False, 301, "http_error")),
])
def test_s3_head(outcome, expected):
    fake = FakeS3(outcome)
    assert _result(_run("s3_head", _ctx(records=_object_store(), s3_client=fake))) == expected
    assert fake.cfg.max_attempts == 1 and fake.cfg.secret_key == FAKE_KEY


def test_s3_head_without_a_record():
    assert _result(_run("s3_head", _ctx())) == (False, None, "not_configured")


# --- crawl4ai --------------------------------------------------------------------------

def _crawl4ai(bearer_status, anonymous_status):
    def schema(request):
        authorized = request.headers.get("authorization") == f"Bearer {FAKE_TOKEN}"
        return httpx.Response(bearer_status if authorized else anonymous_status, json={})
    return Upstream({"/schema": schema})


@pytest.mark.parametrize("bearer,anonymous,expected", [
    (200, 401, (True, 200, "ok")),
    (200, 200, (False, 200, "unexpected_response")),   # the gate is open
    (401, 401, (False, 401, "unauthorized")),
    (500, 401, (False, 500, "http_error")),
])
def test_crawl4ai_bearer(bearer, anonymous, expected):
    upstream = _crawl4ai(bearer, anonymous)
    ctx = _ctx(upstream, environ={"LLM_BAWT_CRAWL4AI_URL": "http://crawl.test:11235/"})
    assert _result(_run("crawl4ai_bearer", ctx)) == expected
    assert all(r.url.host == "crawl.test" for r in upstream.requests)


def test_crawl4ai_bearer_without_a_derivable_token():
    def broken():
        raise RuntimeError("no root")
    assert _result(_run("crawl4ai_bearer", _ctx(derive=broken))) == (False, None, "not_configured")


def test_web_fetch_uses_the_exported_token_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(FAKE_TOKEN)
    environ = {"CRAWL4AI_API_TOKEN_FILE": str(token_file), "LLM_BAWT_CRAWL4AI_URL": "http://crawl.test"}
    assert _result(_run("web_fetch", _ctx(_crawl4ai(200, 401), environ=environ))) == (True, 200, "ok")
    assert _result(_run("web_fetch", _ctx(_crawl4ai(401, 401), environ=environ))) == (False, 401, "unauthorized")


def test_web_fetch_without_a_token(tmp_path):
    environ = {"CRAWL4AI_API_TOKEN_FILE": str(tmp_path / "missing")}
    assert _result(_run("web_fetch", _ctx(environ=environ))) == (False, None, "not_configured")


# --- records -----------------------------------------------------------------------------

def test_records_reports_configured_and_connected():
    stored = {
        ("provider", "zai"): _secret("provider", "zai", secret={"api_key": FAKE_KEY}),
        ("integration", "sonarr"): ConnectionUnavailable("not_connected"),
        ("integration", "radarr"): ConnectionUnavailable("unreadable"),
        ("provider", "claude"): _secret("provider", "claude"),
    }
    parsed = _run("records", _ctx(records=stored))
    assert _result(parsed) == (True, None, "ok")
    records = parsed["records"]
    assert records["zai"] == {"configured": True, "connected": True}
    assert records["claude"] == {"configured": True, "connected": True}
    assert records["sonarr"] == records["radarr"] == {"configured": True, "connected": False}
    assert records["brave"] == {"configured": False, "connected": False}


def test_records_reads_each_key_with_its_kind():
    asked = []

    def read(kind, connection_id, spec):
        asked.append((kind, connection_id))
        return None

    ctx = _ctx()
    ctx.read = read
    _run("records", ctx)
    assert {connection_id for _, connection_id in asked} == pc.RECORD_KEYS
    assert dict((cid, kind) for kind, cid in asked) == {**{s.destination: s.kind for s in SPECS},
                                                        "anthropic-api": "provider", "claude": "provider"}


def test_records_store_outage_is_an_error_without_records():
    stored = {("provider", "zai"): ConnectionUnavailable("store_unavailable")}
    assert _run("records", _ctx(records=stored)) == {"probe": "records", "ok": False, "status": None, "code": "error"}


# --- entry point ---------------------------------------------------------------------------

def _main(argv, monkeypatch, ctx=None):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    if ctx is not None:
        monkeypatch.setattr(cp, "default_context", lambda: ctx)
    assert cp.main(argv) == 0
    return out.getvalue(), err.getvalue()


@pytest.mark.parametrize("argv,probe", [
    (["no-such-probe"], "records"),
    (["broker"], "broker"),
    ([], "records"),
    (["zai_completion", "extra"], "records"),
])
def test_bad_argv_is_one_error_line(argv, probe, monkeypatch):
    out, _ = _main(argv, monkeypatch)
    assert out.count("\n") == 1
    assert pc.parse_line(out) == {"probe": probe, "ok": False, "status": None, "code": "error"}


def test_main_prints_one_line_and_diverts_library_output(monkeypatch):
    class Loud(cp.Probe):
        id = "zai_completion"

        def run(self):
            print("library chatter", FAKE_KEY)
            return cp.ProbeResult.failed(pc.UNAUTHORIZED, 401)

    monkeypatch.setitem(cp.PROBES, "zai_completion", Loud)
    out, err = _main(["zai_completion"], monkeypatch, ctx=_ctx())
    assert out == pc.emit("zai_completion", False, 401, "unauthorized") + "\n"
    assert FAKE_KEY not in out and "library chatter" in err


def test_a_crashing_probe_is_an_error_line_without_its_message(monkeypatch):
    class Crash(cp.Probe):
        id = "s3_head"

        def run(self):
            raise RuntimeError(f"boom {FAKE_KEY}")

    monkeypatch.setitem(cp.PROBES, "s3_head", Crash)
    out, err = _main(["s3_head"], monkeypatch, ctx=_ctx())
    assert json.loads(out) == {"probe": "s3_head", "ok": False, "status": None, "code": "error"}
    assert FAKE_KEY not in out + err


def test_a_failing_context_is_an_error_line(monkeypatch):
    def broken():
        raise RuntimeError("no config")
    monkeypatch.setattr(cp, "default_context", broken)
    out, _ = _main(["zai_completion"], monkeypatch)
    assert pc.parse_line(out)["code"] == "error"
