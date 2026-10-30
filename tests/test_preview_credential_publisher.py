"""Production grant publication is opt-in, scoped and immediately revocable."""

import json
from types import SimpleNamespace

import pytest

from llm_bawt.preview import credential_gateway as gateway
from llm_bawt.preview import credential_publisher as publisher
from llm_bawt.service import preview_credential as broker

NOW = 2_000_000_000
RUN = "00000000-0000-4000-8000-000000000001"


class _Engine:
    def __init__(self):
        self.rows = {}
        self.writes = []

    def begin(self):
        return self

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params):
        self.writes.append(str(sql))
        if str(sql).startswith("DELETE"):
            self.rows.pop(params["key"], None)
        elif str(sql).startswith("INSERT"):
            self.rows[params["key"]] = json.loads(params["value"])
        return SimpleNamespace(scalar_one_or_none=lambda: json.dumps(self.rows[params["key"]])
                               if params["key"] in self.rows else None)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(broker.crypto, "derive_key", lambda _: b"fixture-key")
    engine = _Engine()
    path = tmp_path / "export" / "capability.json"
    env = {publisher.RUN_ENV: RUN, publisher.PROVIDER_ENV: "claude",
           publisher.ENDPOINT_ENV: "7", publisher.EXPIRY_ENV: str(NOW + 1800),
           publisher.EXPORT_ENV: str(path)}
    def resolve_endpoint(endpoint_id, *, harness):
        if endpoint_id != 7 or harness != "claude-code":
            raise LookupError("fixture endpoint not available")
        return SimpleNamespace(access_path=SimpleNamespace(
            vendor="anthropic", auth_mechanism="oauth", key="anthropic-oauth",
        ))

    catalog = SimpleNamespace(resolve_endpoint=resolve_endpoint)
    return engine, catalog, env, path


def test_activation_publishes_gateway_only_bearer_and_db_grant(setup):
    engine, catalog, env, path = setup
    assert publisher.activate(engine, catalog, env, now=NOW) == path
    grant = engine.rows[broker.GRANT_KEY]
    assert grant == {"run_id": RUN, "provider": "claude", "endpoint_id": 7,
                     "expires_at": NOW + 1800}
    data = json.loads(path.read_text())
    assert set(data) == set(grant) | {"capability"}
    assert gateway._capability(gateway.GatewayConfig(path, "http://broker.test", "claude", RUN), NOW)
    assert path.stat().st_mode & 0o777 == 0o600
    assert "refresh_token" not in path.read_text()


def test_refresh_renews_capability_and_revocation_removes_file(setup):
    engine, catalog, env, path = setup
    publisher.activate(engine, catalog, env, now=NOW)
    old = json.loads(path.read_text())["capability"]
    assert publisher.refresh(engine, catalog, path, env, now=NOW + 120)
    assert json.loads(path.read_text())["capability"] != old
    engine.rows.pop(broker.GRANT_KEY)
    assert publisher.refresh(engine, catalog, path, env, now=NOW + 121) is False
    assert not path.exists()


def test_removed_endpoint_revokes_and_does_not_block_production_startup(setup, caplog):
    engine, catalog, env, path = setup
    publisher.activate(engine, catalog, env, now=NOW)
    catalog.resolve_endpoint = lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError())
    assert publisher.activate(engine, catalog, env, now=NOW + 1) is None
    assert broker.GRANT_KEY not in engine.rows
    assert not path.exists()
    assert "endpoint no longer authorized" in caplog.text


def test_expired_run_revokes_and_does_not_block_production_startup(setup, caplog):
    engine, catalog, env, path = setup
    publisher.activate(engine, catalog, env, now=NOW)
    assert publisher.activate(engine, catalog, env, now=NOW + 1801) is None
    assert broker.GRANT_KEY not in engine.rows
    assert not path.exists()
    assert "run expired" in caplog.text


def test_renewal_logs_refusal_even_if_export_volume_disappears(setup, monkeypatch, caplog):
    engine, catalog, env, path = setup
    publisher.activate(engine, catalog, env, now=NOW)
    original = publisher.publish

    def broken_withdraw(file, token, environ, **kwargs):
        if token is None:
            raise OSError("volume gone")
        return original(file, token, environ, **kwargs)

    monkeypatch.setattr(publisher, "publish", broken_withdraw)
    engine.rows.pop(broker.GRANT_KEY)
    assert publisher.refresh(engine, catalog, path, env, now=NOW + 1) is False
    assert "renewal refused" in caplog.text
    assert "export file could not be removed" in caplog.text
    assert "volume gone" not in caplog.text


def test_wrong_provider_never_publishes(setup):
    engine, catalog, env, path = setup
    env[publisher.PROVIDER_ENV] = "openai_chatgpt"
    assert publisher.activate(engine, catalog, env, now=NOW) is None
    assert broker.GRANT_KEY not in engine.rows
    assert not path.exists()


def test_malformed_provider_refuses_before_database_write(setup):
    engine, catalog, env, _ = setup
    env[publisher.PROVIDER_ENV] = "not-a-provider"
    with pytest.raises(publisher.ActivationUnavailable):
        publisher.activate(engine, catalog, env, now=NOW)
    assert not engine.writes


def test_partial_configuration_refuses_before_database_write(setup):
    engine, catalog, env, _ = setup
    del env[publisher.ENDPOINT_ENV]
    with pytest.raises(publisher.ActivationUnavailable):
        publisher.activate(engine, catalog, env, now=NOW)
    assert not engine.writes


def test_failed_export_rolls_back_grant(setup, monkeypatch):
    engine, catalog, env, path = setup
    original = publisher.publish

    def fail(path, token, env, **kwargs):
        if token is not None:
            raise OSError("volume unavailable")
        return original(path, token, env, **kwargs)

    monkeypatch.setattr(publisher, "publish", fail)
    with pytest.raises(publisher.ActivationUnavailable):
        publisher.activate(engine, catalog, env, now=NOW)
    assert broker.GRANT_KEY not in engine.rows
    assert not path.exists()


def test_no_activation_withdraws_existing_grant(setup):
    engine, catalog, env, path = setup
    publisher.activate(engine, catalog, env, now=NOW)
    assert publisher.activate(engine, catalog, {}, now=NOW + 5) is None
    assert broker.GRANT_KEY not in engine.rows
    # No export path in the disabled configuration: old bearer may remain
    # on disk until TTL, but the broker denies it immediately by DB revocation.
    assert path.exists()
