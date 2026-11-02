"""Independent nonsecret-generation rekey; all credentials/HTTP are disposable fakes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from llm_bawt.service import crawl4ai_rekey as rekey
from llm_bawt.service import crawl4ai_token_export as tokens
from llm_bawt.service.providers import crypto


@pytest.fixture
def generation_db(monkeypatch):
    from llm_bawt.utils import config, db

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE runtime_settings (scope_type TEXT, scope_id TEXT, key TEXT, value_json TEXT)"))
    monkeypatch.setattr(config, "Config", lambda: object())
    monkeypatch.setattr(config, "has_database_credentials", lambda _: True)
    monkeypatch.setattr(db, "get_shared_engine", lambda _: engine)
    yield engine
    engine.dispose()


def put(engine, value, *, scope="global", sid="*", raw=False):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO runtime_settings VALUES (:scope, :sid, :key, :value)"),
                     {"scope": scope, "sid": sid, "key": tokens.GENERATION_KEY,
                      "value": value if raw else json.dumps(value)})


def test_global_only_missing_row_and_no_cache(generation_db):
    put(generation_db, 77, scope="bot", sid="fixture")
    assert tokens.current_generation() == 0
    put(generation_db, 1)
    assert tokens.current_generation() == 1
    with generation_db.begin() as conn:
        conn.execute(text("UPDATE runtime_settings SET value_json='2' WHERE scope_type='global'"))
    assert tokens.current_generation() == 2


@pytest.mark.parametrize("value", [None, True, False, -1, 2**31, 1.0, "1", {}, [], "invalid-json"])
def test_bad_row_refuses_instead_of_falling_back(generation_db, value):
    put(generation_db, value, raw=value == "invalid-json")
    with pytest.raises((ValueError, TypeError)):
        tokens.current_generation()


def test_duplicate_generation_refused(generation_db):
    put(generation_db, 1)
    put(generation_db, 2)
    with pytest.raises(ValueError):
        tokens.current_generation()


def test_configured_db_unavailable_never_legacy(monkeypatch):
    from llm_bawt.utils import config, db

    monkeypatch.setattr(config, "Config", lambda: object())
    monkeypatch.setattr(config, "has_database_credentials", lambda _: True)
    monkeypatch.setattr(db, "get_shared_engine", lambda _: None)
    with pytest.raises(RuntimeError):
        tokens.current_generation()


def test_unconfigured_standalone_retains_legacy(monkeypatch):
    from llm_bawt.utils import config

    monkeypatch.setattr(config, "Config", lambda: object())
    monkeypatch.setattr(config, "has_database_credentials", lambda _: False)
    assert tokens.current_generation() == 0


def test_generation_only_changes_crawl4ai_preserves_root_and_ciphertext(monkeypatch):
    key = Fernet.generate_key()
    calls = []
    monkeypatch.setattr(crypto, "_load_or_create_key", lambda **kw: calls.append(kw) or key)
    monkeypatch.setattr(crypto, "_fernet", None)
    ciphertext = crypto.encrypt("disposable-provider-credential")
    unrelated = {label: crypto.derive_key(label) for label in ("bawthub-service-token/v1", "bridge-gate/v1")}
    legacy = crypto.derive_key(tokens.LABEL).hex()
    assert tokens.derive_for_generation(0) == legacy
    one = tokens.derive_for_generation(1)
    two = tokens.derive_for_generation(2)
    assert len({legacy, one, two}) == 3
    assert tokens.derive_for_generation(1) == one
    assert crypto.decrypt(ciphertext) == "disposable-provider-credential"
    assert unrelated == {label: crypto.derive_key(label) for label in unrelated}
    assert crypto._load_or_create_key() == key
    assert any(call.get("require_persisted") for call in calls)


@pytest.mark.parametrize("scope,value", [("bot", 1), ("global", True), ("global", -1),
                                         ("global", "1"), ("global", 2**31)])
def test_settings_api_validates_before_any_write(monkeypatch, scope, value):
    from llm_bawt.service.routes import settings
    from llm_bawt.service.schemas import RuntimeSettingUpsertRequest

    writes = []
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=None, _default_bot="fixture"))
    monkeypatch.setattr(settings, "get_runtime_settings_store", lambda _: SimpleNamespace(
        engine=object(), set_value=lambda *args: writes.append(args)))
    with pytest.raises(HTTPException) as error:
        settings.upsert_runtime_setting(RuntimeSettingUpsertRequest(
            scope_type=scope, scope_id="fixture", key=tokens.GENERATION_KEY, value=value))
    assert error.value.status_code == 400
    assert writes == []


def test_settings_valid_global_and_whole_batch_validation(monkeypatch):
    from llm_bawt.service.routes import settings
    from llm_bawt.service.schemas import (
        RuntimeSettingBatchItem,
        RuntimeSettingBatchUpsertRequest,
        RuntimeSettingUpsertRequest,
    )

    writes = []
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=None, _default_bot="fixture"))
    monkeypatch.setattr(settings, "get_runtime_settings_store", lambda _: SimpleNamespace(
        engine=object(), set_value=lambda *args: writes.append(args)))
    monkeypatch.setattr(settings, "WorkingMemoryPolicy", lambda _: SimpleNamespace(validate_setting_write=lambda *a, **k: None))
    monkeypatch.setattr(settings, "_invalidate_all_instance_cache", lambda _: None)
    settings.upsert_runtime_setting(RuntimeSettingUpsertRequest(
        scope_type="global", key=tokens.GENERATION_KEY, value=1))
    assert writes == [("global", "*", tokens.GENERATION_KEY, 1)]
    writes.clear()
    with pytest.raises(HTTPException):
        settings.batch_upsert_runtime_settings(RuntimeSettingBatchUpsertRequest(items=[
            RuntimeSettingBatchItem(scope_type="global", key="ordinary", value=1),
            RuntimeSettingBatchItem(scope_type="global", key=tokens.GENERATION_KEY, value="bad"),
        ]))
    assert writes == []


def test_invalid_generation_startup_export_withdraws_not_legacy(generation_db, tmp_path):
    put(generation_db, "bad")
    server, bridge = tmp_path / "server", tmp_path / "bridge"
    server.write_text("stale-disposable")
    bridge.write_text("stale-disposable")
    result = tokens.export_crawl4ai_token({tokens.TARGETS[0].path_env: str(server),
                                         tokens.TARGETS[1].path_env: str(bridge)})
    assert result == {"crawl4ai": "removed", "bridges": "removed"}
    assert not server.exists() and not bridge.exists()


def test_default_derivation_uses_current_generation(generation_db, monkeypatch):
    put(generation_db, 2)
    monkeypatch.setattr(tokens, "derive_for_generation", lambda n: n)
    assert tokens._derive() == 2


@pytest.fixture
def published(monkeypatch, tmp_path):
    monkeypatch.setattr(tokens, "current_generation", lambda: 1)
    monkeypatch.setattr(tokens, "derive_for_generation", lambda n: f"disposable-{n}")
    for target in tokens.TARGETS:
        monkeypatch.setenv(target.path_env, str(tmp_path / target.slug))
        monkeypatch.delenv(target.uid_env, raising=False)
    assert rekey.export(1)["ok"]
    return tmp_path


def test_partial_publication_keeps_generation_and_recovers_same_generation(published, monkeypatch):
    monkeypatch.setattr(tokens, "current_generation", lambda: 2)
    monkeypatch.setenv("CRAWL4AI_API_TOKEN_EXPORT_UID", "bad-uid")
    result = rekey.export(2)
    assert result == {"ok": False, "generation_matches": True, "crawl4ai": False, "bridges": True}
    assert (published / "crawl4ai").read_text().strip() == "disposable-1"
    assert (published / "bridges").read_text().strip() == "disposable-2"
    monkeypatch.delenv("CRAWL4AI_API_TOKEN_EXPORT_UID")
    assert rekey.export(2)["ok"]
    assert all((published / t.slug).read_text().strip() == "disposable-2" for t in tokens.TARGETS)


def test_generation_mismatch_does_not_publish(published, monkeypatch):
    monkeypatch.setattr(tokens, "derive_for_generation", lambda _: pytest.fail("must not derive"))
    assert not rekey.export(2)["ok"]


@pytest.mark.parametrize("statuses,ok", [([200, 401, 401], True), ([200, 200, 401], False),
                                       ([200, 401, 200], False), ([302, 401, 401], False),
                                       ([503, 401, 401], False)])
def test_probe_requires_new_accepted_old_and_anonymous_rejected(published, statuses, ok):
    seen = []
    remaining = iter(statuses)

    def handler(request):
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(next(remaining), text="must-not-appear")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = rekey.probe(0, 1, client=client)
    assert result["ok"] is ok
    assert seen == ["Bearer disposable-1", "Bearer disposable-0", None]
    assert all(type(value) is bool for value in result.values())
    assert "disposable" not in json.dumps(result)


def test_probe_export_mismatch_is_not_success(published):
    (published / "bridges").write_text("stale")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        assert not rekey.probe(0, 1, client=client)["ok"]


def test_equal_generation_never_probes_network(published):
    assert not rekey.probe(1, 1)["ok"]


def test_main_suppresses_secret_diagnostics(published, monkeypatch, capsys):
    import logging

    def broken(_):
        print("disposable-secret-diagnostic")
        logging.critical("disposable-secret-diagnostic")  # noqa: LOG015 - root logger is what main must silence
        raise RuntimeError("disposable-secret-diagnostic")

    monkeypatch.setattr(rekey, "export", broken)
    assert rekey.main(["export", "1"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": False}
    assert not captured.err
    assert "disposable-secret" not in captured.out


@pytest.mark.parametrize("args", [[], ["export", "not-a-counter"], ["probe", "0"], ["unknown"]])
def test_bad_cli_args_only_boolean_failure(args, capsys):
    assert rekey.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}
