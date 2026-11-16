"""TASK-2386: the app hands the ops reconciler its secrets; the reconciler holds no master key."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

from llm_bawt.ops import control_protocol, executor, gate_trust, github_workflow, release, secret_exports
from llm_bawt.ops.secret_exports import (
    CONTROL_TOKEN, FILES, GATE_PROVENANCE_KEY, GHCR_PULL, GITHUB_RELEASE, GITHUB_RELEASE_DISPATCH, Unreadable,
    export_all, read, read_pull_auth,
)
from llm_bawt.service.providers import crypto

# Synthetic values only.
PULL = {"username": "someone", "password": "p" * 9}


@pytest.fixture
def app_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(crypto, "_fernet", None)
    monkeypatch.setattr(crypto, "_generation_guard", None)
    monkeypatch.delenv(secret_exports.READ_DIR_ENV, raising=False)


def _sources(**over):
    base = {
        CONTROL_TOKEN: lambda: secret_exports._derived(control_protocol.TOKEN_LABEL),
        GATE_PROVENANCE_KEY: lambda: secret_exports._derived(gate_trust._KEY_LABEL),
        GHCR_PULL: lambda: json.dumps(PULL, sort_keys=True),
        GITHUB_RELEASE: lambda: "r" * 11,
        GITHUB_RELEASE_DISPATCH: lambda: "d" * 12,
    }
    base.update(over)
    return base


def test_reconciler_reads_what_the_app_derives(app_key, tmp_path, monkeypatch):
    out = tmp_path / "ops-secrets"
    expected_token = control_protocol.control_token()
    expected_key = crypto.derive_key(gate_trust._KEY_LABEL)
    assert export_all(out, _sources()) == dict.fromkeys(FILES, "written")
    for name in FILES:
        assert stat.S_IMODE(os.stat(out / name).st_mode) == 0o600

    # The reconciler: no master key at all, only the read-only export.
    monkeypatch.delenv("LLM_BAWT_SECRET_KEY")
    monkeypatch.setattr(crypto, "_fernet", None)
    crypto.refuse_generation_when(lambda: True)
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(out))
    assert control_protocol.control_token() == expected_token
    assert gate_trust._provenance_key() == expected_key
    assert executor._stored_pull_auth() == PULL
    assert release._stored_release_token() == "r" * 11
    assert github_workflow._stored_dispatch_token() == "d" * 12
    assert not (tmp_path / "home" / ".config" / "llm-bawt" / "secret.key").exists()


def test_real_sources_read_the_stored_credentials(app_key, tmp_path, monkeypatch):
    """The default sources, not injected lambdas: imports, Config and adapters resolve."""
    from llm_bawt.service.providers import github_deploy

    monkeypatch.setattr(github_deploy.GhcrPullAdapter, "pull_auth", lambda self: dict(PULL))
    monkeypatch.setattr(github_deploy.GitHubReleaseAdapter, "api_key", lambda self: "r" * 11)
    monkeypatch.setattr(github_deploy.GitHubReleaseDispatchAdapter, "api_key", lambda self: "d" * 12)
    out = tmp_path / "ops-secrets"
    assert export_all(out) == dict.fromkeys(FILES, "written")
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(out))
    assert read_pull_auth() == PULL
    assert read(GITHUB_RELEASE) == "r" * 11 and read(GITHUB_RELEASE_DISPATCH) == "d" * 12


def test_missing_exports_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(tmp_path))
    with pytest.raises(RuntimeError, match="not exported"):
        control_protocol.control_token()
    with pytest.raises(gate_trust.GateTrustError, match="gate_provenance_key_unavailable"):
        gate_trust._mac({"a": 1})
    assert executor._stored_pull_auth() is None
    assert release._stored_release_token() == ""
    assert github_workflow._stored_dispatch_token() == ""


@pytest.mark.parametrize("raw", ["not json", "[]", json.dumps({"username": "u"}),
                                 json.dumps({**PULL, "extra": "x"}), json.dumps({**PULL, "password": ""})])
def test_malformed_pull_auth_is_unavailable(tmp_path, monkeypatch, raw):
    (tmp_path / GHCR_PULL).write_text(raw)
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(tmp_path))
    assert read_pull_auth() is None


def test_disconnect_removes_and_unreadable_keeps_the_last_copy(app_key, tmp_path):
    out = tmp_path / "ops-secrets"
    export_all(out, _sources())

    def unreadable():
        raise Unreadable("OperationalError")

    def no_key():
        raise crypto.MissingRootKeyError("x")

    result = export_all(out, _sources(**{GITHUB_RELEASE: lambda: None, GHCR_PULL: unreadable,
                                         CONTROL_TOKEN: no_key}))
    assert result[GITHUB_RELEASE] == "removed" and not (out / GITHUB_RELEASE).exists()
    assert result[GHCR_PULL] == "kept" and json.loads((out / GHCR_PULL).read_text()) == PULL
    assert result[CONTROL_TOKEN] == "kept" and (out / CONTROL_TOKEN).read_text().strip()


def test_unset_export_dir_does_nothing(monkeypatch):
    monkeypatch.delenv(secret_exports.EXPORT_DIR_ENV, raising=False)
    assert export_all() == {}


def test_read_refuses_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError):
        read("secret.key")


def test_only_exported_credentials_schedule_a_reexport(tmp_path, monkeypatch):
    started = []

    class _Timer:
        def __init__(self, delay, fn):
            self.daemon = False
            self.fn = fn

        def start(self):
            started.append(self.fn)

    monkeypatch.setattr(secret_exports.threading, "Timer", _Timer)
    monkeypatch.setattr(secret_exports, "_pending", None)
    monkeypatch.setenv(secret_exports.EXPORT_DIR_ENV, str(tmp_path))
    secret_exports.credential_changed("openai")
    assert started == []
    secret_exports.credential_changed("ghcr-pull")
    secret_exports.credential_changed("github-release")  # coalesced into the pending one
    assert len(started) == 1
    monkeypatch.setattr(secret_exports, "export_all", lambda: None)
    started[0]()
    assert secret_exports._pending is None
    monkeypatch.delenv(secret_exports.EXPORT_DIR_ENV)
    secret_exports.credential_changed("ghcr-pull")
    assert len(started) == 1


def test_credential_store_writes_and_deletes_notify(monkeypatch):
    from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore

    seen = []
    monkeypatch.setattr(secret_exports, "credential_changed", seen.append)
    CredentialStore._payload(ConnectionRecord(provider="ghcr-pull", status="disconnected"))
    assert seen == ["ghcr-pull"]


def test_compose_gives_the_reconciler_the_export_not_the_key():
    root = Path(__file__).resolve().parent.parent
    ops = yaml.safe_load((root / "docker-compose.ops.yml").read_text())["services"]
    reconciler = ops["ops-reconciler"]
    assert not [v for v in reconciler["volumes"] if "secret.key" in v or ".config/llm-bawt" in v]
    assert "ops-secrets:/run/llm-bawt-ops-secrets:ro" in reconciler["volumes"]
    assert reconciler["environment"][secret_exports.READ_DIR_ENV] == "/run/llm-bawt-ops-secrets"
    assert "ops-secrets:/run/llm-bawt-ops-secrets" in ops["app"]["volumes"]
    assert ops["app"]["environment"][secret_exports.EXPORT_DIR_ENV] == "/run/llm-bawt-ops-secrets"
    for name, service in yaml.safe_load((root / "docker-compose.yml").read_text())["services"].items():
        assert not [v for v in service.get("volumes") or [] if "ops-secrets" in str(v)], name
