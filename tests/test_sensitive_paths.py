"""Sensitive path policy and the local-only snapshot publication boundary."""

from pathlib import Path
import subprocess

import pytest

from agent_bridge.sensitive_paths import is_sensitive_path


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("path", [".env", ".env.bak", "nested/.env.local", "private.key", "nested/PRIVATE.KEY"])
def test_environment_and_key_data_stay_sensitive(path, tracked):
    assert is_sensitive_path(path, tracked=tracked)


@pytest.mark.parametrize("path", [".env.example", ".env.template", "secret-data.example", "private.key.template"])
def test_templates_are_allowed(path):
    assert not is_sensitive_path(path)
    assert not is_sensitive_path(path, tracked=True)


@pytest.mark.parametrize("path", ["secret_store.py", "connection_secrets.py", "tests/test_secrets.py", ".env.docker"])
def test_allowed_source_exceptions_require_tracked_provenance(path):
    assert is_sensitive_path(path)
    assert not is_sensitive_path(path, tracked=True)


@pytest.mark.parametrize(
    ("local", "remote", "rejected"),
    [
        ("refs/heads/main", "refs/heads/main", False),
        ("refs/tags/v1", "refs/tags/v1", False),
        ("refs/heads/main", "refs/heads/refs/bawt/not-private", False),
        ("refs/bawtish/normal", "refs/heads/normal", False),
        ("refs/bawt/turns/t/start", "refs/heads/export", True),
        ("refs/heads/main", "refs/bawt/turns/t/start", True),
        ("(delete)", "refs/bawt/turns/t/start", True),
    ],
)
def test_pre_push_rejects_either_private_ref_side(local, remote, rejected):
    hook = Path(__file__).resolve().parents[1] / ".githooks" / "pre-push"
    assert hook.stat().st_mode & 0o111
    result = subprocess.run(
        (str(hook), "origin", "/unused"),
        input=f"{local} {'1' * 40} {remote} {'0' * 40}\n",
        text=True, capture_output=True,
    )
    assert (result.returncode != 0) is rejected


def test_pre_push_checks_all_ref_updates():
    hook = Path(__file__).resolve().parents[1] / ".githooks" / "pre-push"
    result = subprocess.run(
        (str(hook), "origin", "/unused"),
        input=f"refs/heads/main {'1' * 40} refs/heads/main {'0' * 40}\n"
              f"refs/heads/main {'1' * 40} refs/bawt/turns/t/end {'0' * 40}\n",
        text=True, capture_output=True,
    )
    assert result.returncode != 0
