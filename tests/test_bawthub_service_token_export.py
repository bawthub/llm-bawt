"""TASK-1472: the app exports the BawtHub service token to a shared file."""

from __future__ import annotations

import logging
import os
import stat

import pytest

import httpx

from llm_bawt.bawthub_auth import (
    EXPORT_FILE_ENV,
    EXPORT_UID_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    BawtHubServiceAuth,
    ExportResult,
    ServiceTokenSource,
    TokenCheck,
    export_service_token,
    service_token,
)

DERIVED = "fixture-derived-token-" + "e" * 42
OVERRIDE = "fixture-override-token-0123456789"


@pytest.fixture
def derived(monkeypatch):
    monkeypatch.setattr(service_token, "derive_service_token", lambda: DERIVED)


@pytest.fixture
def target(tmp_path):
    directory = tmp_path / "bawthub-service-token"
    directory.mkdir()
    return directory / "token"


def _export(target, source_env=None, **extra):
    environ = {EXPORT_FILE_ENV: str(target), **extra, **(source_env or {})}
    return export_service_token(ServiceTokenSource(environ))


def _leftovers(target):
    return sorted(p.name for p in target.parent.iterdir() if p.name != target.name)


def test_no_export_path_does_nothing(derived, tmp_path):
    assert export_service_token(ServiceTokenSource({})) is ExportResult.DISABLED
    assert ServiceTokenSource({}).read() == DERIVED
    assert list(tmp_path.iterdir()) == []


def test_writes_the_derived_token_mode_0600(derived, target):
    assert _export(target) is ExportResult.WRITTEN
    assert target.read_text().strip() == DERIVED
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert _leftovers(target) == []


def test_exports_the_explicit_override_so_bawthub_matches(derived, target):
    assert _export(target, {TOKEN_ENV: OVERRIDE}) is ExportResult.WRITTEN
    assert target.read_text().strip() == OVERRIDE


def test_replace_is_atomic_rename_not_in_place_write(derived, target):
    target.write_text("old-token")
    before = target.stat().st_ino
    with target.open() as reader_holding_old_file:
        assert _export(target) is ExportResult.WRITTEN
        assert reader_holding_old_file.read() == "old-token"  # never saw a torn write
    assert target.stat().st_ino != before
    assert target.read_text().strip() == DERIVED


def test_failed_replace_keeps_the_old_file_and_cleans_up(derived, target, monkeypatch):
    target.write_text("old-token")

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    assert _export(target) is ExportResult.FAILED
    assert target.read_text() == "old-token"
    assert _leftovers(target) == []


def test_chowns_to_the_reader_uid(derived, target, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: calls.append((uid, gid)))
    assert _export(target, **{EXPORT_UID_ENV: "1001"}) is ExportResult.WRITTEN
    assert calls == [(1001, -1)]


def test_chown_to_own_uid_really_applies(derived, target):
    assert _export(target, **{EXPORT_UID_ENV: str(os.getuid())}) is ExportResult.WRITTEN
    assert target.stat().st_uid == os.getuid()


@pytest.mark.parametrize("uid", ["hono", "-1", "1001:1001", str(2**32 - 1), str(2**65)])
def test_non_numeric_uid_refuses_to_write(derived, target, uid):
    assert _export(target, **{EXPORT_UID_ENV: uid}) is ExportResult.FAILED
    assert not target.exists()


def test_chown_failure_is_reported_not_raised(derived, target, monkeypatch):
    def denied(path, uid, gid):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chown", denied)
    assert _export(target, **{EXPORT_UID_ENV: "1001"}) is ExportResult.FAILED
    assert not target.exists()
    assert _leftovers(target) == []


def test_no_token_removes_a_stale_export(target):
    """conftest makes derivation fail: both sides must fail closed together."""
    target.write_text("token-from-before-the-key-broke")
    assert _export(target) is ExportResult.REMOVED
    assert not target.exists()


def test_no_token_removes_a_stale_export_even_with_a_bad_uid(target):
    target.write_text("token-from-before-the-key-broke")
    assert _export(target, **{EXPORT_UID_ENV: "hono"}) is ExportResult.REMOVED
    assert not target.exists()


def test_unwritable_location_is_reported_not_raised(derived, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    assert _export(blocker / "token") is ExportResult.FAILED


def test_creates_a_missing_parent_directory(derived, tmp_path):
    target = tmp_path / "run" / "bawthub-service-token" / "token"
    assert _export(target) is ExportResult.WRITTEN
    assert target.read_text().strip() == DERIVED


def test_token_value_is_never_logged(derived, target, caplog):
    caplog.set_level(logging.DEBUG)
    _export(target)
    _export(target, {TOKEN_ENV: OVERRIDE})
    assert DERIVED not in caplog.text
    assert OVERRIDE not in caplog.text
    assert str(target) in caplog.text


# --- the export follows the token in effect (rotation without a restart) -----

ROTATED = "fixture-rotated-token-9876543210"


@pytest.fixture
def rotating(tmp_path, target):
    """An override file plus an export, as a deployment with an operator override."""
    override = tmp_path / "override"
    override.write_text(OVERRIDE)
    source = ServiceTokenSource({TOKEN_FILE_ENV: str(override), EXPORT_FILE_ENV: str(target)})
    assert export_service_token(source) is ExportResult.WRITTEN
    return source, override


def test_rotating_the_override_rewrites_the_export_before_use(rotating, target):
    source, override = rotating
    override.write_text(ROTATED)
    assert source.read() == ROTATED
    assert target.read_text().strip() == ROTATED  # already there when the token is handed out
    assert source.check(f"Bearer {ROTATED}") is TokenCheck.OK
    assert source.check(f"Bearer {OVERRIDE}") is TokenCheck.INVALID


def test_outgoing_bearer_never_runs_ahead_of_the_export(rotating, target):
    source, override = rotating
    override.write_text(ROTATED)
    seen = []

    def bawthub(request):
        # What BawtHub would compare against at the moment the request lands.
        seen.append((request.headers["Authorization"], target.read_text().strip()))
        return httpx.Response(204)

    with httpx.Client(transport=httpx.MockTransport(bawthub), auth=BawtHubServiceAuth(source)) as client:
        client.get("http://bawthub/api/tasks/tasks")
    assert seen == [(f"Bearer {ROTATED}", ROTATED)]


def test_an_override_breaking_withdraws_the_export(rotating, target):
    source, override = rotating
    override.write_text("")
    assert source.read() is None
    assert not target.exists()


def test_unchanged_token_does_not_rewrite(rotating, target):
    source, _ = rotating
    inode = target.stat().st_ino
    for _ in range(3):
        assert source.read() == OVERRIDE
    assert target.stat().st_ino == inode


def _flaky_replace(monkeypatch, failures):
    real_replace = os.replace
    plan = iter(failures)

    def flaky(src, dst):
        if next(plan, False):
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)


def test_failed_rotation_keeps_the_exported_token_in_effect(rotating, target, monkeypatch):
    """Until BawtHub's file holds the new token, llm-bawt neither sends nor accepts it."""
    source, override = rotating
    _flaky_replace(monkeypatch, [True] * 10)
    override.write_text(ROTATED)
    seen = []

    def bawthub(request):
        seen.append((request.headers["Authorization"], target.read_text().strip()))
        return httpx.Response(204)

    with httpx.Client(transport=httpx.MockTransport(bawthub), auth=BawtHubServiceAuth(source)) as client:
        client.get("http://bawthub/api/tasks/tasks")
    assert seen == [(f"Bearer {OVERRIDE}", OVERRIDE)]
    assert source.read() == OVERRIDE
    assert source.check(f"Bearer {OVERRIDE}") is TokenCheck.OK
    assert source.check(f"Bearer {ROTATED}") is TokenCheck.INVALID


def test_failed_sync_is_logged_once_and_retried(rotating, target, monkeypatch, caplog):
    source, override = rotating
    _flaky_replace(monkeypatch, [True, True, False])
    caplog.set_level(logging.ERROR, logger="llm_bawt.bawthub_auth.service_token")
    override.write_text(ROTATED)
    assert source.read() == OVERRIDE
    assert source.read() == OVERRIDE
    assert target.read_text().strip() == OVERRIDE  # both writes failed
    assert len(caplog.records) == 1
    assert source.read() == ROTATED
    assert target.read_text().strip() == ROTATED  # third try lands
    assert ROTATED not in caplog.text and OVERRIDE not in caplog.text
    assert _leftovers(target) == []


def test_no_token_until_the_first_export_lands(derived, target, monkeypatch):
    """A startup write that fails leaves BawtHub's copy unknown: fail closed together."""
    target.write_text("token-from-a-previous-process")
    _flaky_replace(monkeypatch, [True, True])
    source = ServiceTokenSource({EXPORT_FILE_ENV: str(target)})
    assert export_service_token(source) is ExportResult.FAILED
    assert source.check(f"Bearer {DERIVED}") is TokenCheck.UNCONFIGURED  # its retry fails too
    assert target.read_text().strip() == "token-from-a-previous-process"
    assert source.read() == DERIVED  # the next retry lands
    assert target.read_text().strip() == DERIVED


def test_failed_withdrawal_keeps_the_exported_token(rotating, target, monkeypatch):
    """If the stale file can't be removed, BawtHub still accepts it, so llm-bawt keeps it."""
    source, override = rotating

    def stuck(self, missing_ok=False):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(target), "unlink", stuck)
    override.write_text("")
    assert source.read() == OVERRIDE
    assert target.read_text().strip() == OVERRIDE


def test_both_sides_agree_through_rotation_with_failing_writes(tmp_path, target, monkeypatch):
    """Loopy's TASK-1472 B1 edge, end to end: override rotation while export writes fail.

    BawtHub accepts exactly what its file holds. At every step llm-bawt's token in
    effect is either that token or none (fail closed), never one BawtHub lacks.
    """
    target.write_text("token-from-a-previous-process\n")
    override = tmp_path / "override"
    override.write_text(OVERRIDE)
    source = ServiceTokenSource({TOKEN_FILE_ENV: str(override), EXPORT_FILE_ENV: str(target)})
    failing = {"on": True}
    real_replace = os.replace

    def replace(src, dst):
        if failing["on"]:
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    sent = []

    def bawthub(request):
        held = target.read_text().strip() if target.exists() else None
        bearer = request.headers.get("Authorization")
        sent.append(bearer)
        return httpx.Response(204 if held and bearer == f"Bearer {held}" else 401)

    client = httpx.Client(transport=httpx.MockTransport(bawthub), auth=BawtHubServiceAuth(source))

    def agree():
        in_effect = source.read()  # may export first
        held = target.read_text().strip() if target.exists() else None
        assert in_effect is None or in_effect == held
        if in_effect is None:  # nothing to send, nothing accepted
            assert source.check(f"Bearer {held}") is TokenCheck.UNCONFIGURED
            assert client.get("http://bawthub/x").request.headers.get("Authorization") is None
        else:  # each side accepts the other's bearer, and only that one
            assert source.check(f"Bearer {held}") is TokenCheck.OK
            assert client.get("http://bawthub/x").status_code == 204
        return in_effect

    assert export_service_token(source) is ExportResult.FAILED
    assert agree() is None  # startup export failed: fail closed together
    failing["on"] = False
    assert agree() == OVERRIDE  # first export lands
    failing["on"] = True
    override.write_text(ROTATED)  # operator rotates while the volume is full
    assert agree() == OVERRIDE
    assert source.check(f"Bearer {ROTATED}") is TokenCheck.INVALID
    assert agree() == OVERRIDE  # retried, still failing
    failing["on"] = False
    assert agree() == ROTATED  # write lands; both move together
    assert source.check(f"Bearer {OVERRIDE}") is TokenCheck.INVALID
    client.close()
