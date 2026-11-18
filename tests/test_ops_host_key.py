"""TASK-2430: fixed host-key action — validation, argv and remote scripts."""
from __future__ import annotations

import base64
import os
import subprocess

import pytest

from llm_bawt.ops import host_key as hk


def _blob(seed: bytes = b"a") -> str:
    name = b"ssh-ed25519"
    raw = len(name).to_bytes(4, "big") + name + (32).to_bytes(4, "big") + (seed * 32)[:32]
    return base64.b64encode(raw).decode()


BLOB = _blob()
OTHER = _blob(b"b")
KEY = f"ssh-ed25519 {BLOB} bridge@bots"


def _target(**over):
    target = {"host_id": 1, "host": "box.example", "port": 22, "login_user": "deploy",
              "key_fingerprint": hk.fingerprint(BLOB), "known_host_keys": [f"ssh-ed25519 {OTHER}"]}
    target.update(over)
    return target


def _run(op: str, home) -> str:
    script = hk.remote_command(op, "ssh-ed25519", BLOB)
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "LC_ALL": "C"}
    proc = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_spec_is_closed():
    assert hk.validate_spec({"action": "host_key", "op": "verify", "key_container": "bridge"})
    for bad in ({"action": "host_key", "op": "exec", "key_container": "b"},
                {"action": "host_key", "op": "add", "key_container": "b", "target_host": "x"},
                {"action": "host_key", "op": "add", "key_container": "-x"}):
        with pytest.raises(ValueError):
            hk.validate_spec(bad)


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "a b", "a;b", "", "x" * 254, "a..b"])
def test_host_rejects_unsafe(host):
    with pytest.raises(ValueError):
        hk.validate_host(host)


def test_host_accepts_names_and_ips():
    for host in ("box", "box.lan.example", "10.0.0.5", "fe80::1"):
        assert hk.validate_host(host) == host


def test_target_rejects_bad_user_and_keys():
    with pytest.raises(ValueError):
        hk.validate_target(_target(login_user="Root;x"))
    with pytest.raises(ValueError):
        hk.validate_target(_target(known_host_keys=["ssh-ed25519 x; rm"]))
    with pytest.raises(ValueError):
        hk.validate_target({**_target(), "command": "id"})


def test_public_key_parsing_ignores_comment_and_checks_blob():
    assert hk.parse_public_key(KEY + "\n") == ("ssh-ed25519", BLOB)
    assert hk.parse_public_key(f"ssh-ed25519 {BLOB}") == ("ssh-ed25519", BLOB)
    with pytest.raises(ValueError):
        hk.parse_public_key(f"ssh-ed25519 {BLOB} evil'quote")
    rsa_named = base64.b64encode(b"\x00\x00\x00\x07ssh-rsa" + b"\x00" * 40).decode()
    with pytest.raises(ValueError):
        hk.parse_public_key(f"ssh-ed25519 {rsa_named[:68]}")


def test_argv_forces_identity_and_strict_pinned_trust():
    argv = hk.ssh_argv(_target(), "remove", "cmd", "/tmp/j")
    joined = " ".join(argv)
    assert argv[:4] == ["timeout", "-k", "5", "45"]
    for opt in ("-F none", "IdentitiesOnly=yes", "IdentityAgent=none", "ControlPath=none",
                "PasswordAuthentication=no", "StrictHostKeyChecking=yes", "GlobalKnownHostsFile=none",
                "UserKnownHostsFile=/tmp/j/known_hosts"):
        assert opt in joined
    assert argv[-3:] == ["--", "box.example", "cmd"]


def test_first_use_only_for_verify():
    target = _target(known_host_keys=[])
    assert "StrictHostKeyChecking=accept-new" in hk.ssh_argv(target, "verify", "c", "/tmp/j")
    for op in ("add", "remove"):
        with pytest.raises(ValueError):
            hk.ssh_argv(target, op, "c", "/tmp/j")


def test_known_hosts_lines_use_port_form():
    assert hk.known_hosts_text(_target(port=2222)).startswith("[box.example]:2222 ssh-ed25519 ")
    assert hk.known_hosts_text(_target()).startswith("box.example ssh-ed25519 ")


def test_add_is_idempotent_and_preserves_other_keys(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "authorized_keys").write_text(f"ssh-ed25519 {OTHER} someone\n")
    assert "HK_ADD=added" in _run("add", tmp_path)
    assert "HK_ADD=present" in _run("add", tmp_path)
    lines = (ssh / "authorized_keys").read_text().splitlines()
    assert lines[0] == f"ssh-ed25519 {OTHER} someone"
    assert sum(BLOB in line for line in lines) == 1
    assert oct((ssh / "authorized_keys").stat().st_mode & 0o777) == "0o600"


def test_add_creates_missing_ssh_dir(tmp_path):
    out = _run("add", tmp_path)
    assert "HK_DONE=add" in out
    assert BLOB in (tmp_path / ".ssh" / "authorized_keys").read_text()


def test_remove_matches_blob_across_comments_and_options(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    keys = ssh / "authorized_keys"
    keys.write_text(f"ssh-ed25519 {BLOB} bridge@one\n"
                    f"ssh-ed25519 {OTHER} keep-me\n"
                    f'from="10.0.0.0/8" ssh-ed25519 {BLOB} another-comment\n'
                    f"# comment line {BLOB}x\n")
    keys.chmod(0o640)
    out = _run("remove", tmp_path)
    assert "HK_REMOVE=removed" in out and "HK_DONE=remove" in out
    text = keys.read_text()
    assert f" {BLOB} " not in text
    assert f"ssh-ed25519 {OTHER} keep-me" in text
    assert f"# comment line {BLOB}x" in text
    assert oct(keys.stat().st_mode & 0o777) == "0o640"
    assert BLOB in (ssh / "authorized_keys.bawthub-bak").read_text()


def test_remove_absent_file(tmp_path):
    assert "HK_REMOVE=absent" in _run("remove", tmp_path)


def test_verify_markers_are_closed_fields(tmp_path):
    fields = hk.verify_fields(hk.parse_markers(_run("verify", tmp_path)))
    assert set(fields) == {"user", "groups", "passwordless_sudo_probe", "docker"}


@pytest.mark.parametrize("code,stderr,reason", [
    (255, "deploy@h: Permission denied (publickey).", "key_refused"),
    (255, "Host key verification failed.", "host_key_mismatch"),
    (255, "ssh: Could not resolve hostname x", "dns_failed"),
    (255, "connect to host x port 22: Connection refused", "connection_refused"),
    (124, "", "timeout"),
    (255, "kex_exchange_identification: read: Connection reset", "ssh_failed"),
])
def test_failure_classification(code, stderr, reason):
    assert hk.classify_failure(code, stderr) == reason


class FakeApi:
    """Docker API stub: answers the key read and scripted ssh results."""

    def __init__(self, ssh_results, pubkey=KEY):
        self.ssh_results = list(ssh_results)
        self.pubkey = pubkey
        self.calls = []
        self.side_effect_started = False
        self._next = None

    def request(self, method, path, *, body=None, raw=False, headers=None):
        self.calls.append((method, path, body))
        if path.endswith("/exec"):
            argv = body["Cmd"]
            if argv[0] == "cat":
                self._next = (0, self.pubkey if argv[-1].endswith(".pub") else
                              f"box.example ssh-ed25519 {OTHER}\n", "")
            elif argv[0] == "timeout":
                self._next = self.ssh_results.pop(0)
            else:
                self._next = (0, "", "")
            return {"Id": "e1"}
        if path.endswith("/start"):
            return b""
        if path.endswith("/json"):
            return {"ExitCode": self._next[0]}
        raise AssertionError(path)


def _demux_factory(api):
    return lambda _raw: (api._next[1], api._next[2])


def _snapshot(op, **target):
    return {"spec": {"action": "host_key", "op": op, "key_container": "bridge"},
            "execution": {"timeout_seconds": 150}, "host_key": _target(**target)}


def _run_worker(op, results, **target):
    api = FakeApi(results)
    receipt = {}
    out = hk.run_worker(api, _snapshot(op, **target), "job-1", receipt, demux=_demux_factory(api))
    return api, receipt, out


def test_worker_refuses_changed_bot_key():
    api = FakeApi([])
    snap = _snapshot("verify", key_fingerprint=hk.fingerprint(OTHER))
    with pytest.raises(hk.HostKeyFailed):
        hk.run_worker(api, snap, "j", {}, demux=_demux_factory(api))
    assert not any(c[2] and c[2].get("Cmd", [""])[0] == "timeout" for c in api.calls)


def test_worker_verify_learns_first_use_host_keys():
    _api, receipt, _out = _run_worker("verify", [(0, "HK_USER=deploy\nHK_GROUPS=deploy,docker\n"
                                                    "HK_SUDO_N=0\nHK_DOCKER=1\nHK_DONE=verify\n", "")],
                                      known_host_keys=[])
    assert receipt["host_key"]["learned_host_keys"] == [f"ssh-ed25519 {OTHER}"]
    assert receipt["host_key"]["docker"] is True


def test_worker_remove_requires_ack_and_corroborates():
    _api, receipt, _ = _run_worker("remove", [(0, "HK_REMOVE=removed\nHK_DONE=remove\n", ""),
                                              (255, "", "Permission denied (publickey).")])
    assert receipt["host_key"]["corroboration"] == "login_refused"


def test_worker_remove_network_drop_is_unknown_not_success():
    with pytest.raises(hk.HostKeyUnknown):
        _run_worker("remove", [(255, "", "Connection reset by peer")])


def test_worker_remove_refused_key_before_script_is_unknown():
    with pytest.raises(hk.HostKeyUnknown):
        _run_worker("remove", [(255, "", "Permission denied (publickey).")])


def test_worker_remove_still_logging_in_fails():
    with pytest.raises(hk.HostKeyFailed):
        _run_worker("remove", [(0, "HK_REMOVE=removed\nHK_DONE=remove\n", ""),
                               (0, "HK_USER=deploy\nHK_DONE=verify\n", "")])


def test_worker_add_timeout_is_unknown():
    with pytest.raises(hk.HostKeyUnknown):
        _run_worker("add", [(124, "", "")])


# --- Codex implementation review cmwlnc39b01750mqwyr3lcub9 ---

def test_add_after_final_line_without_newline_makes_a_real_entry(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    keys = ssh / "authorized_keys"
    keys.write_text(f"ssh-ed25519 {OTHER} existing-owner")  # no trailing newline
    assert "HK_ADD=added" in _run("add", tmp_path)
    lines = keys.read_text().splitlines()
    assert lines == [f"ssh-ed25519 {OTHER} existing-owner", f"ssh-ed25519 {BLOB} bawthub-bots"]


def test_blob_in_another_keys_comment_is_not_a_match(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    keys = ssh / "authorized_keys"
    owner = f"ssh-ed25519 {OTHER} owner copied-key {BLOB}"
    keys.write_text(owner + "\n" + f"# {BLOB}\n")
    assert "HK_ADD=added" in _run("add", tmp_path)
    assert "HK_REMOVE=removed" in _run("remove", tmp_path)
    assert keys.read_text().splitlines() == [owner, f"# {BLOB}"]


def test_quoted_options_with_spaces_still_find_the_key(tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    keys = ssh / "authorized_keys"
    keys.write_text(f'command="echo a b",no-pty ssh-ed25519 {BLOB} x\n  ssh-ed25519 {BLOB} indented\n')
    assert "HK_ADD=present" in _run("add", tmp_path)
    _run("remove", tmp_path)
    assert BLOB not in keys.read_text()


def test_worker_remove_inconclusive_corroboration_is_unknown():
    with pytest.raises(hk.HostKeyUnknown):
        _run_worker("remove", [(0, "HK_REMOVE=removed\nHK_DONE=remove\n", ""), (124, "", "")])


def test_approval_text_names_frozen_destination_and_key():
    text = hk.approval_text({"action": "host_key", "op": "remove", "key_container": "b"}, _target(port=2222), None)
    assert "deploy@box.example:2222" in text and hk.fingerprint(BLOB) in text and "pinned host key" in text
    first = hk.approval_text({"action": "host_key", "op": "verify", "key_container": "b"},
                             _target(known_host_keys=[]), "Verify?")
    assert first.startswith("Verify?") and "first use" in first
