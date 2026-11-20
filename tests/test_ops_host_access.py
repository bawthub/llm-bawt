"""TASK-2430: host access records, approval pinning and job settlement."""
import base64
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.ops import host_access as ha
from llm_bawt.ops import host_key as hk

FP = hk.fingerprint(base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"a" * 32).decode())
HOSTKEY = "ssh-ed25519 " + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"h" * 32).decode()
HOSTKEY2 = "ssh-ed25519 " + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"k" * 32).decode()


@pytest.fixture
def engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _host(engine, **over):
    return ha.create_host(engine, {"label": "build-box", "host": "192.0.2.10", "port": 22,
                                   "login_user": "agent", **over}, actor="owner")


def _job(state, output=None, error=None):
    return SimpleNamespace(id="job-1", state=state, output_tail=output, error_text=error)


def test_create_validates_and_rejects_duplicates(engine):
    row = _host(engine)
    assert row["status"] == "unverified" and row["known_host_keys"] == []
    with pytest.raises(ha.HostAccessError):
        _host(engine)
    for bad in ({"host": "-oProxyCommand=x"}, {"login_user": "Root;"}, {"port": 0}):
        with pytest.raises(ha.HostAccessError):
            _host(engine, label="x", **bad)


def _scan(engine, row, keys=(HOSTKEY,)):
    target = ha.pin(engine, row["id"], FP, "scan")
    result = {"op": "scan", "scanned_host_keys": [{"key": k, "fingerprint": hk.host_key_fingerprint(k)} for k in keys]}
    ha.settle(engine, _job("succeeded", "host keys scanned: " + json.dumps(result)),
              {"spec": {"op": "scan"}, "host_key": target})
    return target


def _trust(engine, row):
    _scan(engine, row)
    return ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY)])


def test_logins_need_a_confirmed_host_key(engine):
    row = _host(engine)
    assert ha.pin(engine, row["id"], FP, "scan")["known_host_keys"] == []
    for op in ("verify", "add", "remove"):
        with pytest.raises(ha.HostAccessError):
            ha.pin(engine, row["id"], FP, op)
    with pytest.raises(ha.HostAccessError):
        ha.pin(engine, "1", FP, "scan")


def test_scan_offers_keys_and_confirm_pins_only_scanned(engine):
    row = _host(engine)
    with pytest.raises(ha.HostAccessError):  # nothing scanned yet
        ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY)])
    scan_target = _scan(engine, row)
    scanned = ha.list_hosts(engine)[0]
    assert scanned["status"] == "scanned" and scanned["known_host_keys"] == []
    assert scanned["last_result"]["scanned_host_keys"][0]["key"] == HOSTKEY
    with pytest.raises(ha.HostAccessError):  # a fingerprint that was not offered
        ha.confirm_host_keys(engine, row["id"], [FP])
    after = ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY)])
    assert after["status"] == "trusted" and after["known_host_keys"] == [HOSTKEY]
    with pytest.raises(ha.HostAccessError):  # the scan's approval no longer matches
        ha.check_pin(engine, scan_target)
    verify = ha.pin(engine, row["id"], FP, "verify")
    ha.check_pin(engine, verify)
    ha.settle(engine, _job("succeeded", "host login verified: " + json.dumps(
        {"op": "verify", "login_ok": True, "passwordless_sudo_probe": True})),
              {"spec": {"op": "verify"}, "host_key": verify})
    final = ha.list_hosts(engine)[0]
    assert final["status"] == "verified" and final["passwordless_sudo"] is True
    assert final["known_host_keys"] == [HOSTKEY]
    with pytest.raises(KeyError):
        ha.confirm_host_keys(engine, 999, [FP])


def test_editing_destination_clears_trust_and_invalidates_approval(engine):
    row = _host(engine)
    _trust(engine, row)
    ha.update_host(engine, row["id"], {"label": "echo2"})
    assert ha.list_hosts(engine)[0]["known_host_keys"] == [HOSTKEY]
    pinned = ha.pin(engine, row["id"], FP, "add")
    ha.update_host(engine, row["id"], {"host": "192.0.2.11"})
    after = ha.list_hosts(engine)[0]
    assert after["known_host_keys"] == [] and after["status"] == "unverified"
    with pytest.raises(ha.HostAccessError):
        ha.check_pin(engine, pinned)
    # A late result for the old destination is not folded in.
    ha.settle(engine, _job("succeeded", "bot key present on host: {}"), {"spec": {"op": "add"}, "host_key": pinned})
    assert ha.list_hosts(engine)[0]["status"] == "unverified"


def test_unknown_outcome_and_failures(engine):
    row = _host(engine)
    _trust(engine, row)
    target = ha.pin(engine, row["id"], FP, "verify")
    ha.settle(engine, _job("failed", error="login check failed: key_refused"),
              {"spec": {"op": "verify"}, "host_key": target})
    assert ha.list_hosts(engine)[0]["status"] == "login_failed"
    ha.settle(engine, _job("lost"), {"spec": {"op": "remove"}, "host_key": target})
    assert ha.list_hosts(engine)[0]["status"] == "unknown"


def test_delete(engine):
    row = _host(engine)
    assert ha.delete_host(engine, row["id"]) and not ha.delete_host(engine, row["id"])


def test_host_key_seeds_pass_catalog_validation():
    from llm_bawt.ops.models import OpsOperation
    from llm_bawt.ops.seeds import SEEDS
    from llm_bawt.ops.store import OpsStore

    seeds = [s for s in SEEDS if s["slug"].startswith("llm-bawt.host-key-")]
    assert sorted(s["slug"] for s in seeds) == [
        "llm-bawt.host-key-add", "llm-bawt.host-key-remove", "llm-bawt.host-key-scan", "llm-bawt.host-key-verify"]
    fields = set(OpsOperation.__table__.columns.keys())
    for seed in seeds:
        assert seed["enabled"] is False
        OpsStore._validate_op(OpsOperation(id="t", **{k: v for k, v in seed.items() if k in fields}))


def test_tenant_seeds_name_the_bridge_and_adopt_untouched_rows(monkeypatch):
    from llm_bawt.ops import seeds as seeds_mod

    monkeypatch.setenv("LLM_BAWT_HOST_ACCESS_KEY_CONTAINER", "acme-agent-bridge")
    seed = seeds_mod._host_key_seed("verify", "t", "d", "low", "p")
    assert seed["enabled"] is True
    assert json.loads(seed["command_script"])["key_container"] == "acme-agent-bridge"
    legacy = json.dumps({"action": "host_key", "op": "verify", "key_container": "llm-bawt-claude-code-bridge"})
    rows = {s["slug"]: SimpleNamespace(slug=s["slug"], enabled=False, soft_deleted_at=None, version=1,
                                       command_script=legacy) for s in seeds_mod._HOST_KEY_SEEDS}
    rows["llm-bawt.host-key-add"].version = 2  # operator-edited: kept
    updates = []
    store = SimpleNamespace(get_operation_by_slug=rows.get,
                            update_operation=lambda slug, data, **kw: updates.append((slug, data, kw)))
    seeds_mod._adopt_host_key_container(store)
    assert sorted(u[0] for u in updates) == [
        "llm-bawt.host-key-remove", "llm-bawt.host-key-scan", "llm-bawt.host-key-verify"]
    assert all(u[1]["enabled"] and u[2]["expect_version"] == 1 for u in updates)
    monkeypatch.delenv("LLM_BAWT_HOST_ACCESS_KEY_CONTAINER")
    updates.clear()
    seeds_mod._adopt_host_key_container(store)
    assert updates == []


def test_bridge_ssh_identity(tmp_path, monkeypatch):
    import shutil

    from agent_bridge.ssh_identity import ensure_ssh_identity

    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen unavailable")
    assert ensure_ssh_identity(tmp_path) == "generated"
    pub = (tmp_path / ".ssh" / "id_ed25519.pub").read_text()
    assert pub.startswith("ssh-ed25519 ")
    assert ensure_ssh_identity(tmp_path) == "present"
    assert (tmp_path / ".ssh" / "id_ed25519.pub").read_text() == pub
    monkeypatch.setenv("BRIDGE_SSH_IDENTITY", "0")
    assert ensure_ssh_identity(tmp_path / "other") == "unavailable"


def test_bot_public_key_reads_through_the_host_key_spec():
    from llm_bawt.ops.executor import ExecutorError
    from llm_bawt.ops.service import OpsDispatchError, OpsService

    line = "ssh-ed25519 " + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"a" * 32).decode()
    spec = {"action": "host_key", "op": "verify", "key_container": "bridge"}
    seen = []

    class Exec:
        def __init__(self, fail=False):
            self.fail = fail

        def host_key_public_key(self, s):
            seen.append(s)
            if self.fail:
                raise ExecutorError("the key container has no bot public key")
            return line

    ops = [SimpleNamespace(command_script='{"action":"wake_on_lan"}', executor_kind="docker"),
           SimpleNamespace(command_script=json.dumps(spec), executor_kind="docker")]
    svc = OpsService.__new__(OpsService)
    svc.admission_only = False
    svc.store = SimpleNamespace(list_operations=lambda **kw: ops)
    svc._executors = {"docker": Exec()}
    assert svc.bot_public_key() == {"public_key": line, "fingerprint": FP} and seen == [spec]
    svc._executors = {"docker": Exec(fail=True)}
    with pytest.raises(OpsDispatchError, match="no bot public key"):
        svc.bot_public_key()
    svc.store = SimpleNamespace(list_operations=lambda **kw: ops[:1])
    with pytest.raises(OpsDispatchError, match="no host access operation"):
        svc.bot_public_key()
    admission = OpsService.__new__(OpsService)
    admission.admission_only = True
    admission.preparer = SimpleNamespace(bot_public_key=lambda: {"public_key": line})
    assert admission.bot_public_key() == {"public_key": line}


def test_scan_of_old_destination_cannot_be_confirmed_after_edit(engine):
    row = _host(engine)
    _scan(engine, row)
    ha.update_host(engine, row["id"], {"host": "192.0.2.11"})
    after = ha.list_hosts(engine)[0]
    assert after["last_op"] is None and after["last_result"] is None
    with pytest.raises(ha.HostAccessError):
        ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY)])


def test_confirm_pins_only_the_one_checked_key(engine):
    row = _host(engine)
    _scan(engine, row, keys=(HOSTKEY, HOSTKEY2))
    with pytest.raises(ha.HostAccessError):  # all offered keys at once
        ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY),
                                                 hk.host_key_fingerprint(HOSTKEY2)])
    after = ha.confirm_host_keys(engine, row["id"], [hk.host_key_fingerprint(HOSTKEY2)])
    assert after["known_host_keys"] == [HOSTKEY2]
