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


def test_add_and_remove_need_a_pinned_host_key(engine):
    row = _host(engine)
    assert ha.pin(engine, row["id"], FP, "verify")["known_host_keys"] == []
    with pytest.raises(ha.HostAccessError):
        ha.pin(engine, row["id"], FP, "add")
    with pytest.raises(ha.HostAccessError):
        ha.pin(engine, "1", FP, "verify")


def test_first_verify_learns_keys_then_pin_and_check(engine):
    row = _host(engine)
    target = ha.pin(engine, row["id"], FP, "verify")
    result = {"op": "verify", "login_ok": True, "passwordless_sudo_probe": True, "learned_host_keys": [HOSTKEY]}
    ha.settle(engine, _job("succeeded", "host login verified: " + json.dumps(result)),
              {"spec": {"op": "verify"}, "host_key": target})
    after = ha.list_hosts(engine)[0]
    assert after["status"] == "verified" and after["passwordless_sudo"] is True
    assert after["known_host_keys"] == [HOSTKEY]
    # The old approval no longer matches the record (keys were learned).
    with pytest.raises(ha.HostAccessError):
        ha.check_pin(engine, target)
    add = ha.pin(engine, row["id"], FP, "add")
    ha.check_pin(engine, add)


def test_editing_destination_clears_trust_and_invalidates_approval(engine):
    row = _host(engine)
    target = ha.pin(engine, row["id"], FP, "verify")
    ha.settle(engine, _job("succeeded", "host login verified: " + json.dumps({"learned_host_keys": [HOSTKEY]})),
              {"spec": {"op": "verify"}, "host_key": target})
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
        "llm-bawt.host-key-add", "llm-bawt.host-key-remove", "llm-bawt.host-key-verify"]
    fields = set(OpsOperation.__table__.columns.keys())
    for seed in seeds:
        assert seed["enabled"] is False
        OpsStore._validate_op(OpsOperation(id="t", **{k: v for k, v in seed.items() if k in fields}))


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
