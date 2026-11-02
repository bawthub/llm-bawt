"""Fixed launcher contract/lifecycle with fake Docker; never launch a container."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from llm_bawt.ops import gate_runner as runner
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.gate_source import GateSource

JOB = "1" * 32
RUNNER_ID = "sha256:" + "2" * 64
PG_ID = "sha256:" + "3" * 64
RUNNER_IMAGE = "fixture/runner@sha256:" + "4" * 64


def evidence(source):
    return {"schema": "llm-bawt.ops.gate-environment/v1", "repository": "llm-bawt",
        "source_sha": source.source_sha, "tree_sha": source.tree_sha, "workload": "test-all/offline",
        "platform": "linux/amd64", "runner_image": RUNNER_IMAGE, "runner_image_id": RUNNER_ID,
        "postgres_image": runner._PG_IMAGE, "postgres_image_id": PG_ID,
        "toolchain_sha256": "5" * 64, "dependency_closure_sha256": "6" * 64,
        "asset_closure_sha256": "7" * 64}


class Stream:
    def __init__(self, api):
        self.api = api

    def read(self, limit):
        assert limit == runner._MAX_OUTPUT + 1
        if self.api.fault == "read":
            raise OSError("fixture sensitive transport detail")
        if self.api.fault == "deadline":
            raise runner.GateLaunchError("deadline_exceeded")
        if self.api.fault == "interrupt":
            raise KeyboardInterrupt()
        return self.api.output

    def close(self):
        self.api.closed = True


class Docker:
    def __init__(self):
        self.calls, self.containers, self.bodies, self.deleted = [], {}, [], []
        self.output, self.exit_code, self.fault, self.closed = b"fixture untrusted output", 0, None, False
        self.binding = None
        self.images = {RUNNER_IMAGE: {"Id": RUNNER_ID, "RepoDigests": [RUNNER_IMAGE],
            "Os": "linux", "Architecture": "amd64", "Config": {"Env": ["PATH=/fixture", "HOME=/root"]}},
            runner._PG_IMAGE: {"Id": PG_ID, "RepoDigests": [runner._PG_IMAGE],
                "Os": "linux", "Architecture": "amd64",
                "Config": {"Env": ["PGDATA=/var/lib/postgresql/data"],
                    "Volumes": {"/var/lib/postgresql/data": {}}}}}

    def attach_stream(self, identity):
        self.calls.append(("ATTACH", identity, None))
        assert self.containers[identity]["State"]["Running"] is False
        return Stream(self)

    def request(self, method, path, *, body=None, raw=False):
        self.calls.append((method, path, copy.deepcopy(body)))
        url = urlsplit(unquote(path))
        parts = url.path.strip("/").split("/")
        if parts[0] == "images":
            return copy.deepcopy(self.images["/".join(parts[1:-1])])
        if method == "POST" and url.path == "/containers/create":
            name = parse_qs(url.query)["name"][0]
            self.bodies.append(copy.deepcopy(body))
            identity = ("a" if name.endswith("-pg") else "b") * 64
            state = {"Id": identity, "Name": name, "Config": {"Labels": copy.deepcopy(body["Labels"])},
                "State": {"Running": False, "Status": "created", "ExitCode": 0, "OOMKilled": False, "Error": ""}}
            self.containers[identity] = state
            if name.endswith("-runner"):
                mount = next(m for m in body["HostConfig"]["Mounts"] if m["Target"] == "/gate/pg.json")
                self.binding = json.loads(Path(mount["Source"]).read_text())
            if self.fault == "foreign_create":
                state["Config"]["Labels"]["llm-bawt.ops.gate.owner"] = "another-invocation"
                raise RuntimeError("Docker HTTP 409: sensitive fixture detail")
            if self.fault in ("create_lost", "create_misleading_404"):
                if self.fault == "create_lost":
                    raise OSError("fixture lost create response")
                raise RuntimeError("Docker HTTP 500: upstream said HTTP 404 in sensitive body")
            return {"Id": identity}
        if parts[0] == "exec":
            if parts[-1] == "start":
                assert raw
                return b""
            if self.fault == "readiness":
                return {"Running": True, "ExitCode": None}
            return {"Running": False, "ExitCode": 0}
        identity = parts[1]
        state = self.containers.get(identity) or next(
            (s for s in self.containers.values() if s["Name"] == identity), None)
        if state is None:
            raise RuntimeError("Docker HTTP 404: fixture missing")
        if method == "GET":
            return copy.deepcopy(state)
        if method == "DELETE":
            self.deleted.append(state["Id"])
            if self.fault == "remove":
                raise OSError("fixture removal uncertain")
            if self.fault != "remove_noop":
                del self.containers[state["Id"]]
            return None
        if parts[-1] == "exec":
            return {"Id": "c" * 64}
        assert parts[-1] == "start"
        state["State"]["Running"] = True
        if self.fault == "start_lost":
            raise OSError("fixture start response lost")
        if state["Name"].endswith("-runner"):
            state["State"].update(Running=False, Status="exited", ExitCode=self.exit_code)
            if self.fault == "not_started":
                state["State"]["Status"] = "created"
            if self.fault == "still_running":
                state["State"]["Running"] = True
            if self.fault == "oom":
                state["State"]["OOMKilled"] = True
            if self.fault == "exit_bool":
                state["State"]["ExitCode"] = True
        return None


@pytest.fixture
def setup(tmp_path):
    stage_parent = tmp_path / "stage"
    stage_parent.mkdir(mode=0o700)
    root = stage_parent / "source"
    root.mkdir(mode=0o700)
    (root / ".git").mkdir(mode=0o700)
    (root / ".git/HEAD").write_text("8" * 40 + "\n")
    (root / ".git/HEAD").chmod(0o600)
    (root / "tracked").write_bytes(b"fixture source")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source = GateSource(root, "8" * 40, "9" * 40, 1, 14)
    data = json.dumps(evidence(source), sort_keys=True).encode()
    args = {"plan": GatePlan("llm-bawt", source.source_sha), "source": source, "job_id": JOB,
        "scratch_parent": scratch, "environment_json": data,
        "environment_sha256": hashlib.sha256(data).hexdigest()}
    return Docker(), args


def encoded(args, data):
    raw = json.dumps(data, sort_keys=True).encode()
    return {**args, "environment_json": raw, "environment_sha256": hashlib.sha256(raw).hexdigest()}


def test_fixed_hardened_workload_and_positive_cleanup(setup, monkeypatch):
    api, args = setup
    monkeypatch.setenv("LD_PRELOAD", "/fixture/host-secret")
    monkeypatch.setenv("DOCKER_HOST", "tcp://fixture.invalid")
    before = args["source"].directory.joinpath("tracked").read_bytes()
    result = runner.run_gate(api, **args)
    assert result.exit_code == 0 and result.output == api.output and result.cleanup_confirmed
    assert api.closed and not api.containers and len(api.deleted) == 2
    assert list(args["scratch_parent"].iterdir()) == []
    assert args["source"].directory.joinpath("tracked").read_bytes() == before
    pg, run = api.bodies
    for body in (pg, run):
        host = body["HostConfig"]
        assert host["NetworkMode"] == "none" and body["NetworkDisabled"]
        assert host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"]
        assert host["SecurityOpt"] == ["no-new-privileges:true"]
        assert host["LogConfig"]["Type"] == "none" and host["AutoRemove"] is False
        assert host["Privileged"] is False
        assert host["NanoCpus"] == 2_000_000_000 and not body["OpenStdin"]
        assert not ({"Binds", "VolumesFrom", "Devices", "PortBindings", "GroupAdd"} & host.keys())
        assert body["Entrypoint"][:2] == ["/usr/bin/env", "-i"]
        assert body["Healthcheck"] == {"Test": ["NONE"]}
        assert "host-secret" not in json.dumps(body) and "DOCKER_HOST" not in json.dumps(body)
        assert all("docker.sock" not in m["Source"] for m in host["Mounts"])
        assert body["Labels"]["llm-bawt.ops.job"] == JOB
    assert pg["User"] == "999:999" and run["User"] == "65532:65532"
    assert pg["HostConfig"]["PidsLimit"] == 256 and run["HostConfig"]["PidsLimit"] == 2048
    # Docker defaults tmpfs to noexec; fixture programs need explicit exec.
    assert all(o.split(",")[:2] == ["rw", "exec"] for o in run["HostConfig"]["Tmpfs"].values())
    assert "TMPDIR=/tmp" in run["Entrypoint"] and "/opt/gate/venv/bin/python" in run["Cmd"][-1]
    assert pg["Image"] == PG_ID and run["Image"] == RUNNER_ID
    assert len(pg["HostConfig"]["Mounts"]) == 1 and len(run["HostConfig"]["Mounts"]) == 3
    assert all(m["ReadOnly"] for m in run["HostConfig"]["Mounts"])
    assert not pg["HostConfig"]["Mounts"][0]["ReadOnly"]
    assert "listen_addresses=" in pg["Cmd"] and "port=55439" in pg["Cmd"]
    assert "uv run" not in run["Cmd"][-1] and "uv-cache" not in run["Cmd"][-1]
    assert run["Cmd"][-1].endswith("python -m scripts.test_suite offline\n")
    assert "OPS_GATE_SANDBOX=1" in run["Entrypoint"]
    assert "OPS_GATE_PG_BINDING=/gate/pg.json" in run["Entrypoint"]
    assert api.binding == {"schema": "ops.gate.pg/v1", "job_id": JOB, "repo": "llm-bawt",
        "sha": args["plan"].source_sha, "mode": "unix", "jobs": {"private-pg": {
            "image": runner._PG_IMAGE, "socket_dir": "/var/run/postgresql", "port": "55439",
            "user": "test_user", "database": "llm_bawt_test"}}}
    attach = next(i for i, c in enumerate(api.calls) if c[0] == "ATTACH")
    start = next(i for i, c in enumerate(api.calls) if c[:2] == ("POST", "/containers/" + "b" * 64 + "/start"))
    assert attach < start
    assert not any("images/create" in c[1] or "/wait" in c[1] for c in api.calls)


def test_red_workload_is_not_translated_to_pass(setup):
    api, args = setup
    api.exit_code = 23
    result = runner.run_gate(api, **args)
    assert result.exit_code == 23 and result.cleanup_confirmed


@pytest.mark.parametrize("field,value", [
    ("schema", "finite-assets/v1"), ("repository", "bawthub"), ("source_sha", "a" * 40),
    ("tree_sha", "a" * 40), ("platform", "linux/arm64"), ("workload", "shell"),
    ("runner_image", "fixture:latest"), ("runner_image_id", "sha256:bad"),
    ("postgres_image", "fixture:latest"), ("postgres_image_id", "3" * 64),
    ("toolchain_sha256", ""), ("dependency_closure_sha256", False), ("asset_closure_sha256", None)])
def test_incompatible_environment_refuses_before_docker(setup, field, value):
    api, args = setup
    data = evidence(args["source"])
    data[field] = value
    with pytest.raises(runner.GateLaunchError, match="environment_evidence_incompatible"):
        runner.run_gate(api, **encoded(args, data))
    assert not api.calls


@pytest.mark.parametrize("variant", ["missing_pin", "mismatch", "duplicate", "extra", "missing",
    "oversized", "invalid_json", "non_utf8", "not_bytes"])
def test_missing_or_malformed_evidence_refuses_before_docker(setup, variant):
    api, args = setup
    data = evidence(args["source"])
    if variant == "missing_pin":
        args["environment_sha256"] = None
    elif variant == "mismatch":
        args["environment_sha256"] = "a" * 64
    elif variant in ("extra", "missing"):
        if variant == "extra":
            data["command"] = "fixture"
        else:
            del data["dependency_closure_sha256"]
        args = encoded(args, data)
    else:
        raw = {"duplicate": b'{"schema":"one","schema":"two"}', "oversized": b"x" * 8193,
            "invalid_json": b"{", "non_utf8": b"\xff", "not_bytes": "fixture"}[variant]
        args["environment_json"] = raw
        args["environment_sha256"] = hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()
    with pytest.raises(runner.GateLaunchError):
        runner.run_gate(api, **args)
    assert not api.calls


@pytest.mark.parametrize("key,value", [("Id", "sha256:" + "a" * 64), ("Os", "windows"),
    ("Architecture", "arm64"), ("RepoDigests", [])])
def test_local_immutable_image_identity_is_checked(setup, key, value):
    api, args = setup
    api.images[RUNNER_IMAGE][key] = value
    with pytest.raises(runner.GateLaunchError, match="local_image_mismatch"):
        runner.run_gate(api, **args)
    assert not api.bodies


@pytest.mark.parametrize("defaults", [{"Env": ["LD_PRELOAD=/fixture"]}, {"Env": ["BASH_ENV=/fixture"]},
    {"Env": ["PYTHONPATH=/fixture"]}, {"Env": ["HOME=/one", "HOME=/two"]},
    {"Volumes": {"/var/run/docker.sock": {}}}, {"Env": ["PATH=/x\nsecret"]}])
def test_image_defaults_cannot_smuggle_execution_or_mounts(setup, defaults):
    api, args = setup
    api.images[RUNNER_IMAGE]["Config"] = defaults
    with pytest.raises(runner.GateLaunchError, match="local_image_unsafe_defaults"):
        runner.run_gate(api, **args)
    assert not api.bodies


@pytest.mark.parametrize("fault", ["create_lost", "create_misleading_404", "start_lost", "readiness",
    "read", "deadline", "still_running", "not_started", "oom", "exit_bool"])
def test_failed_or_uncertain_execution_is_cleaned_but_never_success(setup, fault):
    api, args = setup
    api.fault = fault
    with pytest.raises(runner.GateLaunchError) as error:
        runner.run_gate(api, **args)
    assert error.value.cleanup_confirmed and not api.containers
    assert "fixture" not in str(error.value)
    assert list(args["scratch_parent"].iterdir()) == []
    assert len(api.bodies) <= 2  # No retry path.


@pytest.mark.parametrize("fault", ["remove", "remove_noop", "foreign_create"])
def test_uncertain_cleanup_retains_scratch_and_does_not_claim_retirement(setup, fault):
    api, args = setup
    api.fault = fault
    with pytest.raises(runner.GateLaunchError, match="cleanup_uncertain") as error:
        runner.run_gate(api, **args)
    assert not error.value.cleanup_confirmed and api.containers
    assert (args["scratch_parent"] / f"ops-gate-{JOB}").is_dir()
    if fault == "foreign_create":
        assert not api.deleted  # Same job/SHA but another custody nonce; never remove it.


def test_preexisting_scratch_is_never_adopted_or_deleted(setup):
    api, args = setup
    scratch = args["scratch_parent"] / f"ops-gate-{JOB}"
    scratch.mkdir(mode=0o700)
    sentinel = scratch / "other-owner"
    sentinel.write_text("preserve")
    with pytest.raises(runner.GateLaunchError, match="launcher_unavailable"):
        runner.run_gate(api, **args)
    assert sentinel.read_text() == "preserve" and not api.bodies


def test_occupied_deterministic_container_is_never_adopted_or_removed(setup):
    api, args = setup
    api.containers["d" * 64] = {"Id": "d" * 64,
        "Name": f"llm-bawt-ops-gate-{JOB}-pg", "Config": {"Labels": {}}, "State": {"Running": True}}
    with pytest.raises(runner.GateLaunchError, match="container_name_occupied"):
        runner.run_gate(api, **args)
    assert not api.bodies and not api.deleted and len(api.containers) == 1


@pytest.mark.parametrize("change", ["symlink", "shared_parent", "shared_scratch", "nested_scratch"])
def test_unsafe_host_handoff_refuses_without_create(setup, change, tmp_path):
    api, args = setup
    root = args["source"].directory
    if change == "symlink":
        (root / "link").symlink_to(tmp_path)
    elif change == "shared_parent":
        root.parent.chmod(0o755)
    elif change == "nested_scratch":
        path = root / "scratch"
        path.mkdir(mode=0o700)
        args["scratch_parent"] = path
    else:
        args["scratch_parent"].chmod(0o777)
    with pytest.raises(runner.GateLaunchError):
        runner.run_gate(api, **args)
    assert not api.bodies


def test_existing_deadline_is_not_replaced(setup):
    api, args = setup
    old = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 100)
    try:
        with pytest.raises(runner.GateLaunchError, match="deadline_unavailable"):
            runner.run_gate(api, **args)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 90
        assert signal.getsignal(signal.SIGALRM) == old
        assert not api.calls
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_output_overflow_refuses_without_truncation(setup):
    api, args = setup
    api.output = b"x" * (runner._MAX_OUTPUT + 1)
    with pytest.raises(runner.GateLaunchError, match="output_bound_exceeded") as error:
        runner.run_gate(api, **args)
    assert error.value.cleanup_confirmed and api.closed and not api.containers


def test_keyboard_interrupt_is_preserved_after_cleanup(setup):
    api, args = setup
    api.fault = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        runner.run_gate(api, **args)
    assert api.closed and not api.containers


def test_persisted_internal_custody_nonce_binds_both_roles(setup):
    api, args = setup
    nonce = "d" * 32
    result = runner.run_gate(api, **args, owner_nonce=nonce)
    assert result.cleanup_confirmed and len(api.bodies) == 2
    for body in api.bodies:
        assert body["Labels"]["llm-bawt.ops.gate.owner"] == nonce
    assert not api.containers


@pytest.mark.parametrize("nonce", ["", "d" * 31, "d" * 33, "D" * 32, "z" * 32, False, 123, []])
def test_invalid_internal_custody_nonce_refuses_before_docker(setup, nonce):
    api, args = setup
    with pytest.raises(runner.GateLaunchError, match="custody_identity_invalid"):
        runner.run_gate(api, **args, owner_nonce=nonce)
    assert not api.calls and not list(args["scratch_parent"].iterdir())
