"""TASK-2403: a bridge restart/reconcile imports the new source before stopping the bridge."""
from __future__ import annotations

import json
import struct

import pytest

from llm_bawt.ops import import_preflight
from llm_bawt.ops.executor import spec_capability_labels, validate_spec
from llm_bawt.ops.import_preflight import ImportPreflightError
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import DockerAPI

CONFIG = {"source_mount": "/app/src", "modules": ["claude_code_bridge.__main__"]}
SPEC = {"action": "restart", "compose_project": "llm-bawt", "compose_service": "claude-code-bridge",
        "import_preflight": CONFIG}


def _frame(stream: int, body: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + struct.pack(">I", len(body)) + body


class _Stream:
    def __init__(self, raw):
        self.raw = raw

    def read(self, n):
        return self.raw[:n]

    def close(self):
        pass


class FakeDocker(DockerAPI):
    """A bridge with /app/src bound; the sealed preflight exits ``status``."""

    def __init__(self, status, output):
        super().__init__(timeout=60, clock=lambda: 0.0, sleep=lambda _s: None)
        self.status, self.output, self.calls, self.created = status, output, [], []

    def request(self, method, path, *, body=None, headers=None, raw=False):
        self.calls.append((method, path))
        if path.startswith("/containers/json?"):
            return [{"Id": "br", "Labels": {}}]
        if method == "GET" and path == "/containers/br/json":
            return {"Id": "br", "Name": "/llm-bawt-claude-code-bridge", "Image": "sha256:img",
                    "Config": {"Image": "bridge"}, "State": {"Running": True, "Status": "running"},
                    "Mounts": [{"Type": "bind", "Destination": "/app/src", "Source": "/srv/src"}]}
        if path.startswith("/images/"):
            return {"Id": "sha256:img"}
        if path.startswith("/containers/create"):
            self.created.append(body)
            return {"Id": "pf"}
        if path == "/containers/pf/wait":
            return {"StatusCode": self.status}
        return None

    def attach_stream(self, container_id, **_kw):
        return _Stream(self.output)


def test_restart_refuses_before_touching_the_bridge_when_an_import_fails():
    api = FakeDocker(1, _frame(2, b"Traceback ...\nModuleNotFoundError: No module named 'rich'\n"))
    with pytest.raises(ImportPreflightError, match="(?s)import_preflight_failed.*No module named 'rich'"):
        api.execute(SPEC, {})
    assert not any(path.startswith("/containers/br/restart") for _m, path in api.calls)
    assert api.side_effect_started is False
    assert ("DELETE", "/containers/pf?force=1&v=1") in api.calls  # throwaway removed


def test_restart_proceeds_after_a_clean_preflight_and_reports_it():
    api = FakeDocker(0, _frame(1, b"ok claude_code_bridge.__main__ (12 imports)\n"))
    out = api.execute(SPEC, {})
    assert "import preflight ok: ok claude_code_bridge.__main__ (12 imports)" in out
    assert ("POST", "/containers/br/restart?t=10") in api.calls
    body = api.created[0]
    # Same image, source read-only at its host path and first on sys.path, no network.
    assert body["Image"] == "sha256:img" and body["HostConfig"]["NetworkMode"] == "none"
    assert body["HostConfig"]["Binds"] == ["/srv/src:/srv/src:ro"]
    assert "PYTHONPATH=/srv/src" in body["Env"] and body["Cmd"] == CONFIG["modules"]


def test_reconcile_fails_known_before_the_swap_on_a_bad_import(monkeypatch):
    from llm_bawt.ops.app_config_reconcile import AppConfigReconciler

    seen = {}

    def fake_run(api, config, *, image, source_dir, tag):
        seen.update(image=image, source_dir=source_dir)
        raise ImportPreflightError("import_preflight_failed (exit 1):\nModuleNotFoundError: rich")

    class Failed(Exception):
        pass

    def fail(message, known=False):
        seen.update(message=message, known=known)
        raise Failed

    monkeypatch.setattr(import_preflight, "run", fake_run)
    rec = object.__new__(AppConfigReconciler)
    rec.spec, rec.job_id, rec.api, rec.detail = {"import_preflight": CONFIG}, "job1", None, {}
    rec._desired = {"mounts": {"/app/src": {"source": "/srv/src", "read_only": False}}}
    rec._approved_image_id, rec._phase, rec._fail = "sha256:img", lambda _p: None, fail
    with pytest.raises(Failed):
        rec._import_preflight()
    assert seen["image"] == "sha256:img" and seen["source_dir"] == "/srv/src"
    assert seen["known"] is True and "rich" in seen["message"] and "nothing changed" in seen["message"]


def test_missing_source_mount_refuses():
    with pytest.raises(ImportPreflightError, match="no bind source"):
        import_preflight.run(None, CONFIG, image="x", source_dir=None, tag="t")


@pytest.mark.parametrize("bad", [
    {}, {"source_mount": "/app/src"}, {"source_mount": "rel", "modules": ["a"]},
    {"source_mount": "/app/src", "modules": []}, {"source_mount": "/app/src", "modules": ["a; rm"]},
    {**CONFIG, "extra": 1},
])
def test_config_shape_is_strict(bad):
    with pytest.raises(ValueError):
        import_preflight.validate(bad)


def test_restart_spec_accepts_preflight_only_on_restart_and_needs_a_capable_worker():
    validate_spec(json.dumps(SPEC))
    with pytest.raises(ValueError, match="restart only"):
        validate_spec(json.dumps({**SPEC, "action": "stop"}))
    assert import_preflight.WORKER_CAPABILITY_LABEL in spec_capability_labels(SPEC)


def test_bridge_seeds_carry_the_preflight():
    rows = {row["slug"]: row["command_script"] for row in SEEDS}
    modules = {"claude_code_bridge.__main__", "codex_bridge.__main__", "agent_bridge.house_preview_supervisor"}
    for slug in ("llm-bawt.restart-bridge", "llm-bawt.reconcile-claude-code-bridge-config"):
        config = validate_spec(rows[slug])["import_preflight"]
        assert config["source_mount"] == "/app/src" and set(config["modules"]) == modules
