"""Focused SDK probes may not silently select ordinary or external tests."""

from contextlib import nullcontext
import os

import pytest

from scripts import test_suite


def test_sdk_probe_rejects_mixed_or_non_probe_nodes(monkeypatch, capsys):
    requested = {"tests/test_sdk.py::test_probe", "tests/test_external.py::test_paid"}
    selected = {"tests/test_sdk.py::test_probe"}
    runs = []
    monkeypatch.setattr(test_suite, "runner_lock", lambda *_: nullcontext())
    monkeypatch.setattr(test_suite, "collect", lambda marker, *_args, **_kwargs:
                        selected if marker == test_suite.SDK else requested)
    monkeypatch.setattr(test_suite, "run", lambda *args: runs.append(args))

    assert test_suite.main(["sdk-probe", "tests/test_sdk.py", "tests/test_external.py"]) == 1
    assert "tests/test_external.py::test_paid" in capsys.readouterr().err
    assert runs == []


def test_sdk_probe_accepts_only_selected_probe_node_ids(monkeypatch):
    nodes = {"tests/test_sdk.py::test_probe"}
    runs = []
    previous = os.environ.get("RUN_MCP_RECONNECT_SDK_PROBE")
    monkeypatch.setattr(test_suite, "runner_lock", lambda *_: nullcontext())
    monkeypatch.setattr(test_suite, "collect", lambda *_args, **_kwargs: nodes)

    def record_run(*args):
        assert os.environ["RUN_MCP_RECONNECT_SDK_PROBE"] == "1"
        runs.append(args)

    monkeypatch.setattr(test_suite, "run", record_run)
    assert test_suite.main(["sdk-probe", "tests/test_sdk.py::test_probe"]) == 0
    assert runs == [(test_suite.SDK, "--require-no-skips", "tests/test_sdk.py::test_probe")]
    assert os.environ.get("RUN_MCP_RECONNECT_SDK_PROBE") == previous


def test_sdk_probe_rejects_unsafe_node_path_before_collection(monkeypatch):
    monkeypatch.setattr(test_suite, "runner_lock", lambda *_: pytest.fail("lock acquired"))
    with pytest.raises(SystemExit, match="2"):
        test_suite.main(["sdk-probe", "tests/../test_sdk.py"])
