"""Offline tests: no downloads, live settings, or external services."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/rea"
spec = importlib.util.spec_from_file_location("rea_install", SCRIPTS / "install.py")
assert spec and spec.loader
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    tool = installer.Installer(tmp_path / "home with spaces/tools/rea")
    calls = []

    def run(*args, cwd=None, timeout=120):
        calls.append(args)
        if args[:2] == ("npm", "ci"):
            package = cwd / "node_modules/rea-agents"
            (package / "scripts").mkdir(parents=True)
            (package / "package.json").write_text(json.dumps({"version": installer.VERSION}))
            (package / "scripts/rea.mjs").write_text(
                "console.log(JSON.stringify(process.argv.slice(2)));"
            )
        if args[-1] == "--help":
            return "REA help"
        return "v24.21.0" if args[0] == "node" else "11.19.0"

    monkeypatch.setattr(tool, "run", run)
    return tool, calls


def test_install_is_atomic_idempotent_and_pinned(fake_install):
    tool, calls = fake_install
    release = tool.install()
    assert (tool.root / "current").resolve() == release
    receipt = json.loads((release / "receipt.json").read_text())
    assert receipt["integrity"] == installer.INTEGRITY
    assert receipt["lifecycle_scripts"] == "disabled"
    mtime = (tool.root / "current").lstat().st_mtime_ns
    assert tool.install() == release
    assert (tool.root / "current").lstat().st_mtime_ns == mtime
    installs = [args for args in calls if args[:2] == ("npm", "ci")]
    assert len(installs) == 1
    assert "--ignore-scripts" in installs[0]
    assert not list((tool.root / "releases").glob(".stage-*"))


@pytest.mark.parametrize("failure", ["install", "smoke"])
def test_failure_preserves_old_selection_and_cleans_stage(fake_install, monkeypatch, failure):
    tool, _ = fake_install
    old = tool.root / "releases/old"
    old.mkdir(parents=True)
    (old / "sentinel").write_text("keep")
    (tool.root / "current").symlink_to("releases/old")
    run = tool.run

    def failing(*args, **kwargs):
        if (failure == "install" and args[:2] == ("npm", "ci")) or (failure == "smoke" and args[-1] == "--help"):
            raise RuntimeError("simulated failure")
        return run(*args, **kwargs)

    monkeypatch.setattr(tool, "run", failing)
    with pytest.raises(RuntimeError, match="simulated"):
        tool.install()
    assert (tool.root / "current").resolve() == old
    assert (old / "sentinel").read_text() == "keep"
    assert list((tool.root / "releases").iterdir()) == [old]


def test_lock_contention_refuses_mutation(fake_install):
    tool, calls = fake_install
    tool.root.mkdir(parents=True)
    with (tool.root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another REA"):
            tool.install()
    assert not any(args[:2] == ("npm", "ci") for args in calls)


@pytest.mark.parametrize("error", [FileNotFoundError("Node missing"), RuntimeError("unsupported Node")])
def test_bad_runtime_leaves_no_state(fake_install, monkeypatch, error):
    tool, _ = fake_install

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tool, "run", fail)
    with pytest.raises(type(error)):
        tool.install()
    assert not tool.root.exists()


def test_existing_release_not_rewritten(fake_install):
    tool, _ = fake_install
    release = tool.install()
    (release / "package.json").write_text("{}")
    with pytest.raises(RuntimeError, match="inputs changed"):
        tool.install()
    assert (release / "package.json").read_text() == "{}"


def test_non_symlink_current_not_overwritten(fake_install):
    tool, _ = fake_install
    tool.root.mkdir(parents=True)
    (tool.root / "current").write_text("unrelated")
    with pytest.raises(RuntimeError, match="not a symlink"):
        tool.install()
    assert (tool.root / "current").read_text() == "unrelated"


def test_launcher_absent_install_is_protocol_clean(tmp_path):
    if not shutil.which("node"):
        pytest.skip("launcher checks need Node")
    result = subprocess.run(["bash", str(SCRIPTS / "rea"), "mcp"],
                            env={**os.environ, "REA_INSTALL_ROOT": str(tmp_path)},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "no installed release" in result.stderr


def test_launcher_preserves_arguments_and_exit_status(fake_install):
    if not shutil.which("node"):
        pytest.skip("launcher checks need Node")
    tool, _ = fake_install
    release = tool.install()
    entry = release / "node_modules/rea-agents/scripts/rea.mjs"
    entry.write_text("console.log(JSON.stringify(process.argv.slice(2))); process.exit(17);")
    result = subprocess.run(["bash", str(SCRIPTS / "rea"), "mcp", "space path", "--flag"],
                            env={**os.environ, "REA_INSTALL_ROOT": str(tool.root)},
                            capture_output=True, text=True)
    assert result.returncode == 17
    assert json.loads(result.stdout) == ["mcp", "space path", "--flag"]
    assert result.stderr == ""


def test_old_release_can_be_reselected_without_mutation(fake_install):
    tool, _ = fake_install
    release = tool.install()
    old = tool.root / "releases/previous"
    old.mkdir()
    (old / "marker").write_text("preserved")
    pointer = tool.root / ".rollback"
    pointer.symlink_to("releases/previous")
    os.replace(pointer, tool.root / "current")
    assert (tool.root / "current").resolve() == old
    assert release.exists()
    assert tool.install() == release
    assert (old / "marker").read_text() == "preserved"


@pytest.mark.parametrize("version,accepted", [
    ("20.19.2", False), ("22.18.0", False), ("22.19.0", True),
    ("23.0.0", False), ("24.10.9", False), ("24.11.0", True),
    ("24.21.0", True), ("25.0.0", False), ("26.0.0", True),
])
def test_engine_boundaries(version, accepted):
    if not shutil.which("node"):
        pytest.skip("engine checks need Node")
    script = """
const vm = require('node:vm');
const fs = require('node:fs');
let accepted = true;
try {
  vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    process: {versions: {node: process.argv[2]}, stderr: {write() {}},
              exit() {throw new Error('rejected');}}
  });
} catch {accepted = false;}
console.log(JSON.stringify(accepted));
"""
    result = subprocess.run(["node", "-e", script, str(SCRIPTS / "check-node.mjs"), version],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) is accepted
