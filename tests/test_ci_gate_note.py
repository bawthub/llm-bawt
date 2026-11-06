"""TASK-2075: master pushes skip the offline suite only for a SHA the gate
already passed, and tenant images publish only on demand."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def test_the_offline_suite_is_skipped_only_behind_a_passed_gate_note():
    jobs = _load("tests.yml")["jobs"]
    note, offline = jobs["gate-note"], jobs["offline"]
    assert note["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/master'"
    assert note["permissions"] == {"contents": "read"}
    assert offline["needs"] == "gate-note"
    # A skipped note job (pull requests, other branches) or a failed one runs the suite.
    assert offline["if"] == "${{ !cancelled() && needs.gate-note.outputs.covered != 'true' }}"


def test_the_note_check_reads_only_a_passed_note(tmp_path):
    script = _load("tests.yml")["jobs"]["gate-note"]["steps"][1]["run"]
    origin, clone = tmp_path / "origin", tmp_path / "clone"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q", str(origin)], check=True)
    subprocess.run([*git, "-C", str(origin), "commit", "-q", "--allow-empty", "-m", "a"], check=True)
    sha = subprocess.run(["git", "-C", str(origin), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)

    def covered() -> str:
        out = tmp_path / "out"
        out.write_text("")
        env = {**os.environ, "SHA": sha, "GITHUB_OUTPUT": str(out)}
        subprocess.run(["bash", "-c", script], cwd=clone, env=env, check=True, capture_output=True)
        return out.read_text().strip()

    assert covered() == "covered=false"
    notes = [*git, "-C", str(origin), "notes", "--ref=bawthub-gate", "add", "-f", "-m"]
    subprocess.run([*notes, "bawthub-gate FAILED g1", sha], check=True)
    assert covered() == "covered=false"
    subprocess.run([*notes, "bawthub-gate PASSED g2", sha], check=True)
    assert covered() == "covered=true"


def test_tenant_images_publish_only_for_a_dispatched_sha_on_master():
    source = (WORKFLOWS / "publish-tenant-app.yml").read_text()
    workflow = yaml.safe_load(source)
    assert set(workflow[True]) == {"pull_request", "workflow_dispatch"}  # PyYAML reads `on:` as True
    assert workflow[True]["workflow_dispatch"]["inputs"]["sha"]["required"] is True
    assert "workflow_run" not in source
    dispatch = "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/master'"
    jobs = workflow["jobs"]
    assert " ".join(jobs["build"]["if"].split()) == f"github.event_name == 'pull_request' || ({dispatch})"
    assert jobs["publish"]["if"] == dispatch
    check = next(s for s in jobs["build"]["steps"] if s.get("name") == "The SHA is a commit on master")
    assert check["env"] == {"SHA": "${{ inputs.sha }}"}
    assert 'git merge-base --is-ancestor "$SHA" origin/master' in check["run"]
