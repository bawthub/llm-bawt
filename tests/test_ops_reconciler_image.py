"""The deployed reconciler must not mix Git source and a baked migration graph."""

import runpy
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
verify = runpy.run_path(str(ROOT / "docker/ops-reconciler-entrypoint.py"))["verify"]
PIN = "LLM_BAWT_OPS_RECONCILER_SOURCE_SHA"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def package(tmp_path):
    repo, app = tmp_path / "repo", tmp_path / "app"
    repo.mkdir()
    app.mkdir()
    git(repo, "init", "-q")
    content = {
        "alembic.ini": "[alembic]\nscript_location = %(here)s/migrations\n",
        "migrations/env.py": "# Migration environment\n",
        "migrations/versions/0002.py": "revision = '0002'\n",
        "migrations/versions/0005.py": "revision = '0005'\n",
        "src/llm_bawt/ops/reconciler.py": "# Reconciler\n",
    }
    for name, data in content.items():
        for root in (repo, app):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
    git(repo, "add", ".")
    git(repo, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-qm", "first")
    sha = git(repo, "rev-parse", "HEAD")
    (app / ".image-source-sha").write_text(sha + "\n")
    return repo, app, sha


def test_baked_commit_stays_usable_after_the_git_input_moves(package):
    repo, app, sha = package
    (repo / "migrations/versions/0006.py").write_text("revision = '0006'\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-qm", "next")
    cache = app / "migrations/versions/__pycache__"
    cache.mkdir()
    (cache / "0005.cpython-312.pyc").write_bytes(b"generated")
    verify(app, repo, sha)
    with pytest.raises(ValueError, match="does not match pin"):
        verify(app, repo, git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize(("name", "contents", "error"), [
    ("migrations/versions/0005.py", None, "missing.*0005"),
    ("migrations/versions/0005.py", "changed", "content differs.*0005"),
    ("migrations/versions/0006.py", "extra", "unexpected file.*0006"),
    ("alembic.ini", "old config", "content differs.*alembic.ini"),
    ("src/llm_bawt/ops/reconciler.py", "wrong source", "content differs.*reconciler.py"),
])
def test_rejects_mislabeled_image_or_shadowed_files(package, name, contents, error):
    repo, app, sha = package
    if contents is None:
        (app / name).unlink()
    else:
        (app / name).write_text(contents)
    with pytest.raises(ValueError, match=error):
        verify(app, repo, sha)


def test_guard_failure_does_not_start_reconciler(package, monkeypatch):
    repo, _, _ = package
    monkeypatch.setenv(PIN, "unknown")
    monkeypatch.setenv("LLM_BAWT_OPS_GATE_REPOSITORY", str(repo))
    with pytest.raises(SystemExit, match="image/source mismatch.*full commit SHA"):
        runpy.run_path(str(ROOT / "docker/ops-reconciler-entrypoint.py"), run_name="__main__")


def test_compose_runs_guard_without_runtime_source_mounts():
    service = yaml.safe_load((ROOT / "docker-compose.ops.yml").read_text())["services"]["ops-reconciler"]
    dockerfile = (ROOT / "Dockerfile.app-cpu").read_text()
    assert service["entrypoint"] == ["python", "-I", "/usr/local/libexec/ops-reconciler-entrypoint.py"]
    assert PIN in service["environment"]
    assert all(not volume.split(":")[1].startswith("/app") for volume in service["volumes"])
    assert ".:/srv/llm-bawt-repo:ro" in service["volumes"]
    assert "COPY docker/ops-reconciler-entrypoint.py /usr/local/libexec/ops-reconciler-entrypoint.py" in dockerfile
