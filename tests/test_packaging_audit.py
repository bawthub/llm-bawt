"""TASK-2270: shell/recipe checks with no installers, VM, Docker or network."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def stub(directory, name, body):
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("deps_only", [False, True])
def test_install_dev_sync_uses_lock(tmp_path, deps_only):
    log = tmp_path / "commands.jsonl"
    stub(tmp_path, "uv", "import json, os, sys\nwith open(os.environ['COMMAND_LOG'], 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    args = ["bash", str(REPO / "install.sh"), "--dev", "--no-cuda"]
    if deps_only:
        args.append("--deps-only")
    subprocess.run(args, cwd=tmp_path, check=True, capture_output=True, text=True,
                   env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "COMMAND_LOG": str(log)})
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    syncs = [call for call in calls if call[0] == "sync"]
    assert len(syncs) == 1
    assert {"--frozen", "--inexact"} <= set(syncs[0])
    assert ("--no-install-project" in syncs[0]) is deps_only


def test_gpu_docker_recipe_copies_lock_before_both_syncs():
    recipe = (REPO / "Dockerfile").read_text()
    copy = recipe.index("COPY pyproject.toml uv.lock install.sh ./")
    dependency_sync = recipe.index("RUN ./install.sh --dev --deps-only")
    project_sync = recipe.index("RUN uv sync --frozen --inexact")
    assert copy < dependency_sync < project_sync
    assert (REPO / "uv.lock").is_file()
