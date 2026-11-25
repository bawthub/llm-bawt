"""TASK-2270: shell/recipe checks with no installers, VM, Docker or network."""
import configparser
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


def test_make_dev_sync_uses_lock(tmp_path):
    """`make dev` is the supported development install; it must honor uv.lock."""
    log = tmp_path / "commands.jsonl"
    stub(tmp_path, "uv", "import json, os, sys\nwith open(os.environ['COMMAND_LOG'], 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    subprocess.run(["make", "-s", "-C", str(REPO), "dev"], check=True, capture_output=True, text=True,
                   env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "COMMAND_LOG": str(log)})
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    syncs = [call for call in calls if call[0] == "sync"]
    assert len(syncs) == 1
    assert {"--frozen", "--inexact"} <= set(syncs[0])


def test_gpu_docker_recipe_copies_lock_before_both_syncs():
    recipe = (REPO / "Dockerfile").read_text()
    copy = recipe.index("COPY pyproject.toml setup.cfg uv.lock ./")
    dependency_sync = recipe.index("RUN uv sync --frozen --inexact --no-install-project")
    project_sync = recipe.index("RUN uv sync --frozen --inexact --extra", dependency_sync)
    assert copy < dependency_sync < project_sync
    assert (REPO / "uv.lock").is_file()


def test_setuptools_metadata_lives_outside_importable_source():
    config = configparser.ConfigParser()
    config.read(REPO / "setup.cfg")
    assert config.get("egg_info", "egg_base") == "."


@pytest.mark.parametrize("dockerfile", [
    "Dockerfile", "Dockerfile.app-cpu", "Dockerfile.agent-bridge",
    "Dockerfile.house", "docker/Dockerfile.gate-runner",
])
def test_images_copy_metadata_policy_before_install(dockerfile):
    recipe = (REPO / dockerfile).read_text()
    metadata_copy = next(
        line for line in recipe.splitlines()
        if line.startswith("COPY pyproject.toml ")
    )
    assert "setup.cfg" in metadata_copy.split()
    install = min(
        recipe.index(marker)
        for marker in ("uv sync", "uv pip install")
        if marker in recipe
    )
    assert recipe.index(metadata_copy) < install
