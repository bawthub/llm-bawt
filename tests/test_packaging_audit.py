"""TASK-2270: shell/recipe checks with no installers, VM, Docker or network."""
import json
import os
from pathlib import Path
import shlex
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


@pytest.mark.parametrize("args,service", [
    (["logs"], None),
    (["logs", "fixture-tenant"], None),
    (["logs", "fixture-tenant", "app"], "app"),
    (["logs", "fixture-tenant", "bridge"], "bridge"),
    (["logs", "fixture-tenant", "a; touch unexpected"], "a; touch unexpected"),
])
def test_tenant_logs_forwards_optional_service_without_changing_tenant(tmp_path, args, service):
    log = tmp_path / "ssh.json"
    # No sudo, SSH or VM command escapes these stubs.
    stub(tmp_path, "sudo", "import sys\nassert sys.argv[1:] == ['virsh', 'net-dhcp-leases', 'default']\nprint('192.0.2.1')\n")
    stub(tmp_path, "ssh", "import json, os, sys\nwith open(os.environ['COMMAND_LOG'], 'w') as f:\n    json.dump(sys.argv[1:], f)\n")
    subprocess.run(["bash", str(REPO / "tenant.sh"), *args], cwd=tmp_path,
                   check=True, capture_output=True, text=True,
                   env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "COMMAND_LOG": str(log)})
    ssh_args = json.loads(log.read_text())
    assert ssh_args[-2] == "bawthub@192.0.2.1"
    expected = ["docker", "compose", "-f", "docker-compose.prod.yml", "logs", "-f", "--tail=100"]
    assert shlex.split(ssh_args[-1]) == expected + ([service] if service else [])
    assert not (tmp_path / "unexpected").exists()
