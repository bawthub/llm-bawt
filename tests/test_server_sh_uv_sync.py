"""server.sh never lets ``uv run`` sync inside a container (TASK-1476).

A start-time sync rebuilt the editable install and wrote
``src/llm_bawt.egg-info`` into the bind-mounted source tree on every app start,
which the ops source guards refuse as an untracked file under a source path.
The image owns dependencies, so a container start runs ``uv run --no-sync``;
a bare host checkout keeps syncing.

These run the real script under bash in file-log mode with a fake ``uv`` that
records its argv and exits, so ``start`` returns on its own.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER_SH = REPO / "server.sh"

FAKE_UV = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_UV_LOG"
exit 0
"""


@pytest.fixture
def run_start(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy2(SERVER_SH, app / "server.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv.log"
    absent_marker = tmp_path / "not-a-container"

    def run(*args: str, **extra_env: str) -> list[str]:
        env = {k: v for k, v in os.environ.items() if k != "UV_NO_SYNC"}
        env.update(
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            FAKE_UV_LOG=str(log),
            LLM_BAWT_CONTAINER_MARKER=str(absent_marker),
        )
        env.update(extra_env)
        subprocess.run(
            ["bash", str(app / "server.sh"), "start", *args],
            env=env, check=True, capture_output=True, text=True, timeout=60,
        )
        # File mode backgrounds both services (`nohup uv run … &`) and returns
        # without waiting, so the fake uv may not have logged yet. Wait for
        # both role lines; the deadline only guards a hang.
        deadline = time.monotonic() + 30
        while True:
            calls = log.read_text().splitlines() if log.exists() else []
            if (any("llm-mcp-server" in c for c in calls)
                    and any("llm-service" in c for c in calls)):
                return calls
            if time.monotonic() > deadline:
                pytest.fail(f"fake uv never saw both services: {calls}")
            time.sleep(0.02)

    run.marker = tmp_path / "dockerenv"  # type: ignore[attr-defined]
    return run


def test_container_start_never_syncs(run_start):
    run_start.marker.touch()
    calls = run_start(LLM_BAWT_CONTAINER_MARKER=str(run_start.marker))
    assert calls and all("--no-sync" in c for c in calls)


def test_uv_no_sync_env_disables_sync(run_start):
    calls = run_start(UV_NO_SYNC="1")
    assert calls and all("--no-sync" in c for c in calls)


def test_dev_mode_disables_sync(run_start):
    calls = run_start("--dev")
    assert calls and all("--no-sync" in c for c in calls)


def test_host_checkout_still_syncs(run_start):
    calls = run_start()
    assert calls and not any("--no-sync" in c for c in calls)


def test_app_image_leaves_sync_policy_to_server_sh():
    # An image ENV would be merged into the ops source plan, where UV_NO_SYNC
    # is unclassified (and the UV_ prefix is reserved), refusing the bind. The
    # container marker already makes server.sh pass --no-sync.
    assert "UV_NO_SYNC" not in (REPO / "Dockerfile.app").read_text()
