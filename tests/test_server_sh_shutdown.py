"""server.sh foreground mode forwards SIGTERM/SIGINT to its children (TASK-1286).

The app container's PID 1 is ``server.sh start --stdout``. Before the trap,
bash ignored SIGTERM as PID 1, so ``docker stop``/restart waited out the 10s
timeout and SIGKILLed the app without running its lifespan shutdown.

These tests run the real script under bash with a fake ``uv`` on PATH that
execs a small Python child per service. The real ``uv run`` layer (does uv
forward TERM to python?) is covered by the scratch-container proof recorded on
TASK-1286, not here.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER_SH = REPO / "server.sh"

# Under Docker, server.sh is PID 1 with default signal dispositions. The test
# suite is not: run as ``nohup make test-all … &`` over ssh (the recommended
# background gate), the pytest process inherits SIGHUP (nohup) and
# SIGINT/SIGQUIT (``&`` with no job control) as SIG_IGN, Python keeps an
# inherited ignore, and a bash child cannot trap a signal ignored at entry
# (bash(1), trap). The SIGINT case then hangs instead of shutting down. This
# launcher restores the defaults and execs server.sh in the same pid, so
# ``proc.pid`` is still bash. (Not ``preexec_fn``: unsafe under xdist's threads.)
RESET_SIGNALS = (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP, signal.SIGTERM)
SIGNAL_RESET_LAUNCHER = (
    "import os, signal, sys\n"
    f"for signum in {tuple(int(s) for s in RESET_SIGNALS)}:\n"
    "    signal.signal(signum, signal.SIG_DFL)\n"
    "os.execvp(sys.argv[1], sys.argv[1:])\n"
)


def launch_with_default_signals(argv: list[str], **popen_kwargs) -> subprocess.Popen:
    """Start ``argv`` with RESET_SIGNALS at SIG_DFL whatever this process inherited."""
    return subprocess.Popen(
        [sys.executable, "-I", "-c", SIGNAL_RESET_LAUNCHER, *argv], **popen_kwargs,
    )


def _ignored_mask(status_text: str) -> int:
    line = next(l for l in status_text.splitlines() if l.startswith("SigIgn:"))
    return int(line.split()[1], 16)

FAKE_UV = """#!/usr/bin/env bash
role=""
probe=""
for a in "$@"; do
  case "$a" in
    llm-service) role=service ;;
    llm-mcp-server) role=mcp ;;
    python) probe=1 ;;
  esac
done
if [[ -z "$role" && -n "$probe" ]]; then
  exit 0  # server.sh's MCP readiness probe
fi
exec "$FAKE_PYTHON" "$FAKE_DIR/child.py" "$role"
"""

FAKE_CHILD = """
import os, pathlib, signal, sys, time
role = sys.argv[1]
d = pathlib.Path(os.environ["FAKE_DIR"])
ignore = os.environ.get("FAKE_IGNORE_TERM") == role

def on_term(signum, frame):
    (d / f"{role}.term").touch()
    if not ignore:
        sys.exit(0)

signal.signal(signal.SIGTERM, on_term)
(d / f"{role}.pid").write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as gone.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _monitor_sleep_running(bash_pid: int) -> bool:
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().split()
            cmdline = (stat.parent / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if fields[3] == str(bash_pid) and cmdline.startswith(b"sleep\x0010"):
            return True
    return False


@pytest.fixture
def stack(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy2(SERVER_SH, app / "server.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "child.py").write_text(FAKE_CHILD)
    procs: list[subprocess.Popen] = []

    def start(**extra_env: str) -> subprocess.Popen:
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_DIR": str(fake),
            "FAKE_PYTHON": sys.executable,
            **extra_env,
        }
        proc = launch_with_default_signals(
            ["bash", str(app / "server.sh"), "start", "--stdout"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        procs.append(proc)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (fake / "service.pid").exists() and (fake / "mcp.pid").exists() \
                    and _monitor_sleep_running(proc.pid):
                return proc
            assert proc.poll() is None, proc.communicate()[0]
            time.sleep(0.02)
        raise AssertionError("server.sh never reached its monitor loop")

    yield start, fake, app
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)  # timing-ok: Popen.wait raises TimeoutExpired, never passes silently
    for name in ("service.pid", "mcp.pid"):
        pid_file = fake / name
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.real_process(reason="checks real signal dispositions of a launched child process")
def test_launcher_resets_signals_the_suite_inherited_as_ignored():
    """A parent that ignores INT/QUIT/HUP (``nohup … &``) still gets a trappable child."""
    probe = (
        "import signal, subprocess, sys\n"
        f"for signum in {tuple(int(s) for s in RESET_SIGNALS[:3])}:\n"
        "    signal.signal(signum, signal.SIG_IGN)\n"
        "bash = ['bash', '-c', 'cat /proc/$$/status']\n"
        "plain = subprocess.run(bash, capture_output=True, text=True, check=True).stdout\n"
        f"reset = subprocess.run([sys.executable, '-I', '-c', {SIGNAL_RESET_LAUNCHER!r}, *bash],\n"
        "                       capture_output=True, text=True, check=True).stdout\n"
        "sys.stdout.write(plain + '\\n===\\n' + reset)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True,
                         check=True).stdout
    plain, reset = out.split("\n===\n")
    wanted = sum(1 << (int(s) - 1) for s in RESET_SIGNALS)
    assert _ignored_mask(plain) & wanted, "the simulated nohup parent did not pass its ignores on"
    assert _ignored_mask(reset) & wanted == 0, reset


@pytest.mark.integration
@pytest.mark.offline_integration
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_shutdown_signal_reaches_both_services_and_exits_cleanly(stack, sig):
    start, fake, app = stack
    proc = start()
    service_pid = int((fake / "service.pid").read_text())
    mcp_pid = int((fake / "mcp.pid").read_text())

    sent = time.monotonic()
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=8)
    elapsed = time.monotonic() - sent

    assert proc.returncode == 0, out
    # The trap fires during `wait` on the monitor's background sleep, not
    # after the old foreground `sleep 10` finished.
    # timing-ok: a real shell trap: under 3s vs the 10s foreground sleep it must interrupt
    assert elapsed < 3, (elapsed, out)
    assert (fake / "service.term").exists() and (fake / "mcp.term").exists()
    assert not _alive(service_pid) and not _alive(mcp_pid)
    assert f"[server] {signal.Signals(sig).name} received" in out
    assert "[server] Shutdown complete" in out
    assert "force killing" not in out
    assert not (app / ".run" / "llm-service.pid").exists()
    assert not (app / ".run" / "mcp-server.pid").exists()


@pytest.mark.integration
@pytest.mark.offline_integration
def test_a_service_that_ignores_sigterm_is_killed_after_the_wait_and_exit_is_nonzero(stack):
    start, fake, _app = stack
    proc = start(FAKE_IGNORE_TERM="service", LLM_BAWT_SHUTDOWN_WAIT_SECONDS="1")
    service_pid = int((fake / "service.pid").read_text())
    mcp_pid = int((fake / "mcp.pid").read_text())

    sent = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=8)
    elapsed = time.monotonic() - sent

    assert proc.returncode == 1, out
    # timing-ok: a real shell escalation: 1s TERM grace, then KILL, vs the stubborn child's 10s
    assert 1 <= elapsed < 4, (elapsed, out)
    assert (fake / "service.term").exists(), "the stubborn service still got TERM first"
    assert f"force killing: {service_pid}" in out
    assert not _alive(service_pid) and not _alive(mcp_pid)
    assert "[server] Shutdown complete" in out
