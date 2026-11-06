"""No live Docker, app, database, provider or fleet access."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import heavy_guard, heavy_queue, heavy_slot


def test_bridge_markers_do_not_accept_queue_bypass():
    assert heavy_guard.in_bridge({"BAWT_AGENT_BRIDGE": "1", "BAWT_HEAVY_QUEUE": "1"})
    assert heavy_guard.in_bridge({"BRIDGE_BROKER_TOKEN_FILE": "/token"})
    assert not heavy_guard.in_bridge({"BAWT_HEAVY_QUEUE": "1"})


def test_python_file_focus_and_expanded_whole_suite(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_a.py", "test_b.py"):
        (tests / name).touch()
    assert heavy_guard.focused_pytest(["-k", "test_b.py", "tests/test_a.py::test_x"], tmp_path)
    assert not heavy_guard.focused_pytest(["-k", "tests/test_a.py"], tmp_path)
    assert not heavy_guard.focused_pytest(["tests"], tmp_path)
    assert not heavy_guard.focused_pytest(["tests/*.py"], tmp_path)
    assert not heavy_guard.focused_pytest(["tests/test_a.py", "tests/test_b.py"], tmp_path)


@pytest.mark.parametrize("command", [
    ["make", "-n", "test"], ["make", "-n", "typecheck"],
])
def test_make_entrypoints_admit_before_dependency_install(command):
    output = subprocess.check_output(command, text=True)
    assert "scripts.heavy_guard" in output


def test_real_bridge_runner_refusal_happens_before_uv():
    env = {**os.environ, "BAWT_AGENT_BRIDGE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "scripts.heavy_guard", "runner", "--", "must-not-exist",
         "python", "-m", "scripts.test_suite", "unit"], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "heavy-queue submit" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_caps_and_no_socket_or_implicit_home_or_db():
    job = dict(id="abc", container="bawt-heavy-abc", cwd="/work/task", image="toolchain:test",
               binds=[], command=["node", "--version"])
    args = heavy_queue.docker_args(job)
    for flag, value in [("--memory", str(16 * 1024**3)), ("--memory-swap", str(16 * 1024**3)),
                        ("--cpus", "8"), ("--pids-limit", "4096"), ("--restart", "no"),
                        ("--network", "none")]:
        assert args[args.index(flag) + 1] == value
    assert "--init" in args
    assert not any("docker.sock" in arg or "DATABASE_URL" in arg for arg in args)
    assert args.count("--mount") == 1
    assert "BAWT_AGENT_BRIDGE=0" in args
    assert "BAWT_HEAVY_QUEUE=1" in args


def test_memory_admission_requires_actual_available(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 64000000 kB\nMemAvailable: 20971520 kB\n")
    assert heavy_queue.available_memory(path) == 20 * 1024**3
    path.write_text("MemTotal: 64000000 kB\n")
    with pytest.raises(RuntimeError, match="MemAvailable"):
        heavy_queue.available_memory(path)


def test_callback_enqueues_one_idempotent_self_delivery_without_log_output():
    requests = []

    @contextmanager
    def opener(request, timeout):
        requests.append((request, timeout))
        yield io.StringIO('{"delivery_id":"delivery-test"}')

    job = dict(id="job", notify_bot="requester", notify_url="http://example.test", task_id="TASK-1")
    receipt = dict(status="FAILED", exit_code=9, log_path="/logs/job", receipt_path="/receipts/job")
    assert heavy_queue.notify(job, receipt, opener=opener) == "delivery-test"
    assert heavy_queue.notify(job, receipt, opener=opener) == "delivery-test"
    body = json.loads(requests[0][0].data)
    assert body["sender_bot_id"] == body["target_bot_id"] == "requester"
    assert body["idempotency_key"] == "heavy-queue:job"
    assert body["prefer_steer"] is False
    assert body["task_id"] == "TASK-1"
    assert requests[0][0].data == requests[1][0].data


@pytest.mark.parametrize("response", [{}, {"delivery_id": None}])
def test_callback_requires_durable_receipt(response):
    @contextmanager
    def opener(_request, timeout):
        yield io.StringIO(json.dumps(response))

    with pytest.raises(RuntimeError, match="delivery ID"):
        heavy_queue.notify(dict(id="job", notify_bot="requester", notify_url="http://example.test"),
                           dict(status="FAILED", log_path="/log", receipt_path="/receipt"), opener=opener)


def test_ambiguous_docker_errors_do_not_prove_cleanup(monkeypatch):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"daemon unavailable")
    monkeypatch.setattr(heavy_queue.subprocess, "run", run)
    assert not heavy_queue.remove_container("bawt-heavy-test")


def test_absence_requires_successful_empty_inventory(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
    monkeypatch.setattr(heavy_queue.subprocess, "run", run)
    assert heavy_queue.remove_container("bawt-heavy-test")
    assert calls[-1][:3] == ["docker", "container", "ls"]


def test_stale_tickets_are_not_capacity_and_sequence_is_fifo(tmp_path):
    lock = tmp_path / "slot.lock"
    first, first_path = heavy_slot.reserve_ticket("queue", lock)
    second, second_path = heavy_slot.reserve_ticket("queue", lock)
    try:
        assert first_path.name < second_path.name
        first.close()
        with heavy_slot.heavy_slot(kind="queue", lock_path=lock, ticket=second,
                                   ticket_path=second_path, wait_seconds=0):
            assert not first_path.exists()
    finally:
        first.close()
        second.close()


def test_gate_pending_blocks_a_reserved_queue_ticket(tmp_path):
    lock = tmp_path / "slot.lock"
    queued, queue_path = heavy_slot.reserve_ticket("queue", lock)
    gate, gate_path = heavy_slot.reserve_ticket("gate", lock)
    try:
        with pytest.raises(RuntimeError, match="another llm-bawt"):
            with heavy_slot.heavy_slot(kind="queue", lock_path=lock, ticket=queued,
                                       ticket_path=queue_path, wait_seconds=0):
                pytest.fail("queue passed a pending gate")
        with heavy_slot.heavy_slot(kind="gate", lock_path=lock, ticket=gate,
                                   ticket_path=gate_path, wait_seconds=0):
            assert gate_path.exists()
    finally:
        gate.close()
        queued.close()


def test_health_wait_and_persistent_cleanup_hold_fail_closed(tmp_path):
    lock = tmp_path / "slot.lock"
    with pytest.raises(RuntimeError):
        with heavy_slot.heavy_slot(kind="queue", lock_path=lock, ready=lambda: False, wait_seconds=0):
            pytest.fail("low-memory admission succeeded")
    tickets = lock.with_name(lock.name + ".tickets")
    (tickets / "hold").write_text("cleanup uncertain")
    with pytest.raises(RuntimeError):
        with heavy_slot.heavy_slot(kind="gate", lock_path=lock, wait_seconds=0):
            pytest.fail("gate ignored cleanup hold")
    (tickets / "hold").unlink()
    with heavy_slot.heavy_slot(kind="gate", lock_path=lock, wait_seconds=0):
        pass


@pytest.mark.parametrize("timeout,cleanup", [(False, True), (True, True), (False, False)])
def test_job_terminal_receipt_timeout_cleanup_and_wakeup(monkeypatch, tmp_path, timeout, cleanup):
    monkeypatch.setattr(heavy_slot.tempfile, "gettempdir", lambda: str(tmp_path))
    job = dict(id="abc", container="bawt-heavy-abc", cwd=str(tmp_path), image="prepared:test",
               binds=[], command=["fake-check"], submitted_at="now",
               notify_url="http://example.test", notify_bot="requester")
    ticket, ticket_path = heavy_slot.reserve_ticket("queue")
    job["ticket_path"] = str(ticket_path)
    heavy_queue.save(tmp_path / "job.json", job)
    monkeypatch.setattr(heavy_queue, "available_memory", lambda: 64 * 1024**3)
    monkeypatch.setattr(heavy_queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(heavy_queue, "remove_container", lambda name: cleanup)
    callbacks = []
    monkeypatch.setattr(heavy_queue, "notify", lambda job, receipt:
                        callbacks.append(dict(receipt)) or "delivery-abc")
    timed_out = False

    class Child:
        stopped = False
        def wait(self, timeout=None):
            nonlocal timed_out
            if not timed_out and globals_timeout:
                timed_out = True
                raise subprocess.TimeoutExpired("fake-check", 1800)
            self.stopped = True
            return 137 if globals_timeout else 0
        def poll(self):
            return 0 if self.stopped else None
        def kill(self):
            self.stopped = True

    globals_timeout = timeout
    def run(args, **kwargs):
        if args[:2] == ["docker", "create"]:
            kwargs["stdout"].write(b"heavy-queue-peak:abc:123456\n")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(heavy_queue.subprocess, "run", run)
    monkeypatch.setattr(heavy_queue.subprocess, "Popen", lambda *args, **kwargs: Child())
    state = {"State": {"ExitCode": 137 if timeout else 0, "Status": "exited", "Running": False},
             "HostConfig": {"Memory": 16 * 1024**3}}
    monkeypatch.setattr(heavy_queue.subprocess, "check_output", lambda *args, **kwargs: json.dumps([state]).encode())
    result = heavy_queue.run_job(tmp_path, os.dup(ticket.fileno()))
    ticket.close()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == ("CLEANUP_UNCERTAIN" if not cleanup else "TIMED_OUT" if timeout else "PASSED")
    assert receipt["memory_peak_bytes"] == 123456
    assert receipt["callback_delivery_id"] == "delivery-abc"
    assert len(callbacks) == 1
    assert result == (0 if not timeout and cleanup else 1)
    if not cleanup:
        assert Path(receipt["cleanup_hold"]).is_file()
