"""Host-only non-SHIP heavy jobs: one capped container, durable result + callback.

No Docker socket, provider secrets, live DB settings or home mounts enter a job.
This does not mint SHIP receipts or modify the TASK-1865 fixed-workload action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

from scripts.heavy_slot import default_lock, heavy_slot, reserve_ticket

MEMORY = 16 * 1024**3
HEADROOM = 4 * 1024**3
WALL_SECONDS = 1800
DEFAULT_ROOT = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "bawt-heavy-queue"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def available_memory(path=Path("/proc/meminfo")):
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host MemAvailable is unavailable")


def docker_args(job):
    # Same-path mounts preserve task-local node_modules and executable paths.
    # The caller supplies a prepared toolchain image; no dependency or DB
    # provisioning happens implicitly, and no home/socket is ever inherited.
    command = [
        "docker", "create", "--name", job["container"], "--init",
        "--memory", str(MEMORY), "--memory-swap", str(MEMORY), "--cpus", "8",
        "--pids-limit", "4096", "--restart", "no", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", job["cwd"],
        "--env", "BAWT_AGENT_BRIDGE=0", "--env", "BAWT_HEAVY_QUEUE=1",
        "--env", "BRIDGE_BROKER_TOKEN_FILE=", "--env", "HOME=/tmp/heavy-home",
        "--env", "LLM_BAWT_TEST_WORKERS=4",
        "--entrypoint", "/usr/bin/timeout",
    ]
    for directory in [job["cwd"], *job["binds"]]:
        command += ["--mount", f"type=bind,source={directory},target={directory}"]
    # Reap and measure the entire job's cgroup, not one compiler/test PID.
    wrapper = (
        'mkdir -p "$HOME"; "$@"; code=$?; '
        f'printf "\\nheavy-queue-peak:{job["id"]}:%s\\n" '
        '"$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || printf unavailable)"; exit "$code"'
    )
    # The container-owned deadline survives SIGKILL/OOM of this host worker.
    return [*command, job["image"], "--signal=KILL", str(WALL_SECONDS),
            "/bin/sh", "-c", wrapper, "heavy-queue", *job["command"]]


def notify(job, receipt, *, opener=urllib.request.urlopen):
    payload = {
        "sender_bot_id": job["notify_bot"], "target_bot_id": job["notify_bot"],
        "message": (f"Heavy queue {job['id']} finished: {receipt['status']}; "
                    f"exit={receipt.get('exit_code')}; log={receipt['log_path']}; "
                    f"receipt={receipt['receipt_path']}; peak_bytes={receipt.get('memory_peak_bytes')}."),
        "idempotency_key": f"heavy-queue:{job['id']}", "task_id": job.get("task_id"),
        "prefer_steer": False,
    }
    request = urllib.request.Request(
        job["notify_url"].rstrip("/") + "/v1/inter-bot-deliveries",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Do not copy arbitrary test output into a prompt or retain secret-bearing
    # HTTP exception bodies. A durable delivery ID proves enqueue, not completion.
    with opener(request, timeout=15) as response:
        result = json.load(response)
    delivery_id = result.get("delivery_id")
    if not isinstance(delivery_id, str) or not delivery_id:
        raise RuntimeError("callback response has no durable delivery ID")
    return delivery_id


def remove_container(name):
    """Confirm absence via a successful inventory, not an ambiguous inspect error."""
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        check = subprocess.run(
            ["docker", "container", "ls", "--all", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"],
            capture_output=True, timeout=30)
        return check.returncode == 0 and not check.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_peak(log_path, identifier):
    # The shell emits the measurement at EOF. Bound host-side memory even when
    # a generic command produced gigabytes or one enormous unterminated line.
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 65536))
        tail = handle.read(65536)
    match = re.search(rb"heavy-queue-peak:" + identifier.encode() + rb":(\d+)", tail)
    return int(match[1]) if match else None


def run_job(directory, ticket_fd):
    job = json.loads((directory / "job.json").read_text())
    receipt_path = directory / "receipt.json"
    receipt = {
        "id": job["id"], "status": "QUEUED", "submitted_at": job["submitted_at"],
        "log_path": str(directory / "run.log"), "receipt_path": str(receipt_path),
        "exit_code": None, "memory_peak_bytes": None, "callback_delivery_id": None,
    }
    save(receipt_path, receipt)
    interrupted = False
    cleaning = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        if not cleaning:
            raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        # Hold the host slot through positive container removal. Admission waits
        # for health; a gate never waits for multiple queued jobs.
        with heavy_slot(kind="queue", wait_seconds=86400,
                        ticket=os.fdopen(ticket_fd, "r+"), ticket_path=job["ticket_path"],
                        ready=lambda: available_memory() >= MEMORY + HEADROOM):
            active = default_lock().with_name(default_lock().name + ".tickets") / "active.json"
            try:
                # Write before create, not merely in finally: SIGKILL cannot
                # run Python cleanup, but the next participant sees uncertainty.
                save(active, {"job": job["id"], "container": job["container"],
                              "receipt_path": str(receipt_path)})
                receipt.update(status="RUNNING", started_at=stamp())
                save(receipt_path, receipt)
                deadline = time.monotonic() + WALL_SECONDS
                with (directory / "run.log").open("wb") as log:
                    subprocess.run(docker_args(job), check=True, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=60)
                    child = subprocess.Popen(["docker", "start", "--attach", job["container"]],
                                             stdout=log, stderr=subprocess.STDOUT)
                    try:
                        child.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        receipt["status"] = "TIMED_OUT"
                    finally:
                        # Stopping the daemon workload, not merely docker attach,
                        # enforces the wall deadline and prevents surviving workers.
                        if child.poll() is None:
                            subprocess.run(["docker", "kill", job["container"]],
                                           stdout=log, stderr=subprocess.STDOUT, timeout=30, check=False)
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=5)
                state = json.loads(subprocess.check_output(
                    ["docker", "inspect", job["container"]], timeout=30))[0]
                if state["State"].get("Running") or state["State"].get("Status") not in {"exited", "dead"}:
                    raise RuntimeError("container exit not confirmed")
                receipt["exit_code"] = state["State"]["ExitCode"]
                receipt["oom_killed"] = state["State"].get("OOMKilled", False)
                receipt["container_memory_bytes"] = state["HostConfig"]["Memory"]
                receipt["memory_peak_bytes"] = read_peak(directory / "run.log", job["id"])
                if receipt["status"] != "TIMED_OUT":
                    receipt["status"] = "PASSED" if receipt["exit_code"] == 0 else "FAILED"
            finally:
                # A lost create response can still have created the named
                # container. Always reconcile this unique name before release.
                # Repeated soft interrupts cannot abandon this bounded cleanup.
                cleaning = True
                if not remove_container(job["container"]):
                    # Never start another workload after uncertain cleanup.
                    receipt["status"] = "CLEANUP_UNCERTAIN"
                    hold = default_lock().with_name(default_lock().name + ".tickets") / "hold"
                    hold.write_text(json.dumps({"job": job["id"], "container": job["container"]}))
                    receipt["cleanup_hold"] = str(hold)
                    save(receipt_path, receipt)
                    raise RuntimeError("container cleanup uncertain")
                active.unlink(missing_ok=True)
                cleaning = False
    except KeyboardInterrupt:
        if receipt["status"] != "CLEANUP_UNCERTAIN":
            receipt["status"] = "INTERRUPTED"
    except Exception as exc:
        if receipt["status"] != "CLEANUP_UNCERTAIN":
            receipt["status"] = "FAILED"
        receipt["error"] = type(exc).__name__
    finally:
        cleaning = False
        if interrupted and receipt["status"] != "CLEANUP_UNCERTAIN":
            receipt["status"] = "INTERRUPTED"
        receipt["finished_at"] = stamp()
        receipt["interrupted"] = interrupted
        save(receipt_path, receipt)
        # Retry enqueue with one idempotency key. A delivery already accepted
        # before a lost response is replayed, never a second callback.
        attempts = 0
        while True:
            try:
                receipt["callback_delivery_id"] = notify(job, receipt)
                receipt.pop("callback_error", None)
                break
            except Exception as exc:
                attempts += 1
                receipt["callback_error"] = type(exc).__name__
            finally:
                save(receipt_path, receipt)
            time.sleep(min(60, 5 * attempts))
    return 0 if receipt["status"] == "PASSED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--cwd", type=Path, required=True)
    submit.add_argument("--image", required=True)
    submit.add_argument("--bind", action="append", type=Path, default=[])
    submit.add_argument("--notify-url", required=True, help="llm-bawt service base URL")
    submit.add_argument("--notify-bot", required=True)
    submit.add_argument("--task-id")
    submit.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    submit.add_argument("command", nargs=argparse.REMAINDER)
    worker = sub.add_parser("_run")
    worker.add_argument("directory", type=Path)
    worker.add_argument("ticket_fd", type=int)
    status = sub.add_parser("status")
    status.add_argument("directory", type=Path)
    callback = sub.add_parser("notify")
    callback.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.action == "status":
        print((args.directory / "receipt.json").read_text())
        return 0
    if args.action == "notify":
        job = json.loads((args.directory / "job.json").read_text())
        receipt = json.loads((args.directory / "receipt.json").read_text())
        if receipt["status"] in {"QUEUED", "RUNNING"}:
            parser.error("only terminal job results may be notified")
        receipt["callback_delivery_id"] = notify(job, receipt)
        receipt.pop("callback_error", None)
        save(args.directory / "receipt.json", receipt)
        return 0
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        parser.error("run heavy-queue on the test host; never mount Docker into a bridge")
    if args.action == "_run":
        return run_job(args.directory, args.ticket_fd)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide an argv command after --")
    cwd = args.cwd.resolve(strict=True)
    binds = [path.resolve(strict=True) for path in args.bind]
    if not cwd.is_dir() or any(not path.is_dir() for path in binds):
        parser.error("cwd and bind paths must be directories")
    forbidden = {Path("/"), Path.home(), Path("/var/run"), Path("/run"), Path("/var")}
    if any(path in forbidden or path.is_relative_to("/run") or path.is_relative_to("/var/run")
           for path in [cwd, *binds]):
        parser.error("mount task directories only, not host roots, homes or runtime sockets")
    identifier = uuid.uuid4().hex
    directory = args.root.resolve() / identifier
    directory.mkdir(parents=True, mode=0o700)
    job = dict(id=identifier, container="bawt-heavy-" + identifier, cwd=str(cwd),
               image=args.image, command=command, binds=[str(path) for path in binds],
               notify_url=args.notify_url, notify_bot=args.notify_bot, task_id=args.task_id,
               submitted_at=stamp())
    ticket, ticket_path = reserve_ticket("queue")
    try:
        job["ticket_path"] = str(ticket_path)
        save(directory / "job.json", job)
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "scripts.heavy_queue", "_run", str(directory), str(ticket.fileno())],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(ticket.fileno(),))
    finally:
        # The worker inherits this already-locked ticket before submit returns:
        # FIFO means admission order, never worker scheduling or wall-clock order.
        ticket.close()
    print(json.dumps({"id": identifier, "pid": process.pid, "directory": str(directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
