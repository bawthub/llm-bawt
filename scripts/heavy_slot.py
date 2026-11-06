"""Shared host heavy-work slot protocol (also vendored by BawtHub).

The original llm-bawt flock remains the exclusion boundary. Live locked tickets
add FIFO within a class and gate-first, non-preemptive admission. Dead processes
release both locks; stale ticket files never reserve capacity.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def default_lock() -> Path:
    return Path(tempfile.gettempdir()) / "llm-bawt-test-suite.lock"


def reserve_ticket(kind, lock_path=None):
    lock_path = Path(lock_path) if lock_path is not None else default_lock()
    tickets = lock_path.with_name(lock_path.name + ".tickets")
    tickets.mkdir(parents=True, exist_ok=True)
    with (tickets / "registry.lock").open("a+") as registry:
        fcntl.flock(registry, fcntl.LOCK_EX)
        counter = tickets / "sequence"
        number = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(number))
        path = tickets / f"{kind}-{number:020d}-{uuid.uuid4().hex}"
        handle = path.open("x+")
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(str(os.getpid()))
        handle.flush()
        return handle, path


def cleanup_refusal(path):
    try:
        container = json.loads(path.read_text()).get("container", "unknown")
    except (OSError, ValueError):
        container = "unknown"
    return (f"heavy-work cleanup hold: {path}; container={container}. "
            "Reconcile/remove that exact container, confirm absence with a successful "
            "Docker container inventory, then remove this marker; never clear it blindly.")


@contextmanager
def heavy_slot(*, kind="gate", wait_seconds=2700, poll_seconds=1.0,
               lock_path=None, ready=lambda: True, clock=time.monotonic, sleep=time.sleep,
               ticket=None, ticket_path=None):
    if kind not in {"gate", "queue"}:
        raise ValueError("heavy slot class must be gate or queue")
    lock_path = Path(lock_path) if lock_path is not None else default_lock()
    tickets = lock_path.with_name(lock_path.name + ".tickets")
    tickets.mkdir(parents=True, exist_ok=True)
    started = clock()
    # Registry arbitration closes the check-tickets/acquire-flock race. It is
    # held only for admission, never for a workload or while waiting.
    with (tickets / "registry.lock").open("a+") as registry, lock_path.open("a+") as slot:
        try:
            if ticket is None:
                ticket, ticket_path = reserve_ticket(kind, lock_path)
            ticket_path = Path(ticket_path)
            announced = False
            while True:
                admitted = False
                fcntl.flock(registry, fcntl.LOCK_EX)
                try:
                    live = []
                    for path in tickets.iterdir():
                        if not path.name.startswith(("gate-", "queue-")):
                            continue
                        if path == ticket_path:
                            live.append(path.name)
                            continue
                        try:
                            with path.open("r+") as probe:
                                try:
                                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    live.append(path.name)
                                    continue
                                path.unlink(missing_ok=True)
                        except FileNotFoundError:
                            pass
                    # Gate tickets sort before queue tickets; sequence orders
                    # each class. Running work keeps its ticket until release.
                    if (tickets / "hold").exists():
                        raise RuntimeError(cleanup_refusal(tickets / "hold"))
                    if min(live) == ticket_path.name and ready():
                        try:
                            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            # Persisted before Docker create. A dead Python owner
                            # cannot release a still-running daemon workload into
                            # an apparently free slot. Only positive removal clears it.
                            if (tickets / "active.json").exists():
                                raise RuntimeError(cleanup_refusal(tickets / "active.json"))
                            admitted = True
                finally:
                    fcntl.flock(registry, fcntl.LOCK_UN)
                if admitted:
                    break
                waited = clock() - started
                if waited >= wait_seconds:
                    raise RuntimeError(f"another llm-bawt test suite is running on this host "
                                       f"(or waiting/admission held; waited {waited:.1f}s)")
                if not announced:
                    print(f"Test runner lock: busy, waiting up to {wait_seconds:.0f}s", flush=True)
                    announced = True
                sleep(min(poll_seconds, wait_seconds - waited))
            if announced:
                print(f"Test runner lock: acquired after {clock() - started:.1f}s", flush=True)
            yield
        finally:
            if ticket is not None:
                # Deleting under registry arbitration prevents a stale-file
                # observer from racing with a normal ticket release.
                fcntl.flock(registry, fcntl.LOCK_EX)
                try:
                    ticket_path.unlink(missing_ok=True)
                    ticket.close()
                finally:
                    fcntl.flock(registry, fcntl.LOCK_UN)
