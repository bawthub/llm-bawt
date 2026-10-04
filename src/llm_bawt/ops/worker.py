"""Standalone Docker-owned operation worker. STANDARD LIBRARY ONLY.

Executed as a file in a small immutable image, never importing llm_bawt or its
DB/config. Request + durable receipt live on a dedicated Docker volume. A
started marker and flock prevent replay even if somebody restarts the worker.
"""
from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

# Image deploy/rollback (TASK-997) lives in the sibling stdlib-only module. The
# app/tests import it as a package; the worker image runs this file directly
# (see __main__). Single source of truth for IMAGE_ACTIONS is image_deploy.
if __package__:
    from .image_deploy import IMAGE_ACTIONS, ImageDeployer
else:  # pragma: no cover - standalone worker image
    # `python3 -I` drops the script directory from sys.path; re-add ONLY this
    # file's own image-owned, read-only directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from image_deploy import IMAGE_ACTIONS, ImageDeployer


OPS_JOB_LABEL = "llm-bawt.ops.job"


def target_selector(spec: dict, args: dict) -> tuple[str | None, dict | None]:
    """The container a restart/start/stop/pull spec targets.

    Returns ``(name, None)`` or ``(None, compose label filters)``. Single source
    of truth shared by the worker (execution) and the app's approval-time
    target check (TASK-1002), so both resolve exactly the same container.
    """
    if "container_name" in spec or "container_name_from_arg" in spec:
        return str(spec.get("container_name") or args[spec["container_name_from_arg"]]), None
    service = spec.get("compose_service") or args[spec["compose_service_from_arg"]]
    return None, {"label": [f"com.docker.compose.project={spec['compose_project']}",
                            f"com.docker.compose.service={service}"]}


def is_ops_worker(labels: dict | None) -> bool:
    """Images can carry inherited Compose labels. An operation worker built from
    such an image must never qualify as its own target, even if a future
    dispatcher forgets to neutralize those labels."""
    return OPS_JOB_LABEL in (labels or {})


class UnixHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


class DockerAPI:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.side_effect_started = False

    def request(self, method: str, path: str, *, body: dict | None = None,
                headers: dict | None = None, raw: bool = False):
        conn = UnixHTTPConnection("localhost", timeout=self.timeout)
        try:
            send_headers = dict(headers or {})
            payload = None
            if body is not None:
                payload = json.dumps(body).encode()
                send_headers["Content-Type"] = "application/json"
            conn.request(method, path, body=payload, headers=send_headers)
            response = conn.getresponse()
            if raw:
                # Exec attach stream: bounded, returned verbatim for demuxing.
                data = response.read(1024 * 1024 + 1)
                if response.status >= 300 or len(data) > 1024 * 1024:
                    raise RuntimeError(f"Docker HTTP {response.status} (raw stream)")
                return data
            # Pull emits an unbounded progress stream. Drain in bounded chunks;
            # retain only the tail to catch Docker's in-stream error response.
            tail = b""
            if path.startswith("/images/create"):
                line = b""
                while chunk := response.read(65536):
                    line += chunk
                    lines = line.split(b"\n")
                    line = lines.pop()
                    if len(line) > 1024 * 1024:
                        raise RuntimeError("oversized Docker progress record")
                    for record in lines:
                        if record.strip() and json.loads(record).get("error"):
                            raise RuntimeError("Docker image pull failed")
                tail = line
                if tail.strip() and json.loads(tail).get("error"):
                    raise RuntimeError("Docker image pull failed")
            else:
                tail = response.read(4 * 1024 * 1024 + 1)
                if len(tail) > 4 * 1024 * 1024:
                    raise RuntimeError("Docker response too large")
            if response.status >= 300:
                raise RuntimeError(f"Docker HTTP {response.status}: {tail[-2048:].decode(errors='replace')}")
            return json.loads(tail) if tail.strip() else None
        finally:
            conn.close()

    def execute(self, spec: dict, args: dict) -> str:
        name, filters = target_selector(spec, args)
        if name is not None:
            container = self.request("GET", f"/containers/{quote(name, safe='')}/json")
        else:
            matches = self.request("GET", "/containers/json?" + urlencode({"all": "1", "filters": json.dumps(filters)}))
            matches = [row for row in matches if not is_ops_worker(row.get("Labels"))]
            if len(matches) != 1:
                raise RuntimeError(f"selector matched {len(matches)} containers, expected one")
            container = self.request("GET", f"/containers/{matches[0]['Id']}/json")
        action = spec["action"]
        container_id = quote(container["Id"], safe="")
        grace = spec.get("stop_grace_seconds", 10)
        if action == "pull":
            image = container["Config"]["Image"]
            path = "/images/create?" + urlencode({"fromImage": image})
        else:
            path = f"/containers/{container_id}/{action}"
            if action in ("stop", "restart"):
                path += f"?t={grace}"
        # Once handed to Docker, timeout/transport failure means the side effect
        # MAY still finish in the daemon. Never interpret that as safe to retry.
        self.side_effect_started = True
        self.request("POST", path)
        return f"Docker {action} completed for {container['Id']}"


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(request_path: Path, expected_hash: str, *, api_factory=DockerAPI, sleep=time.sleep) -> int:
    raw = request_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("immutable request hash mismatch")
    request = json.loads(raw)
    job_id = request["job_id"]
    status_path = request_path.parent / "receipt.json"
    lock = (request_path.parent / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    try:
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous.get("state") in ("succeeded", "failed", "timed_out", "lost"):
                return 0
            if previous.get("started_at"):
                atomic_json(status_path, {**previous, "state": "lost", "finished_at": utcnow(),
                                          "error": "worker interrupted after execution began; side effect unknown"})
                return 1
        snapshot = request["snapshot"]
        execution = snapshot["execution"]
        receipt = {"job_id": job_id, "request_hash": expected_hash, "state": "accepted",
                   "accepted_at": utcnow(), "started_at": None, "finished_at": None}
        atomic_json(status_path, receipt)
        # Delay is owned by this container, not by an app daemon thread.
        sleep(execution["start_delay_seconds"])
        receipt.update(state="running", started_at=utcnow())
        atomic_json(status_path, receipt)
        api = api_factory(execution["timeout_seconds"])
        def deadline(_signum, _frame):
            raise TimeoutError("operation execution deadline exceeded; Docker may still complete a submitted action")
        old_handler = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, execution["timeout_seconds"])
        try:
            if snapshot["spec"]["action"] in IMAGE_ACTIONS:
                output = ImageDeployer(api, snapshot, job_id,
                                           lock_root=request_path.parent.parent / ".target-locks").run()
            else:
                output = api.execute(snapshot["spec"], snapshot["resolved_args"])
            receipt.update(state="succeeded", exit_code=0, output_tail=output)
        except TimeoutError as exc:
            receipt.update(state="timed_out", exit_code=None, error=str(exc),
                           side_effect_unknown=api.side_effect_started)
        except Exception as exc:
            # A deploy that verifiably restored the previous container is a
            # KNOWN outcome even though production was touched.
            uncertain = api.side_effect_started and not getattr(exc, "outcome_known", False)
            receipt.update(state="lost" if uncertain else "failed", exit_code=None,
                           error=str(exc), side_effect_unknown=uncertain)
            if getattr(exc, "output", None):
                receipt["output_tail"] = exc.output
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        receipt["finished_at"] = utcnow()
        limit = execution["max_output_bytes"]
        for field in ("output_tail", "error"):
            if receipt.get(field):
                receipt[field] = receipt[field].encode()[-limit:].decode(errors="replace")
        atomic_json(status_path, receipt)
        return 0 if receipt["state"] == "succeeded" else 1
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), sys.argv[2]))
