"""Credential-free, wake-only LAN action (TASK-1734). STANDARD LIBRARY ONLY.

The normal worker stays network-isolated. It runs one bounded, socketless
(Docker/DB/credential-free) helper in the Docker host's network namespace, so
the UDP broadcast originates on the LAN rather than a container bridge.
Targets are literal catalog configuration, never caller arguments.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import re
import signal
import socket
import sys
import time
from urllib.parse import quote, urlencode

ACTION = "wake_on_lan"
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.wake-on-lan"
POLL_SECONDS = 2
OUTPUT_LIMIT = 8192
HELPER_LABEL = "llm-bawt.ops.wake-helper"
JOB_LABEL = "llm-bawt.ops.job"
# A live parent deletes its helper seconds after it exits, so a helper exited
# this long ago has a dead parent. Far beyond any helper's own alarm (<=301s).
ORPHAN_MIN_AGE_SECONDS = 600
ORPHAN_SWEEP_LIMIT = 20
_HELPER_NAME = re.compile(r"/?llm-bawt-ops-wake-([0-9a-f]{24})")
LIFECYCLE_MARGIN_SECONDS = 30
_KEYS = frozenset({
    "action", "mac_address", "broadcast_address", "host_address", "tcp_ports",
    "health_port", "health_path", "health_field", "wait_seconds",
})


class WakeNotReady(RuntimeError):
    """A helper completed with a known readiness failure, not an unknown wake."""

    outcome_known = True


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _KEYS or spec.get("action") != ACTION:
        raise ValueError("wake_on_lan requires exactly its fixed target and readiness fields")
    mac = spec["mac_address"]
    if not isinstance(mac, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", mac):
        raise ValueError("mac_address must be six colon-separated hex bytes")
    octets = bytes.fromhex(mac.replace(":", ""))
    if octets == bytes(6) or octets[0] & 1:
        raise ValueError("mac_address must be a nonzero unicast address")
    for key in ("broadcast_address", "host_address"):
        try:
            value = ipaddress.IPv4Address(spec[key])
        except (ValueError, TypeError, ipaddress.AddressValueError):
            raise ValueError(f"{key} must be a literal IPv4 address") from None
        if not isinstance(spec[key], str) or value.is_multicast or value.is_loopback or value.is_unspecified:
            raise ValueError(f"{key} must be a LAN IPv4 address")
    ports = spec["tcp_ports"]
    if (not isinstance(ports, list) or not 1 <= len(ports) <= 8
            or any(type(p) is not int or not 1 <= p <= 65535 for p in ports)
            or len(set(ports)) != len(ports)):
        raise ValueError("tcp_ports must contain 1..8 distinct ports")
    if type(spec["health_port"]) is not int or not 1 <= spec["health_port"] <= 65535:
        raise ValueError("health_port must be a port number")
    if not isinstance(spec["health_path"], str) or not re.fullmatch(r"/[A-Za-z0-9._/-]{0,127}", spec["health_path"]):
        raise ValueError("health_path must be a fixed HTTP path")
    if not isinstance(spec["health_field"], str) or not re.fullmatch(r"[a-z_]{1,32}", spec["health_field"]):
        raise ValueError("health_field must be a short lowercase JSON field")
    if type(spec["wait_seconds"]) is not int or not 10 <= spec["wait_seconds"] <= 300:
        raise ValueError("wait_seconds must be an integer in 10..300")
    return spec


def magic_packet(mac: str) -> bytes:
    return b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16


def validate_budget(spec: dict, timeout_seconds: int) -> None:
    if timeout_seconds < spec["wait_seconds"] + LIFECYCLE_MARGIN_SECONDS:
        raise ValueError("wake_on_lan timeout must allow wait_seconds plus 30s for helper lifecycle")


def send_packet(spec: dict) -> int:
    sent = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(2)
        packet = magic_packet(spec["mac_address"])
        for _ in range(3):
            try:
                sock.sendto(packet, (spec["broadcast_address"], 9))
                sent += 1
            except OSError:
                pass
    return sent


def probe(spec: dict, deadline: float, *, clock=time.monotonic) -> tuple[dict, bool]:
    ports = {}
    for port in spec["tcp_ports"]:
        ports[str(port)] = False
        remaining = deadline - clock()
        if remaining <= 0:
            continue
        try:
            with socket.create_connection((spec["host_address"], port), timeout=min(2, remaining)):
                ports[str(port)] = True
        except OSError:
            pass
    healthy = False
    remaining = deadline - clock()
    if remaining > 0:
        conn = http.client.HTTPConnection(spec["host_address"], spec["health_port"], timeout=min(2, remaining))
        try:
            conn.request("GET", spec["health_path"], headers={"Connection": "close"})
            response = conn.getresponse()
            raw = response.read(4097)
            if response.status == 200 and len(raw) <= 4096:
                data = json.loads(raw)
                value = data.get(spec["health_field"]) if isinstance(data, dict) else None
                healthy = value is True or (isinstance(value, str) and bool(value.strip()))
        except (OSError, ValueError, http.client.HTTPException):
            pass
        finally:
            conn.close()
    return ports, healthy


def wake(spec: dict, *, clock=time.monotonic, sleep=time.sleep, check=probe, send=send_packet) -> dict:
    validate_spec(spec)
    started = clock()
    deadline = started + spec["wait_seconds"]
    result = {"ready": False, "already_awake": False, "packet_sent": False,
              "tcp_ready": {str(port): False for port in spec["tcp_ports"]}, "http_ready": False}
    try:
        ports, healthy = check(spec, deadline, clock=clock)
        already = all(ports.values()) and healthy
        result.update(ready=already, already_awake=already, tcp_ready=ports, http_ready=healthy)
        if not already and clock() < deadline:
            try:
                result["packet_sent"] = send(spec) > 0
            except OSError:
                pass
            if not result["packet_sent"]:
                result["error"] = "magic packet send failed"
                return {**result, "elapsed_seconds": round(clock() - started, 2)}
            while clock() < deadline:
                ports, healthy = check(spec, deadline, clock=clock)
                result.update(tcp_ready=ports, http_ready=healthy, ready=all(ports.values()) and healthy)
                if result["ready"]:
                    break
                sleep(min(POLL_SECONDS, max(0, deadline - clock())))
    except TimeoutError:
        # SIGALRM interrupts slow-trickle HTTP too. Keep the last known packet
        # and readiness state rather than dying silently and losing evidence.
        result["ready"] = False
    if not result["ready"]:
        result.setdefault("error", "readiness deadline exceeded")
    return {**result, "elapsed_seconds": round(clock() - started, 2)}


def _finished_epoch(value) -> float | None:
    from datetime import datetime
    if not isinstance(value, str) or value.startswith("0001-"):
        return None
    try:
        return datetime.fromisoformat(re.sub(r"(\.\d{6})\d*", r"\1", value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def reap_orphans(api, job_id: str, *, now=None) -> dict:
    """Remove exited helpers left by a parent worker that died (TASK-2161).

    Best effort and bounded; never fails the wake. A container is removed only
    when it carries the helper label and a job label whose hash is its own
    name, is not this job's, is stopped, and stopped ORPHAN_MIN_AGE_SECONDS
    ago. The DELETE is unforced, so Docker refuses one that started since.
    Nothing is started, so no packet can be replayed.
    """
    now = now or time.time
    report = {"removed": [], "skipped": 0}
    try:
        filters = json.dumps({"label": [f"{HELPER_LABEL}=1"], "status": ["exited", "dead"]})
        listed = api.request("GET", "/containers/json?" + urlencode({"all": "1", "filters": filters})) or []
        for item in listed[:ORPHAN_SWEEP_LIMIT]:
            info = api.request("GET", f"/containers/{quote(str(item.get('Id', '')), safe='')}/json") or {}
            labels = (info.get("Config") or {}).get("Labels") or {}
            state = info.get("State") or {}
            owner = labels.get(JOB_LABEL)
            match = _HELPER_NAME.fullmatch(info.get("Name") or "")
            finished = _finished_epoch(state.get("FinishedAt"))
            if (labels.get(HELPER_LABEL) != "1" or not isinstance(owner, str) or owner == job_id
                    or not match or match.group(1) != hashlib.sha256(owner.encode()).hexdigest()[:24]
                    or state.get("Status") not in ("exited", "dead") or state.get("Running")
                    or finished is None or now() - finished < ORPHAN_MIN_AGE_SECONDS):
                report["skipped"] += 1
                continue
            api.request("DELETE", f"/containers/{quote(info['Id'], safe='')}?v=1")
            report["removed"].append({"container_id": info["Id"], "job_id": owner})
    except TimeoutError:
        raise  # the worker's execution deadline (SIGALRM), never a cleanup failure
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report["error"] = type(exc).__name__
    return report


def run_worker(api, snapshot: dict, job_id: str, receipt: dict) -> str:
    """Host-net helper has no mounts, secrets, shell or power-off mechanism.

    Its own alarm bounds lifetime if the parent worker dies. A transport failure
    after start remains unknown; an observed terminal helper result is known.
    """
    if __package__:
        from .sealed_container import demux_streams
    else:  # pragma: no cover - standalone worker image
        from sealed_container import demux_streams
    spec = validate_spec(snapshot["spec"])
    validate_budget(spec, snapshot["execution"]["timeout_seconds"])
    if snapshot.get("resolved_args"):
        raise ValueError("wake_on_lan accepts no caller arguments")
    image = snapshot["execution"]["worker_image"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}|[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("wake helper requires the immutable worker image")
    reaped = reap_orphans(api, job_id)
    if reaped["removed"] or reaped.get("error"):
        receipt["wake_on_lan_orphans"] = reaped
    tag = hashlib.sha256(job_id.encode()).hexdigest()[:24]
    name = f"llm-bawt-ops-wake-{tag}"
    body = {
        "Image": image, "Entrypoint": ["python3", "-I", "/ops/wake_on_lan.py"],
        "Cmd": [json.dumps(spec, sort_keys=True)], "Env": [], "WorkingDir": "/",
        "AttachStdout": True, "AttachStderr": True, "Tty": False,
        "Labels": {"llm-bawt.ops.job": job_id, "llm-bawt.ops.wake-helper": "1",
                   "com.docker.compose.project": "llm-bawt-ops-worker",
                   "com.docker.compose.service": "wake-helper"},
        "HostConfig": {
            "NetworkMode": "host", "Binds": [], "ReadonlyRootfs": True, "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"], "Privileged": False,
            "LogConfig": {"Type": "none", "Config": {}}, "RestartPolicy": {"Name": "no"},
            "Memory": 64 * 1024 * 1024, "PidsLimit": 16,
        },
    }
    container = api.request("POST", "/containers/create?" + urlencode({"name": name}), body=body)["Id"]
    cid = quote(container, safe="")
    try:
        stream = api.attach_stream(container)
        try:
            # After submitting start, losing the response must never claim that
            # no packet went out or automatically repeat the action.
            api.side_effect_started = True
            api.request("POST", f"/containers/{cid}/start")
            raw = stream.read(OUTPUT_LIMIT + 1)
        finally:
            stream.close()
        status = api.request("POST", f"/containers/{cid}/wait").get("StatusCode")
        if len(raw) > OUTPUT_LIMIT:
            raise RuntimeError("wake helper output exceeded limit")
        out, _err = demux_streams(raw)
        result = json.loads(out)
        if (not isinstance(result, dict) or type(result.get("ready")) is not bool
                or type(result.get("packet_sent")) is not bool):
            raise RuntimeError("wake helper returned invalid result")
        receipt["wake_on_lan"] = result
        if status != 0 or not result["ready"]:
            raise WakeNotReady("wake completed without readiness: " + json.dumps(result, sort_keys=True))
        return "wake readiness verified: " + json.dumps(result, sort_keys=True)
    finally:
        original_error = sys.exc_info()[0]
        try:
            api.request("DELETE", f"/containers/{cid}?force=1&v=1")
        except (OSError, RuntimeError):
            receipt["wake_on_lan_cleanup"] = {"removed": False, "container_id": container}
            if original_error is None:
                raise WakeNotReady("wake readiness verified but helper cleanup failed") from None


def main() -> int:
    spec = validate_spec(json.loads(sys.argv[1]))
    # Includes the initial probe, packet transmission and every retry. Also
    # stops slow-trickle HTTP peers, whose individual reads reset socket timeouts.
    def expired(_signum, _frame):
        raise TimeoutError("wake helper deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(spec["wait_seconds"] + 1)
    result = wake(spec)
    signal.alarm(0)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["ready"] else 1


if __name__ == "__main__":  # pragma: no cover - helper image entrypoint
    sys.exit(main())
