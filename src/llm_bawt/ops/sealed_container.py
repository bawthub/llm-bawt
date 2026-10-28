"""One hardened, disposable container on a digest-pinned tool image. STANDARD LIBRARY ONLY.

Shared by the Compose renderer (TASK-1518) and the bind-mounted source check
(TASK-1474). Both read an operator's project directory without trusting it, so
both get the same container:

* no network, no Docker socket, a read-only root filesystem, every capability
  dropped, ``no-new-privileges``, memory and PID caps;
* the logging driver ``none``: output is read once, never kept by the daemon;
* exactly one read-only bind, the project directory at the same path, plus
  (optionally) a small ``noexec`` tmpfs at ``/tmp`` for scratch files;
* labels that keep it out of Compose-target selection and mark it as an ops
  job's helper.

Two shapes: :meth:`SealedContainer.run` attaches, starts and waits for one
fixed argv; :meth:`SealedContainer.session` keeps one container alive so a
sequence of fixed-argv ``exec`` calls can share its tmpfs. The container is
removed (with anonymous volumes) in every outcome. Nothing here interprets
output: callers parse it strictly and never echo stderr.
"""
from __future__ import annotations

import os
import re
import struct
from contextlib import contextmanager
from urllib.parse import quote, urlencode

# Same label the dispatcher puts on workers: Compose-target selection skips it.
OPS_JOB_LABEL = "llm-bawt.ops.job"
TMPFS_BYTES = 64 * 1024 * 1024
EMPTY_DIR_BYTES = 1024 * 1024
# A session's keepalive process. The longest a sequence of execs may take; the
# container is removed as soon as the session ends, whatever happens.
SESSION_SECONDS = 600


class SealedRunError(Exception):
    """``kind`` is one of ``image_missing``, ``unavailable``, ``output_too_large``."""

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


def demux_streams(raw: bytes) -> tuple[bytes, bytes]:
    """(stdout, stderr) bytes of Docker's multiplexed non-TTY stream."""
    out, err, i = bytearray(), bytearray(), 0
    while i + 8 <= len(raw):
        stream, size = raw[i], struct.unpack(">I", raw[i + 4:i + 8])[0]
        (out if stream == 1 else err).extend(raw[i + 8:i + 8 + size])
        i += 8 + size
    return bytes(out), bytes(err)


def sealed_body(image: str, *, entrypoint: list[str], cmd: list[str], env: list[str], directory: str,
                labels: dict, role: str, tmpfs: bool = False, empty_dirs: tuple = ()) -> dict:
    """Create body of the hardened container. ``labels`` adds the caller's own
    marker; ``role`` names it as the Compose service it pretends to be.
    ``empty_dirs`` are extra small ``noexec`` tmpfs mounts that start empty, so
    nothing the image or the project ships can appear there (TASK-1476)."""
    host = {
        "NetworkMode": "none", "Binds": [f"{directory}:{directory}:ro"], "ReadonlyRootfs": True,
        "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges:true"], "LogConfig": {"Type": "none", "Config": {}},
        "RestartPolicy": {"Name": "no"}, "AutoRemove": False, "Memory": 256 * 1024 * 1024, "PidsLimit": 64,
        "Privileged": False,
    }
    mounts = {path: f"rw,noexec,nosuid,nodev,size={EMPTY_DIR_BYTES}" for path in empty_dirs}
    if tmpfs:
        mounts["/tmp"] = f"rw,noexec,nosuid,nodev,size={TMPFS_BYTES}"
    if mounts:
        host["Tmpfs"] = mounts
    return {
        "Image": image, "Entrypoint": entrypoint, "Cmd": cmd, "Env": env, "WorkingDir": directory,
        "AttachStdout": True, "AttachStderr": True, "AttachStdin": False, "OpenStdin": False, "Tty": False,
        "NetworkDisabled": True,
        "Labels": {**labels,
                   # Never a Compose target, even if the image carries such labels.
                   "com.docker.compose.project": "llm-bawt-ops-render",
                   "com.docker.compose.service": role},
        "HostConfig": host,
    }


class SealedContainer:
    """``api`` is the worker's stdlib DockerAPI shape: ``request(method, path,
    body=..., raw=...)`` plus ``attach_stream(id)``."""

    def __init__(self, api, image: str, *, role: str, tag: str, labels: dict):
        if not re.fullmatch(r"[a-z0-9]{1,32}", tag):
            raise ValueError("sealed container tag must be short lowercase alphanumerics")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", role):
            raise ValueError("sealed container role must be a short lowercase name")
        self.api = api
        self.image = image
        self.role = role
        self.tag = tag
        self.labels = {**labels, OPS_JOB_LABEL: tag}

    # ── lifecycle ───────────────────────────────────────────────────────
    def _create(self, body: dict) -> str:
        try:
            self.api.request("GET", f"/images/{quote(self.image, safe='')}/json")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                raise SealedRunError("image_missing") from None
            raise SealedRunError("unavailable") from None
        name = f"llm-bawt-ops-{self.role}-{self.tag}-{os.urandom(4).hex()}"
        try:
            return self.api.request("POST", "/containers/create?" + urlencode({"name": name}), body=body)["Id"]
        except (RuntimeError, OSError, KeyError, TypeError):
            raise SealedRunError("unavailable") from None

    def _remove(self, container_id: str) -> None:
        try:
            self.api.request("DELETE", f"/containers/{container_id}?force=1&v=1")
        except Exception:  # noqa: BLE001 - the outcome stands; a leftover is labelled
            pass

    def _body(self, *, entrypoint, cmd, env, directory, tmpfs, empty_dirs=()) -> dict:
        return sealed_body(self.image, entrypoint=entrypoint, cmd=cmd, env=env, directory=directory,
                           labels=self.labels, role=self.role, tmpfs=tmpfs, empty_dirs=empty_dirs)

    # ── one-shot: attach, start, wait ───────────────────────────────────
    def run(self, *, entrypoint: list[str], cmd: list[str], env: list[str], directory: str,
            max_bytes: int, empty_dirs: tuple = ()) -> tuple[int | None, bytes]:
        """``(exit status, raw multiplexed output)``; the output is never longer than ``max_bytes``."""
        container_id = self._create(self._body(entrypoint=entrypoint, cmd=cmd, env=env,
                                               directory=directory, tmpfs=False, empty_dirs=empty_dirs))
        try:
            try:
                stream = self.api.attach_stream(container_id)
                try:
                    self.api.request("POST", f"/containers/{container_id}/start")
                    raw = stream.read(max_bytes + 1)
                finally:
                    stream.close()
                status = (self.api.request("POST", f"/containers/{container_id}/wait") or {}).get("StatusCode")
            except (RuntimeError, OSError):
                raise SealedRunError("unavailable") from None
        finally:
            self._remove(container_id)
        if len(raw) > max_bytes:
            raise SealedRunError("output_too_large")
        return status, raw

    # ── session: one live container, several fixed-argv execs ───────────
    @contextmanager
    def session(self, *, env: list[str], directory: str):
        """Yields ``exec(argv, env=()) -> (exit code, stdout bytes, stderr bytes)``.

        The keepalive is the image's own ``sleep``; every exec runs in the
        project directory with the session's env plus its own additions."""
        container_id = self._create(self._body(entrypoint=["sleep"], cmd=[str(SESSION_SECONDS)], env=env,
                                               directory=directory, tmpfs=True))
        try:
            try:
                self.api.request("POST", f"/containers/{container_id}/start")
            except (RuntimeError, OSError):
                raise SealedRunError("unavailable") from None
            yield lambda argv, extra_env=(): self._exec(container_id, argv, extra_env)
        finally:
            self._remove(container_id)

    def _exec(self, container_id: str, argv: list[str], extra_env) -> tuple[int, bytes, bytes]:
        try:
            exec_id = self.api.request("POST", f"/containers/{container_id}/exec", body={
                "AttachStdout": True, "AttachStderr": True, "AttachStdin": False, "Tty": False,
                "Cmd": list(argv), "Env": list(extra_env)})["Id"]
            raw = self.api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
            state = self.api.request("GET", f"/exec/{exec_id}/json") or {}
        except RuntimeError as exc:
            # The worker's DockerAPI refuses a successful raw stream over its bound.
            too_large = re.search(r"HTTP 2\d\d \(raw stream\)", str(exc))
            raise SealedRunError("output_too_large" if too_large else "unavailable") from None
        except (OSError, KeyError, TypeError):
            raise SealedRunError("unavailable") from None
        code = state.get("ExitCode")
        if state.get("Running") or not isinstance(code, int):
            raise SealedRunError("unavailable")
        out, err = demux_streams(raw if isinstance(raw, (bytes, bytearray)) else b"")
        return code, out, err


__all__ = ["OPS_JOB_LABEL", "SESSION_SECONDS", "SealedContainer", "SealedRunError", "demux_streams",
           "sealed_body"]
