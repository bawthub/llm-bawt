"""Move bootstrap secrets into the root-only Compose secrets volume (TASK-2386).
STANDARD LIBRARY ONLY.

The renderer (``compose_render``) reads bootstrap secrets from the Docker
volume ``llm-bawt-compose-secrets``, one Compose env file per name. Only the
Docker daemon (and so only the ops worker) can write it. This action fills it
from the files the values live in today, without any value leaving Docker:

* ``write``: a sealed helper from the immutable worker image reads each
  declared source (the Compose project's ``.env``, or the master key file),
  first copies those source files byte-for-byte into the root-only recovery
  volume, then publishes ``<NAME>.env`` atomically (mode 0400), reads each one
  back through the same parser and compares.
* ``purge_recovery``: after acceptance, delete every recovery copy.

The helper has no network, a read-only root, every capability dropped except
``DAC_READ_SEARCH`` (the key file is the operator's 0600 file) and binds only
the named source files (read-only, refused by Docker if absent) and the two
volumes. Its output and every error carry names and lengths, never a value.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from urllib.parse import quote, urlencode

ACTION = "compose_secrets_write"
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.compose-secrets-write"
SECRETS_VOLUME = "llm-bawt-compose-secrets"  # == compose_render.COMPOSE_SECRETS_VOLUME
RECOVERY_VOLUME = "llm-bawt-env-recovery"
# == compose_render.COMPOSE_SECRET_NAMES (kept literal: this file also runs alone
# inside the helper, where no sibling module is importable).
SECRET_NAMES = frozenset({"LLM_BAWT_SECRET_KEY", "LLM_BAWT_POSTGRES_PASSWORD", "POSTGRES_PASSWORD", "DATABASE_URL",
                          "LLM_BAWT_REDIS_PASSWORD", "LLM_BAWT_REDIS_URL"})
SOURCE_KINDS = frozenset({"project_env", "key_file"})
MODES = frozenset({"write", "purge_recovery"})
IN_PROJECT_ENV = "/in/project.env"
IN_KEY_FILE = "/in/secret.key"
OUT_DIR = "/out"
RECOVERY_DIR = "/recovery"
OUTPUT_LIMIT = 8192
HELPER_LABEL = "llm-bawt.ops.compose-secrets-helper"
_PROJECT = "com.docker.compose.project"
_WORKING_DIR = "com.docker.compose.project.working_dir"
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._/-]{1,255}")
_PROJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
_KEY_FILE_SUFFIX = "/.config/llm-bawt/secret.key"
# A value the helper can carry with no ambiguity: unquoted in the source
# without whitespace, quotes, ``#``, ``$`` or backslashes (Compose reads those
# literally), or single-quoted (literal in Compose) without a single quote.
_PLAIN = re.compile(r"[^\s'\"#$\\]+")
_LINE = re.compile(r"(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=(.*)")


class SecretsError(RuntimeError):
    """Fixed code plus names; never a value."""

    outcome_known = True

    def __init__(self, code: str, names=()):
        self.code, self.names = code, sorted(set(names))
        super().__init__(code + (": " + ", ".join(self.names) if self.names else ""))


def _clean_path(value) -> bool:
    return (isinstance(value, str) and bool(_SAFE_PATH.fullmatch(value)) and os.path.normpath(value) == value
            and ".." not in value.split("/"))


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("action") != ACTION or spec.get("mode") not in MODES:
        raise ValueError("compose_secrets_write requires action and mode (write or purge_recovery)")
    if spec["mode"] == "purge_recovery":
        if set(spec) != {"action", "mode"}:
            raise ValueError("purge_recovery takes no other fields")
        return spec
    allowed = {"action", "mode", "compose_project", "sources", "key_file"}
    if not {"action", "mode", "compose_project", "sources"} <= set(spec) <= allowed:
        raise ValueError("write requires compose_project and sources (and key_file for a key_file source)")
    if not isinstance(spec["compose_project"], str) or not _PROJECT_NAME.fullmatch(spec["compose_project"]):
        raise ValueError("compose_project is invalid")
    sources = spec["sources"]
    if (not isinstance(sources, dict) or not sources
            or not all(n in SECRET_NAMES and k in SOURCE_KINDS for n, k in sources.items())):
        raise ValueError("sources must map compose secret names to project_env or key_file")
    kinds = list(sources.values())
    if kinds.count("key_file") > 1 or ("key_file" in kinds) != ("key_file" in spec):
        raise ValueError("key_file is required by, and only by, exactly one key_file source")
    if "key_file" in spec and not (_clean_path(spec["key_file"]) and spec["key_file"].endswith(_KEY_FILE_SUFFIX)):
        raise ValueError("key_file must be an absolute path ending in " + _KEY_FILE_SUFFIX)
    return spec


# ── helper side (runs inside the sealed container) ───────────────────────
def parse_env(text: str) -> dict[str, str | None]:
    """Compose env-file lines this helper can read unambiguously. A name whose
    value has any other shape maps to None, so it is refused, never guessed."""
    values: dict[str, str | None] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 3 and value[0] == value[-1] == "'" and "'" not in value[1:-1]:
            values[name] = value[1:-1]  # Compose reads single quotes literally
        elif _PLAIN.fullmatch(value):
            values[name] = value
        else:
            values[name] = None
    return values


def _write_atomic(directory: str, name: str, data: bytes, mode: int = 0o400) -> None:
    tmp = os.path.join(directory, f".{name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, mode)
    os.replace(tmp, os.path.join(directory, name))
    dfd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def collect(sources: dict, *, project_env: str = IN_PROJECT_ENV, key_file: str = IN_KEY_FILE) -> dict[str, str]:
    env = None
    values = {}
    for name, kind in sorted(sources.items()):
        if kind == "key_file":
            with open(key_file, encoding="ascii") as handle:
                value = handle.read().strip()
            values[name] = value if _PLAIN.fullmatch(value) else None
        else:
            if env is None:
                with open(project_env, encoding="utf-8") as handle:
                    env = parse_env(handle.read())
            if name not in env or env[name] == "":
                raise SecretsError("source_missing", [name])
            values[name] = env[name]
    bad = [n for n, v in values.items() if v is None]
    if bad:
        raise SecretsError("source_unsupported", bad)
    return values


def helper_write(spec: dict, *, project_env: str = IN_PROJECT_ENV, key_file: str = IN_KEY_FILE,
                 out_dir: str = OUT_DIR, recovery_dir: str = RECOVERY_DIR, stamp: str | None = None) -> dict:
    values = collect(spec["sources"], project_env=project_env, key_file=key_file)
    # Recovery first: nothing is published until the originals are kept.
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = f"{spec['compose_project']}-{stamp}"
    target = os.path.join(recovery_dir, folder)
    os.mkdir(target, 0o700)
    kinds = set(spec["sources"].values())
    if "project_env" in kinds:
        with open(project_env, "rb") as handle:
            _write_atomic(target, "project.env", handle.read())
    if "key_file" in kinds:
        with open(key_file, "rb") as handle:
            _write_atomic(target, "secret.key", handle.read())
    written = []
    for name, value in sorted(values.items()):
        _write_atomic(out_dir, f"{name}.env", f"{name}='{value}'\n".encode())
        with open(os.path.join(out_dir, f"{name}.env"), encoding="utf-8") as handle:
            if parse_env(handle.read()) != {name: value}:
                raise SecretsError("readback_mismatch", [name])
        written.append({"name": name, "length": len(value)})
    return {"written": written, "recovery": folder}


def helper_purge(*, recovery_dir: str = RECOVERY_DIR) -> dict:
    removed = 0
    for folder in sorted(os.listdir(recovery_dir)):
        path = os.path.join(recovery_dir, folder)
        if os.path.isdir(path) and not os.path.islink(path):
            for entry in os.listdir(path):
                os.unlink(os.path.join(path, entry))
            os.rmdir(path)
        else:
            os.unlink(path)
        removed += 1
    return {"purged": removed}


def main() -> int:
    spec = validate_spec(json.loads(sys.argv[1]))
    try:
        result = helper_write(spec) if spec["mode"] == "write" else helper_purge()
    except SecretsError as exc:
        print(json.dumps({"error": exc.code, "names": exc.names}), flush=True)
        return 1
    except OSError as exc:  # the path names a fixed in-container location, never a value
        print(json.dumps({"error": "io_" + type(exc).__name__, "names": []}), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


# ── worker side ───────────────────────────────────────────────────────────
def project_dir(api, project: str) -> str:
    """The checkout every running container of the Compose project agrees on."""
    rows = api.request("GET", "/containers/json?" + urlencode(
        {"filters": json.dumps({"label": [f"{_PROJECT}={project}"], "status": ["running"]})})) or []
    dirs = {(row.get("Labels") or {}).get(_WORKING_DIR) for row in rows} - {None, ""}
    if len(dirs) != 1:
        raise SecretsError("project_dir_unresolved")
    (found,) = dirs
    if not _clean_path(found) or found == "/":
        raise SecretsError("project_dir_unresolved")
    return found


def helper_body(spec: dict, image: str, job_id: str, directory: str | None) -> dict:
    mounts = [{"Type": "volume", "Source": RECOVERY_VOLUME, "Target": RECOVERY_DIR, "ReadOnly": False}]
    if spec["mode"] == "write":
        mounts.append({"Type": "volume", "Source": SECRETS_VOLUME, "Target": OUT_DIR, "ReadOnly": False})
        kinds = set(spec["sources"].values())
        # Type=bind refuses a missing source instead of creating a directory there.
        if "project_env" in kinds:
            mounts.append({"Type": "bind", "Source": f"{directory}/.env", "Target": IN_PROJECT_ENV,
                           "ReadOnly": True})
        if "key_file" in kinds:
            mounts.append({"Type": "bind", "Source": spec["key_file"], "Target": IN_KEY_FILE, "ReadOnly": True})
    return {
        "Image": image, "Entrypoint": ["python3", "-I", "/ops/compose_secrets.py"],
        "Cmd": [json.dumps(spec, sort_keys=True)], "Env": [], "WorkingDir": "/",
        "AttachStdout": True, "AttachStderr": True, "Tty": False, "NetworkDisabled": True,
        "Labels": {"llm-bawt.ops.job": job_id, HELPER_LABEL: "1",
                   "com.docker.compose.project": "llm-bawt-ops-worker",
                   "com.docker.compose.service": "compose-secrets-helper"},
        "HostConfig": {
            "NetworkMode": "none", "Binds": [], "Mounts": mounts, "ReadonlyRootfs": True,
            "CapDrop": ["ALL"], "CapAdd": ["DAC_READ_SEARCH"] if spec["mode"] == "write" else [],
            "SecurityOpt": ["no-new-privileges:true"], "Privileged": False,
            "LogConfig": {"Type": "none", "Config": {}}, "RestartPolicy": {"Name": "no"},
            "Memory": 64 * 1024 * 1024, "PidsLimit": 16,
        },
    }


def run_worker(api, snapshot: dict, job_id: str, receipt: dict) -> str:
    if __package__:
        from .sealed_container import demux_streams
    else:  # pragma: no cover - standalone worker image
        from sealed_container import demux_streams
    spec = validate_spec(snapshot["spec"])
    if snapshot.get("resolved_args"):
        raise ValueError("compose_secrets_write accepts no caller arguments")
    image = snapshot["execution"]["worker_image"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}|[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("compose secrets helper requires the immutable worker image")
    directory = None
    if spec["mode"] == "write" and "project_env" in spec["sources"].values():
        directory = project_dir(api, spec["compose_project"])
        receipt["compose_secrets_project_dir"] = directory
    name = "llm-bawt-ops-secrets-" + hashlib.sha256(job_id.encode()).hexdigest()[:24]
    created = api.request("POST", "/containers/create?" + urlencode({"name": name}),
                          body=helper_body(spec, image, job_id, directory))["Id"]
    cid = quote(created, safe="")
    try:
        stream = api.attach_stream(created)
        try:
            api.side_effect_started = True
            api.request("POST", f"/containers/{cid}/start")
            raw = stream.read(OUTPUT_LIMIT + 1)
        finally:
            stream.close()
        status = api.request("POST", f"/containers/{cid}/wait").get("StatusCode")
        if len(raw) > OUTPUT_LIMIT:
            raise RuntimeError("compose secrets helper output exceeded limit")
        out, _err = demux_streams(raw)  # stderr is never echoed
        try:
            result = json.loads(out)
        except ValueError:
            raise RuntimeError("compose secrets helper returned invalid result") from None
        if not isinstance(result, dict):
            raise RuntimeError("compose secrets helper returned invalid result")
        receipt["compose_secrets"] = result
        if status != 0 or "error" in result:
            raise SecretsError(str(result.get("error") or "helper_failed"), result.get("names") or [])
        return "compose secrets " + spec["mode"] + ": " + json.dumps(result, sort_keys=True)
    finally:
        try:
            api.request("DELETE", f"/containers/{cid}?force=1&v=1")
        except (OSError, RuntimeError):
            receipt["compose_secrets_cleanup"] = {"removed": False, "container_id": created}


if __name__ == "__main__":  # pragma: no cover - helper image entrypoint
    sys.exit(main())
