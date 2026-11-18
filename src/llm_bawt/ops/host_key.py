"""Fixed host-key actions over SSH (TASK-2430). STANDARD LIBRARY ONLY.

The ops system otherwise refuses SSH (TASK-639: no target_host, run_as_user or
working_directory). This action relaxes that for exactly three operations:
verify a login, add the bot public key, remove the bot public key.

Boundary:
- The host, port and login user come from an owner-managed host record,
  resolved and pinned at approval; caller arguments carry only its id.
- Every remote command is a module constant. The only value placed in one is
  the bot public key's algorithm and base64 blob, read from the key container
  and validated; its fingerprint is pinned at approval and rechecked here.
- SSH runs inside the container that already holds the key (docker exec), with
  that identity forced: no config file, agent, multiplexing, other default
  keys or non-key authentication.
- Host trust is persisted by the caller in the host record, not in the key
  container's read-only ~/.ssh. A per-job known_hosts file is written to the
  container's /tmp; with pinned host keys checking is strict and a changed key
  is refused. A host with no pinned key is first use: verify may accept it and
  returns the learned keys for the caller to persist; add/remove refuse it.
- ``timeout -k`` bounds the exec inside the container, independent of the ops
  worker's own lifetime.
- Remove succeeds only on the remote script's positive acknowledgement after
  an atomic rewrite. Anything else after the command started is unknown, never
  revocation, and is not replayed.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
from urllib.parse import quote

ACTION = "host_key"
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.host-key"
OPS = ("verify", "add", "remove")
OUTPUT_LIMIT = 8192
CONNECT_TIMEOUT_SECONDS = 10
EXEC_TIMEOUT_SECONDS = 45
KILL_AFTER_SECONDS = 5
LIFECYCLE_MARGIN_SECONDS = 30
KEY_PATH = "/home/bridge/.ssh/id_ed25519"
JOB_ROOT = "/tmp/llm-bawt-host-key"
_KEYS = frozenset({"action", "op", "key_container"})
_CONTAINER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}))*")
_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_PUBKEY = re.compile(r"(ssh-ed25519) ([A-Za-z0-9+/]{68})(?: [A-Za-z0-9@._-]{1,64}){0,8}")
_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")
_HOSTKEY_TYPES = ("ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384",
                  "ecdsa-sha2-nistp521", "rsa-sha2-512", "rsa-sha2-256", "ssh-rsa")
_HOSTKEY = re.compile(r"(ssh-ed25519|ecdsa-sha2-nistp(?:256|384|521)|ssh-rsa) ([A-Za-z0-9+/]{40,1024}={0,2})")


class HostKeyFailed(RuntimeError):
    """A known failure: nothing was changed on the host, or the result is known."""

    outcome_known = True


class HostKeyUnknown(RuntimeError):
    """The remote command may have run; its effect is not known. Never replay."""

    outcome_known = False


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _KEYS or spec.get("action") != ACTION:
        raise ValueError("host_key requires exactly action, op and key_container")
    if spec["op"] not in OPS:
        raise ValueError("host_key op must be verify, add or remove")
    if not isinstance(spec["key_container"], str) or not _CONTAINER.fullmatch(spec["key_container"]):
        raise ValueError("key_container must be a container name")
    return spec


def validate_budget(timeout_seconds: int) -> None:
    # Remove runs the mutation and then one corroborating login attempt.
    if timeout_seconds < 2 * (EXEC_TIMEOUT_SECONDS + KILL_AFTER_SECONDS) + LIFECYCLE_MARGIN_SECONDS:
        raise ValueError("host_key timeout must allow two bounded SSH execs plus 30s")


def validate_host(host: object) -> str:
    if not isinstance(host, str) or not 1 <= len(host) <= 253 or host.startswith("-"):
        raise ValueError("host must be a hostname or literal IP address")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if not _HOST.fullmatch(host):
        raise ValueError("host must be a hostname or literal IP address")
    return host


def validate_target(target: dict) -> dict:
    """The pinned host record an approval carries."""
    keys = {"host_id", "host", "port", "login_user", "key_fingerprint", "known_host_keys"}
    if not isinstance(target, dict) or set(target) != keys:
        raise ValueError("host_key target must carry exactly the pinned host fields")
    if type(target["host_id"]) is not int or target["host_id"] < 1:
        raise ValueError("host_id must be a positive integer")
    validate_host(target["host"])
    if type(target["port"]) is not int or not 1 <= target["port"] <= 65535:
        raise ValueError("port must be 1..65535")
    if not isinstance(target["login_user"], str) or not _USER.fullmatch(target["login_user"]):
        raise ValueError("login_user must be a POSIX user name")
    if not isinstance(target["key_fingerprint"], str) or not _FINGERPRINT.fullmatch(target["key_fingerprint"]):
        raise ValueError("key_fingerprint must be an SHA256 fingerprint")
    validate_host_keys(target["known_host_keys"])
    return target


def validate_host_keys(keys: object) -> list[str]:
    if (not isinstance(keys, list) or len(keys) > 8
            or any(not isinstance(k, str) or not _HOSTKEY.fullmatch(k) for k in keys)):
        raise ValueError("known_host_keys must be up to 8 '<type> <base64>' host keys")
    return keys


def parse_public_key(text: object) -> tuple[str, str]:
    """(algorithm, base64 blob) of the bot key; the comment is never used."""
    line = text.strip() if isinstance(text, str) else ""
    match = _PUBKEY.fullmatch(line)
    if not match:
        raise ValueError("bot public key must be one ssh-ed25519 key")
    algorithm, blob = match.groups()
    raw = base64.b64decode(blob, validate=True)
    name = algorithm.encode()
    if raw[:4] != len(name).to_bytes(4, "big") or raw[4:4 + len(name)] != name:
        raise ValueError("bot public key blob does not match its algorithm")
    return algorithm, blob


def fingerprint(blob: str) -> str:
    digest = hashlib.sha256(base64.b64decode(blob, validate=True)).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


# Remote commands. Each is a fixed POSIX sh script run by the login shell.
# __ALG__ and __BLOB__ are the only substitutions, validated to
# [A-Za-z0-9+/=-]. Keys are matched on each authorization's key-blob FIELD:
# comment and blank lines are skipped, quoted option strings are collapsed
# first (they may contain spaces), and the comment is never read. So a copy
# of the key with other options or comment is found, and a blob that merely
# appears in another key's comment is not.
_KEYBLOB = (
    'function kb(l,  f,n,s){if(l~/^[ \t]*(#|$)/)return "";'
    'gsub(/"([^"\\\\]|\\\\.)*"/,"Q",l);sub(/^[ \t]+/,"",l);n=split(l,f,/[ \t]+/);'
    'if(f[1]~/^(ssh-|ecdsa-|sk-)/)return f[2];return f[3]}'
)
_COUNT = "awk -v b='__BLOB__' '" + _KEYBLOB + "kb($0)==b{n++}END{print n+0}'"

VERIFY = (
    "echo HK_USER=$(id -un); echo HK_GROUPS=$(id -Gn | tr ' ' ','); "
    "if sudo -n true >/dev/null 2>&1; then echo HK_SUDO_N=1; else echo HK_SUDO_N=0; fi; "
    "if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; "
    "then echo HK_DOCKER=1; else echo HK_DOCKER=0; fi; echo HK_DONE=verify"
)

ADD = (
    "set -eu; umask 077; d=\"$HOME/.ssh\"; f=\"$d/authorized_keys\"; mkdir -p \"$d\"; "
    "chmod 700 \"$d\"; [ -e \"$f\" ] || : > \"$f\"; "
    "n=$(" + _COUNT + " \"$f\"); "
    "if [ \"$n\" -gt 0 ]; then echo HK_ADD=present; else "
    # A final line without a newline must end before the new record starts.
    "if [ -s \"$f\" ] && [ -n \"$(tail -c 1 \"$f\")\" ]; then printf '\\n' >> \"$f\"; fi; "
    "printf '%s %s %s\\n' '__ALG__' '__BLOB__' 'bawthub-bots' >> \"$f\"; "
    "[ \"$(" + _COUNT + " \"$f\")\" -gt 0 ]; echo HK_ADD=added; fi; chmod 600 \"$f\"; echo HK_DONE=add"
)

REMOVE = (
    "set -eu; umask 077; f=\"$HOME/.ssh/authorized_keys\"; "
    "if [ ! -e \"$f\" ]; then echo HK_REMOVE=absent; echo HK_DONE=remove; exit 0; fi; "
    "cp -p \"$f\" \"$f.bawthub-bak\"; cmp -s \"$f\" \"$f.bawthub-bak\"; "
    "t=$(mktemp \"$f.XXXXXX\"); "
    "awk -v b='__BLOB__' '" + _KEYBLOB + "kb($0)!=b' \"$f\" > \"$t\"; "
    "[ \"$(" + _COUNT + " \"$t\")\" -eq 0 ]; chmod --reference=\"$f\" \"$t\" 2>/dev/null || chmod 600 \"$t\"; "
    "mv -f \"$t\" \"$f\"; "
    "[ \"$(" + _COUNT + " \"$f\")\" -eq 0 ]; echo HK_REMOVE=removed; echo HK_DONE=remove"
)

REMOTE = {"verify": VERIFY, "add": ADD, "remove": REMOVE}


def remote_command(op: str, algorithm: str, blob: str) -> str:
    if op not in REMOTE:
        raise ValueError("unknown host_key op")
    if not re.fullmatch(r"[a-z0-9-]{1,32}", algorithm) or not re.fullmatch(r"[A-Za-z0-9+/=]{1,1024}", blob):
        raise ValueError("refusing unvalidated key material")
    return REMOTE[op].replace("__ALG__", algorithm).replace("__BLOB__", blob)


_APPROVAL_VERB = {"verify": "Check that the bots can log in to",
                  "add": "Install the bots' SSH key for",
                  "remove": "Remove the bots' SSH key from"}


def approval_text(spec: dict, target: dict, prompt: str | None) -> str:
    """Approval card text from the frozen snapshot, not caller arguments."""
    trust = (f"pinned host key ({len(target['known_host_keys'])})" if target["known_host_keys"]
             else "first use: the host's key is learned and pinned by this check")
    lines = [prompt or "Run a host-key operation?",
             f"{_APPROVAL_VERB[spec['op']]} {target['login_user']}@{target['host']}:{target['port']}",
             f"Bot key: {target['key_fingerprint']}", f"Host trust: {trust}"]
    if spec["op"] == "remove":
        lines.append("Removes only entries for this key (other keys kept, backup made); "
                     "succeeds only when a fresh login with the key is then refused.")
    elif spec["op"] == "add":
        lines.append("Adds one entry if the key is not already authorized; other keys untouched.")
    lines.append("A changed host record or bot key after approval refuses execution.")
    return "\n".join(lines)


def known_hosts_text(target: dict) -> str:
    """Pinned host keys as known_hosts lines for the target's host:port."""
    host = target["host"] if target["port"] == 22 else f"[{target['host']}]:{target['port']}"
    return "".join(f"{host} {key}\n" for key in target["known_host_keys"])


def ssh_argv(target: dict, op: str, command: str, job_dir: str) -> list[str]:
    """Fixed argv. Forces the one identity and fresh public-key authentication."""
    first_use = not target["known_host_keys"]
    if first_use and op != "verify":
        raise ValueError("add/remove need a pinned host key; verify the host first")
    return [
        "timeout", "-k", str(KILL_AFTER_SECONDS), str(EXEC_TIMEOUT_SECONDS),
        "ssh", "-F", "none", "-n", "-T",
        "-i", KEY_PATH,
        "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "AddKeysToAgent=no",
        "-o", "PubkeyAuthentication=yes", "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no", "-o", "HostbasedAuthentication=no",
        "-o", "GSSAPIAuthentication=no", "-o", "BatchMode=yes",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
        "-o", f"ConnectTimeout={CONNECT_TIMEOUT_SECONDS}",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
        "-o", f"UserKnownHostsFile={job_dir}/known_hosts", "-o", "GlobalKnownHostsFile=none",
        "-o", "StrictHostKeyChecking=" + ("accept-new" if first_use else "yes"),
        "-o", "UpdateHostKeys=no", "-o", "LogLevel=ERROR",
        "-p", str(target["port"]), "-l", target["login_user"], "--", target["host"], command,
    ]


def classify_failure(exit_code: object, stderr: str) -> str:
    """Closed reason for a failed SSH exec. Only ``key_refused`` means the host
    answered and refused this key; everything else proves nothing about it."""
    text = stderr.lower()
    if exit_code in (124, 137):
        return "timeout"
    if "host key verification failed" in text or "remote host identification has changed" in text:
        return "host_key_mismatch"
    if "permission denied (publickey" in text:
        return "key_refused"
    if "could not resolve hostname" in text:
        return "dns_failed"
    if "connection refused" in text:
        return "connection_refused"
    if "timed out" in text or "no route to host" in text or "network is unreachable" in text:
        return "unreachable"
    return "ssh_failed"


def parse_markers(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        match = re.fullmatch(r"(HK_[A-Z_]+)=([A-Za-z0-9_,.-]{0,512})", line.strip())
        if match and match.group(1) not in out:
            out[match.group(1)] = match.group(2)
    return out


def verify_fields(markers: dict) -> dict:
    """Closed fields only. ``passwordless_sudo_probe`` reports that ``sudo -n
    true`` ran without a password, not that every sudo command is allowed."""
    if markers.get("HK_DONE") != "verify" or not _USER.fullmatch(markers.get("HK_USER", "")):
        raise ValueError("verify output incomplete")
    groups = [g for g in markers.get("HK_GROUPS", "").split(",") if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", g)]
    return {"user": markers["HK_USER"], "groups": groups[:64],
            "passwordless_sudo_probe": markers.get("HK_SUDO_N") == "1",
            "docker": markers.get("HK_DOCKER") == "1"}


# --- Docker plumbing -------------------------------------------------------

def read_file(api, container: str, path: str, *, demux) -> str:
    code, out, _err = _exec(api, container, ["cat", "--", path], demux=demux)
    if code != 0:
        raise HostKeyFailed(f"cannot read {path} in the key container")
    return out


def read_public_key(api, container: str, *, demux) -> tuple[str, str, str]:
    algorithm, blob = parse_public_key(read_file(api, container, KEY_PATH + ".pub", demux=demux))
    return algorithm, blob, fingerprint(blob)


# Fixed script; the content and path arrive as positional arguments, never as
# shell text.
_WRITE = 'umask 077; printf %s "$1" > "$2"'


def _exec(api, container: str, argv: list[str], *, demux) -> tuple[object, str, str]:
    cid = quote(container, safe="")
    exec_id = api.request("POST", f"/containers/{cid}/exec", body={
        "Cmd": argv, "AttachStdout": True, "AttachStderr": True, "Tty": False,
        "Env": ["LC_ALL=C"], "WorkingDir": "/tmp"})["Id"]
    raw = api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
    state = api.request("GET", f"/exec/{exec_id}/json") or {}
    if len(raw) > OUTPUT_LIMIT * 2:
        raise HostKeyUnknown("ssh output exceeded limit")
    out, err = demux(raw)
    out = out.decode(errors="replace") if isinstance(out, bytes) else out
    err = err.decode(errors="replace") if isinstance(err, bytes) else err
    return state.get("ExitCode"), out[:OUTPUT_LIMIT], err[:OUTPUT_LIMIT]


def _ssh(api, container, target, op, command, job_dir, *, demux):
    code, _o, _e = _exec(api, container, ["sh", "-c", _WRITE, "sh", known_hosts_text(target),
                                          f"{job_dir}/known_hosts"], demux=demux)
    if code != 0:
        raise HostKeyFailed("cannot write the job known_hosts file")
    return _exec(api, container, ssh_argv(target, op, command, job_dir), demux=demux)


def learned_host_keys(api, container: str, job_dir: str, *, demux) -> list[str]:
    keys = []
    for line in read_file(api, container, f"{job_dir}/known_hosts", demux=demux).splitlines():
        parts = line.split()
        if len(parts) >= 3 and _HOSTKEY.fullmatch(f"{parts[1]} {parts[2]}"):
            keys.append(f"{parts[1]} {parts[2]}")
    return list(dict.fromkeys(keys))[:8]


def run_worker(api, snapshot: dict, job_id: str, receipt: dict, *, demux=None) -> str:
    if demux is None:
        if __package__:
            from .sealed_container import demux_streams as demux
        else:  # pragma: no cover - standalone worker image
            from sealed_container import demux_streams as demux
    spec = validate_spec(snapshot["spec"])
    validate_budget(snapshot["execution"]["timeout_seconds"])
    target = validate_target(snapshot["host_key"])
    container, op = spec["key_container"], spec["op"]
    algorithm, blob, actual = read_public_key(api, container, demux=demux)
    if actual != target["key_fingerprint"]:
        raise HostKeyFailed("bot public key changed since approval; nothing was run")
    tag = hashlib.sha256(job_id.encode()).hexdigest()[:24]
    job_dir = f"{JOB_ROOT}-{tag}"
    # mkdir via exec with a fixed argv; the archive PUT needs an existing dir.
    _exec(api, container, ["mkdir", "-m", "700", "-p", job_dir], demux=demux)
    result = {"op": op, "host_id": target["host_id"], "key_fingerprint": actual}
    receipt["host_key"] = result
    try:
        api.side_effect_started = op != "verify"
        code, out, err = _ssh(api, container, target, op, remote_command(op, algorithm, blob), job_dir, demux=demux)
        markers = parse_markers(out)
        if op == "verify":
            if code != 0:
                result["reason"] = classify_failure(code, err)
                result["login_ok"] = False
                raise HostKeyFailed(f"login check failed: {result['reason']}")
            result.update(login_ok=True, **verify_fields(markers))
            if not target["known_host_keys"]:
                result["learned_host_keys"] = learned_host_keys(api, container, job_dir, demux=demux)
            return "host login verified: " + json.dumps(result, sort_keys=True)
        if op == "add":
            if code != 0 or markers.get("HK_DONE") != "add" or markers.get("HK_ADD") not in ("added", "present"):
                reason = classify_failure(code, err) if code != 0 else "no_acknowledgement"
                result["reason"] = reason
                if reason in ("dns_failed", "connection_refused", "host_key_mismatch", "key_refused"):
                    raise HostKeyFailed(f"key not added: {reason}")
                raise HostKeyUnknown(f"add outcome unknown: {reason}")
            result["add"] = markers["HK_ADD"]
            return "bot key present on host: " + json.dumps(result, sort_keys=True)
        # remove
        if code != 0 or markers.get("HK_DONE") != "remove" or markers.get("HK_REMOVE") not in ("removed", "absent"):
            reason = classify_failure(code, err) if code != 0 else "no_acknowledgement"
            result["reason"] = reason
            if reason in ("dns_failed", "connection_refused", "host_key_mismatch"):
                raise HostKeyFailed(f"key not removed: {reason}")
            # key_refused before the script ran means the key was already gone,
            # but that is corroboration, not this job's acknowledgement.
            raise HostKeyUnknown(f"removal not acknowledged: {reason}")
        result["remove"] = markers["HK_REMOVE"]
        # Corroborate with a fresh login; only a key-specific refusal counts.
        code2, _out2, err2 = _ssh(api, container, target, "verify", remote_command("verify", algorithm, blob),
                                  job_dir, demux=demux)
        result["corroboration"] = ("login_refused" if code2 != 0 and classify_failure(code2, err2) == "key_refused"
                                   else "login_still_works" if code2 == 0
                                   else "inconclusive:" + classify_failure(code2, err2))
        if result["corroboration"] == "login_still_works":
            raise HostKeyFailed("removal acknowledged but the key still logs in (another authorization path)")
        if result["corroboration"] != "login_refused":
            # The file entry is gone (acknowledged), but access is unverified.
            raise HostKeyUnknown("removal acknowledged; revocation unverified: " + result["corroboration"])
        return "bot key removed from host: " + json.dumps(result, sort_keys=True)
    finally:
        try:
            _exec(api, container, ["rm", "-rf", "--", job_dir], demux=demux)
        except (OSError, RuntimeError, ValueError, KeyError):
            receipt["host_key_cleanup"] = {"removed": False, "path": job_dir}
