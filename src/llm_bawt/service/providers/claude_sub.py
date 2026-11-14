"""Claude subscription-login adapter — drive `claude setup-token` under a PTY.

The **sanctioned** subscription path: the tenant runs the official Claude Code
CLI (bundled with the Agent SDK) to log in with the user's Claude subscription.
`setup-token` performs the standard authorize-code + PKCE OAuth against the
first-party Claude Code client, then on success:

  * writes the ``claudeAiOauth`` bundle to ``<config_dir>/.credentials.json`` —
    exactly what ``claude_code_bridge`` already consumes + refreshes, and
  * prints a long-lived (1y, ``scope=user:inference``) token.

This is NOT the banned path: we are authenticating the official Claude Code
harness with a subscription (the carve-out Anthropic keeps), not pointing a sub
token at the raw ``/v1/messages`` API from a third-party tool.

Why a PTY: the CLI is an Ink/TTY app — it renders **nothing** to a plain pipe.
We drive it through a pseudo-terminal. The flow is two-step and stateful: the
authorize URL carries a per-session PKCE challenge, so the SAME process must stay
alive between ``start_cli_login`` (capture URL) and ``complete_cli_login``
(inject the pasted code). Live sessions live in a module-level registry keyed by
an opaque session id, reaped on a TTL.

Cross-container placement: ``setup-token`` writes to the config dir resolved from
``CLAUDE_CONFIG_DIR`` (default ``~/.claude``). For a tenant, point env
``CLAUDE_LOGIN_CONFIG_DIR`` at the directory the claude-code bridge mounts as its
``~/.claude`` so the bundle lands where the bridge reads it. When the app and
bridge are separate containers, that directory must be a shared volume.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pty
import re
import select
import signal
import struct
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from ...utils.config import Config
from .base import (
    AUTH_CLI_OAUTH,
    STATUS_CONNECTED,
    CliLoginResult,
    CliLoginStart,
    ConnectionRecord,
    ProviderAdapter,
)

logger = logging.getLogger(__name__)

# How long a started-but-not-completed login session is kept alive.
_SESSION_TTL_S = 600.0
# Bounds for the two blocking phases (seconds).
_URL_WAIT_S = 30.0
_COMPLETE_WAIT_S = 45.0

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[=>()][A-Za-z0-9]?")
_URL_RE = re.compile(r"https://claude\.com/\S*oauth/authorize\S*")
_SUCCESS_RE = re.compile(r"created successfully|login successful|logged in as", re.I)
# CLI-reported exchange failure (e.g. "Login failed: Request failed with status
# code 400" on a bad/expired/mismatched code). Surface it fast — don't sit out
# the full completion timeout.
_FAIL_RE = re.compile(r"login failed[^\r\n]*", re.I)
# Claude long-lived OAuth token, printed on success (fallback to the file bundle).
_TOKEN_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")


class ClaudeLoginError(RuntimeError):
    """Raised when the login CLI can't be located or driven."""


def _target_config_dir() -> Path:
    """Where `setup-token` should write `.credentials.json`.

    Defaults to ``~/.claude``; override with ``CLAUDE_LOGIN_CONFIG_DIR`` (or an
    already-set ``CLAUDE_CONFIG_DIR``) to land the bundle in the bridge's mount.
    """
    override = os.getenv("CLAUDE_LOGIN_CONFIG_DIR") or os.getenv("CLAUDE_CONFIG_DIR")
    return Path(override) if override else (Path.home() / ".claude")


def _claude_binary() -> str:
    """Absolute path to the CLI bundled inside the installed Agent SDK."""
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise ClaudeLoginError(f"claude_agent_sdk not importable: {e}") from e
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if not path.exists():
        raise ClaudeLoginError(f"bundled claude CLI not found at {path}")
    return str(path)


def _clean(raw: bytes) -> str:
    return _ANSI_RE.sub("", raw.decode(errors="replace"))


def _read_bundle(creds_path: Path) -> dict | None:
    try:
        data = json.loads(creds_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    bundle = data.get("claudeAiOauth")
    if isinstance(bundle, dict) and bundle.get("accessToken"):
        return bundle
    return None


class _LoginSession:
    """A live `setup-token` PTY process awaiting the pasted code."""

    def __init__(self, session_id: str, pid: int, fd: int, config_dir: Path):
        self.session_id = session_id
        self.pid = pid
        self.fd = fd
        self.config_dir = config_dir
        self.created = time.monotonic()
        self.buf = bytearray()
        # Completed OAuth exchanges can retry persistence without reusing the code.
        self.record: ConnectionRecord | None = None
        self.cleanup: Callable[[Path], None] | None = None

    def pump(self, timeout: float = 0.3) -> None:
        try:
            r, _, _ = select.select([self.fd], [], [], timeout)
        except OSError:
            return
        if not r:
            return
        try:
            data = os.read(self.fd, 65536)
        except OSError:
            return
        if data:
            self.buf += data

    def text(self) -> str:
        return _clean(bytes(self.buf))

    def close(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass


_SESSIONS: dict[str, _LoginSession] = {}
_LOCK = threading.Lock()


def _reap() -> None:
    now = time.monotonic()
    with _LOCK:
        stale = [sid for sid, s in _SESSIONS.items() if now - s.created > _SESSION_TTL_S]
        expired = [_SESSIONS.pop(sid) for sid in stale]
    for session in expired:
        session.close()
        if session.cleanup is not None:
            session.cleanup(session.config_dir)


class ClaudeSubAdapter(ProviderAdapter):
    id = "claude-sub"
    label = "Claude (subscription login)"
    auth_methods = (AUTH_CLI_OAUTH,)

    def __init__(self, config: Config):
        super().__init__(config)

    # --- subclass hooks (see claude.ClaudeAdapter) ---------------------------
    def _cli_args(self) -> list[str]:
        """argv (after the binary) for the login CLI invocation."""
        return ["setup-token"]

    def _login_config_dir(self) -> Path:
        """CLAUDE_CONFIG_DIR the CLI writes ``.credentials.json`` into."""
        return _target_config_dir()

    def _cleanup_login_dir(self, config_dir: Path) -> None:
        """Called after persistence or a failed exchange. No-op for the shared dir."""

    # --- cli oauth ----------------------------------------------------------
    def start_cli_login(self) -> CliLoginStart:
        _reap()
        binary = _claude_binary()
        config_dir = self._login_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex

        pid, fd = pty.fork()
        if pid == 0:  # child
            try:
                os.environ["CLAUDE_CONFIG_DIR"] = str(config_dir)
                os.environ.setdefault("HOME", str(config_dir.parent))
                # Force the interactive login flow — ignore any inherited creds.
                for var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                    os.environ.pop(var, None)
                os.execv(binary, [binary, *self._cli_args()])
            except Exception:  # noqa: BLE001
                os._exit(127)

        # parent — widen the pty so the authorize URL doesn't line-wrap.
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 60, 1000, 0, 0))
        except OSError:
            pass

        session = _LoginSession(session_id, pid, fd, config_dir)
        session.cleanup = self._cleanup_login_dir
        deadline = time.monotonic() + _URL_WAIT_S
        url: str | None = None
        while time.monotonic() < deadline:
            session.pump(0.3)
            m = _URL_RE.search(session.text())
            if m:
                url = m.group(0).rstrip(").,\"'")
                break
        if not url:
            session.close()
            self._cleanup_login_dir(config_dir)
            raise ClaudeLoginError("timed out waiting for the Claude authorize URL")

        with _LOCK:
            _SESSIONS[session_id] = session
        return CliLoginStart(
            session_id=session_id,
            verification_uri=url,
            instructions="Open the URL, log in and approve, then paste the code shown back here.",
        )

    def complete_cli_login(self, session_id: str, code: str) -> CliLoginResult:
        with _LOCK:
            session = _SESSIONS.get(session_id)
        if session is None:
            return CliLoginResult(status="error", detail="unknown or expired login session")

        if session.record is not None:
            return self._persist_login(session)

        creds_path = session.config_dir / ".credentials.json"
        try:
            os.write(session.fd, (code.strip() + "\r").encode())
        except OSError as e:
            self._drop(session_id)
            self._cleanup_login_dir(session.config_dir)
            return CliLoginResult(status="error", detail=f"failed to submit code: {e}")

        deadline = time.monotonic() + _COMPLETE_WAIT_S
        bundle: dict | None = None
        ok = False
        while time.monotonic() < deadline:
            session.pump(0.3)
            txt = session.text()
            if _SUCCESS_RE.search(txt):
                ok = True
                break
            fail = _FAIL_RE.search(txt)
            if fail:
                detail = fail.group(0).strip()
                self._drop(session_id)
                self._cleanup_login_dir(session.config_dir)
                return CliLoginResult(status="error", detail=detail)
            bundle = _read_bundle(creds_path)
            if bundle is not None:
                ok = True
                break

        # Let the CLI flush the credentials file / final output.
        session.pump(0.5)
        if bundle is None:
            bundle = _read_bundle(creds_path)
        token_match = _TOKEN_RE.search(session.text())
        token = token_match.group(0) if token_match else None

        if not ok and bundle is None and token is None:
            # Log the CLI's screen tail for diagnosis — the paste prompt may not
            # have rendered at all (e.g. CLI too old for the headless code flow).
            logger.warning(
                "cli login for %s timed out with no result; output tail: %r",
                self.id,
                session.text()[-300:],
            )
            self._drop(session_id)
            self._cleanup_login_dir(session.config_dir)
            return CliLoginResult(
                status="error",
                detail="login did not complete — no credentials written (bad code or timeout)",
            )

        try:
            session.record = self._build_record(bundle, token, creds_path)
        except Exception as e:  # noqa: BLE001
            self._drop(session_id)
            self._cleanup_login_dir(session.config_dir)
            return CliLoginResult(status="error", detail=str(e))
        return self._persist_login(session)

    def _persist_login(self, session: _LoginSession) -> CliLoginResult:
        record = session.record
        assert record is not None
        try:
            self.store.save(record)
        except Exception:  # noqa: BLE001 — DB/encryption errors may contain secrets
            logger.warning("Failed to persist Claude login credential")
            # Retain the completed session and bundle for retry/recovery. Never
            # resubmit the already-exchanged OAuth code on a persistence retry.
            return CliLoginResult(
                status="error",
                detail="Could not save Claude credential. Retry completing this login.",
            )
        self._drop(session.session_id)
        self._cleanup_login_dir(session.config_dir)
        logger.info("Persisted Claude login credential")
        return CliLoginResult(status="connected", record=record)

    def _build_record(
        self, bundle: dict | None, token: str | None, creds_path: Path
    ) -> ConnectionRecord:
        """Turn a completed login into the persisted ConnectionRecord."""
        account = None
        expires_at = None
        if bundle:
            account = bundle.get("subscriptionType") or "claude-subscription"
            expires_at = bundle.get("expiresAt")

        record = ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_CLI_OAUTH,
            account=account or "claude-subscription",
            meta={
                "credentials_path": str(creds_path),
                "bundle_written": bundle is not None,
                "has_long_lived_token": token is not None,
                "expires_at": expires_at,
            },
        )
        # The .credentials.json bundle is the bridge's source of truth; we keep
        # the printed long-lived token only as an encrypted fallback (env path).
        if token:
            record.secret = {"oauth_token": token}
        return record

    def disconnect(self) -> bool:
        ok = self.store.delete(self.id)
        # Also strip the bundle so the bridge is actually logged out.
        creds_path = _target_config_dir() / ".credentials.json"
        try:
            if creds_path.exists():
                data = json.loads(creds_path.read_text())
                if data.pop("claudeAiOauth", None) is not None:
                    creds_path.write_text(json.dumps(data, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to clear Claude credentials on disconnect: %s", e)
        return ok

    @staticmethod
    def _drop(session_id: str) -> None:
        with _LOCK:
            session = _SESSIONS.pop(session_id, None)
        if session is not None:
            session.close()
