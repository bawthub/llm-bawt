from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
except ImportError:
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device identity helpers
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padded = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(padded)


_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def _public_key_raw(pub_pem: str) -> bytes:
    """Extract the raw 32-byte Ed25519 public key from a PEM."""
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    key = load_pem_public_key(pub_pem.encode())
    spki = key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    prefix_len = len(_ED25519_SPKI_PREFIX)
    if len(spki) == prefix_len + 32 and spki[:prefix_len] == _ED25519_SPKI_PREFIX:
        return spki[prefix_len:]
    return spki


def _derive_device_id(pub_pem: str) -> str:
    """SHA-256 hex of the raw public key bytes — matches OpenClaw's deriveDeviceIdFromPublicKey."""
    return hashlib.sha256(_public_key_raw(pub_pem)).hexdigest()


def _sign_payload(priv_pem: str, payload: str) -> str:
    """Ed25519 sign *payload* and return base64url signature."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key = load_pem_private_key(priv_pem.encode(), password=None)
    sig = key.sign(payload.encode())
    return _b64url_encode(sig)


def _build_device_auth_payload_v3(
    *,
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str | None,
    nonce: str,
    platform: str,
    device_family: str,
) -> str:
    """Build the v3 payload string that must be signed."""
    return "|".join([
        "v3",
        device_id,
        client_id,
        client_mode,
        role,
        ",".join(scopes),
        str(signed_at_ms),
        token or "",
        nonce,
        platform or "",
        device_family or "",
    ])


# ---------------------------------------------------------------------------
# Persistent device identity store
# ---------------------------------------------------------------------------

_DEFAULT_IDENTITY_PATH = os.path.expanduser("~/.config/llm-bawt/openclaw-device.json")


class _PairingPendingError(Exception):
    """Raised when the gateway returns NOT_PAIRED — the bridge should retry."""
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"pairing pending: {request_id}")


def _load_or_create_identity(path: str | None = None) -> dict:
    """Load or generate an Ed25519 device identity.

    Returns dict with keys: deviceId, publicKeyPem, privateKeyPem, deviceToken (may be None).
    """
    if Ed25519PrivateKey is None:
        raise RuntimeError("cryptography package required for device identity — pip install cryptography")

    fpath = Path(path or _DEFAULT_IDENTITY_PATH)

    if fpath.exists():
        try:
            stored = json.loads(fpath.read_text())
            if (
                stored.get("version") == 1
                and isinstance(stored.get("publicKeyPem"), str)
                and isinstance(stored.get("privateKeyPem"), str)
            ):
                # Re-derive device ID to be safe
                device_id = _derive_device_id(stored["publicKeyPem"])
                return {
                    "deviceId": device_id,
                    "publicKeyPem": stored["publicKeyPem"],
                    "privateKeyPem": stored["privateKeyPem"],
                    "deviceToken": stored.get("deviceToken"),
                }
        except Exception:
            logger.warning("Corrupt device identity at %s, regenerating", fpath)

    # Generate new identity
    priv_key = Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()
    pub_pem = pub_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    priv_pem = priv_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    device_id = _derive_device_id(pub_pem)

    fpath.parent.mkdir(parents=True, exist_ok=True)
    stored = {
        "version": 1,
        "deviceId": device_id,
        "publicKeyPem": pub_pem,
        "privateKeyPem": priv_pem,
        "deviceToken": None,
        "createdAtMs": int(time.time() * 1000),
    }
    fpath.write_text(json.dumps(stored, indent=2) + "\n")
    fpath.chmod(0o600)
    logger.info("Generated new device identity: %s (stored at %s)", device_id[:16] + "…", fpath)
    return {
        "deviceId": device_id,
        "publicKeyPem": pub_pem,
        "privateKeyPem": priv_pem,
        "deviceToken": None,
    }


def _save_device_token(token: str, path: str | None = None) -> None:
    """Persist the device token after a successful pairing."""
    fpath = Path(path or _DEFAULT_IDENTITY_PATH)
    if not fpath.exists():
        return
    stored = json.loads(fpath.read_text())
    stored["deviceToken"] = token
    fpath.write_text(json.dumps(stored, indent=2) + "\n")
    fpath.chmod(0o600)
    logger.info("Device token saved to %s", fpath)


@dataclass
class OpenClawWsConfig:
    url: str = ""
    token: str = ""
    session_keys: list[str] = field(default_factory=list)
    reconnect_max_delay: int = 60
    device_identity_path: str = ""


class OpenClawWsClient:
    def __init__(self, config: OpenClawWsConfig) -> None:
        self._config = config
        self._ws = None
        self._connected = False
        self._subscribed_sessions: set[str] = set(config.session_keys)
        self._event_callback: Callable[[dict], Awaitable[None]] | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._closing = False
        self._stream_seq = 0
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._run_queues: dict[str, asyncio.Queue[dict | None]] = {}
        # Per-session cancel events — set by chat.abort to unblock send_and_stream
        self._session_cancel_events: dict[str, asyncio.Event] = {}
        # Set whenever the WS is connected; cleared on disconnect.  Used by
        # _request() to briefly wait through transient gateway restarts
        # (close code 1012) instead of failing the user's request instantly.
        self._connected_event: asyncio.Event = asyncio.Event()
        # How long _request() will wait for the WS to (re)connect before
        # raising "OpenClaw WS not connected".  The gateway typically comes
        # back within ~5–15s after a service-restart close.
        self._connect_wait_timeout: float = float(
            os.environ.get("OPENCLAW_BRIDGE_CONNECT_WAIT_S", "20")
        )
        # Device identity (loaded lazily on first connect)
        self._identity: dict | None = None
        self._identity_path: str = config.device_identity_path or ""

    async def connect(self) -> None:
        if not self._config.url:
            logger.warning("OpenClaw WS URL not configured, skipping connect")
            return
        self._closing = False
        try:
            await self._do_connect()
        except _PairingPendingError:
            # Pairing required — start reconnect loop that polls for approval
            self._schedule_reconnect()

    async def _do_connect(self) -> None:
        if websockets is None:
            logger.error("websockets package not installed. Install with: pip install websockets>=13.0")
            return

        try:
            self._ws = await websockets.connect(
                self._config.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )

            raw_first = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            first = json.loads(raw_first) if isinstance(raw_first, str) else json.loads(raw_first.decode())
            nonce = (((first or {}).get("payload") or {}).get("nonce") if first.get("event") == "connect.challenge" else None)
            if not nonce:
                raise RuntimeError("gateway connect challenge missing nonce")

            # --- Device identity ---
            if self._identity is None:
                try:
                    id_path = self._identity_path or None
                    self._identity = _load_or_create_identity(id_path)
                    logger.info("Device identity loaded: %s…", self._identity["deviceId"][:16])
                except Exception as exc:
                    logger.warning("Device identity unavailable (%s), connecting without it", exc)

            identity = self._identity
            client_id = "gateway-client"
            client_mode = "backend"
            client_platform = "linux"
            role = "operator"
            scopes = ["operator.read", "operator.write", "operator.admin"]
            caps = ["tool-events"]

            # Build auth block
            auth: dict = {}
            if self._config.token:
                auth["token"] = self._config.token
            if identity and identity.get("deviceToken"):
                auth["deviceToken"] = identity["deviceToken"]

            # Build device identity block (signed)
            device_block: dict | None = None
            if identity:
                signed_at_ms = int(time.time() * 1000)
                payload_str = _build_device_auth_payload_v3(
                    device_id=identity["deviceId"],
                    client_id=client_id,
                    client_mode=client_mode,
                    role=role,
                    scopes=scopes,
                    signed_at_ms=signed_at_ms,
                    token=self._config.token or None,
                    nonce=nonce,
                    platform=client_platform,
                    device_family="",
                )
                signature = _sign_payload(identity["privateKeyPem"], payload_str)
                device_block = {
                    "id": identity["deviceId"],
                    "publicKey": _b64url_encode(_public_key_raw(identity["publicKeyPem"])),
                    "signature": signature,
                    "signedAt": signed_at_ms,
                    "nonce": nonce,
                }

            req_id = uuid.uuid4().hex
            connect_req = {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": client_id,
                        "version": "0.1.0",
                        "platform": client_platform,
                        "mode": client_mode,
                    },
                    "role": role,
                    "caps": caps,
                    "scopes": scopes,
                    "auth": auth,
                },
            }
            if device_block:
                connect_req["params"]["device"] = device_block

            await self._ws.send(json.dumps(connect_req))

            raw_res = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            res = json.loads(raw_res) if isinstance(raw_res, str) else json.loads(raw_res.decode())
            if res.get("type") != "res" or res.get("id") != req_id or not res.get("ok"):
                err = ((res or {}).get("error") or {}).get("message") or str(res)
                error_code = ((res or {}).get("error") or {}).get("code", "")
                details = ((res or {}).get("error") or {}).get("details") or {}

                # Device not yet paired — log approval instructions and poll
                if error_code == "NOT_PAIRED" and details.get("requestId"):
                    request_id = details["requestId"]
                    logger.warning(
                        "⏳ Device pairing required! Approve on the gateway host:\n"
                        "   openclaw devices approve %s\n"
                        "   The bridge will retry automatically after approval.",
                        request_id,
                    )
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                    # Set a flag so reconnect_with_backoff knows to retry
                    raise _PairingPendingError(request_id)

                # If device token was rejected, clear it and retry without it
                if identity and identity.get("deviceToken") and "device" in err.lower():
                    logger.warning("Device token rejected (%s), clearing and will re-pair", err)
                    identity["deviceToken"] = None
                    _save_device_token("", self._identity_path or None)

                raise RuntimeError(f"gateway connect failed: {err}")

            # Check if we got a device token in the response (auto-pair for local)
            res_payload = res.get("payload") or {}
            if res_payload.get("deviceToken") and identity:
                identity["deviceToken"] = res_payload["deviceToken"]
                _save_device_token(res_payload["deviceToken"], self._identity_path or None)
                logger.info("Device token received from gateway (auto-paired)")

            self._connected = True
            self._connected_event.set()
            logger.info("OpenClaw WS connected to %s (caps=%s)", self._config.url, ",".join(caps))
            self._receive_task = asyncio.create_task(self._receive_loop())

            # --- Handle pairing if needed ---
            # If we have a device identity but no device token, and the
            # connect response includes one, save it for future use.
            # (No need to start a pairing task - if we got here, we're paired.)

        except _PairingPendingError:
            self._connected = False
            self._connected_event.clear()
            # Re-raise so callers (reconnect loop) can use shorter retry interval
            raise
        except Exception as e:
            self._connected = False
            self._connected_event.clear()
            logger.warning("OpenClaw WS connect failed: %s", e)
            if not self._closing:
                self._schedule_reconnect()

    async def _receive_loop(self) -> None:
        try:
            async for raw_msg in self._ws:
                if self._closing:
                    break
                self._stream_seq += 1
                try:
                    data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.debug("OpenClaw WS non-JSON message")
                    continue

                if data.get("type") == "res":
                    req_id = data.get("id", "")
                    fut = self._pending_requests.pop(req_id, None)
                    if fut and not fut.done():
                        fut.set_result(data)
                    continue

                # Handle device pairing resolution events
                if data.get("event") == "device.pair.resolved":
                    self._handle_pair_resolved(data)

                # Route to per-run listeners if this is an event with a runId
                if data.get("type") == "event":
                    payload = data.get("payload") or {}
                    run_id = payload.get("runId") or payload.get("run_id")
                    if run_id and run_id in self._run_queues:
                        try:
                            self._run_queues[run_id].put_nowait(data)
                        except asyncio.QueueFull:
                            logger.warning("Run queue full for %s, dropping event", run_id)

                if self._event_callback:
                    try:
                        await self._event_callback(data)
                    except Exception:
                        logger.exception("Error in WS event callback")

        except Exception as e:
            if not self._closing:
                logger.warning("OpenClaw WS receive loop error: %s", e)
        finally:
            self._connected = False
            self._connected_event.clear()
            if not self._closing:
                self._schedule_reconnect()

    def _handle_pair_resolved(self, data: dict) -> None:
        """Process a device.pair.resolved broadcast from the gateway."""
        payload = data.get("payload") or {}
        status = payload.get("status")
        device_token = payload.get("deviceToken")
        device_id = payload.get("deviceId") or (payload.get("device") or {}).get("deviceId")

        if not self._identity:
            return

        # Only process events for our device
        if device_id and device_id != self._identity.get("deviceId"):
            return

        if status == "approved" and device_token:
            self._identity["deviceToken"] = device_token
            _save_device_token(device_token, self._identity_path or None)
            logger.info("✅ Device pairing approved — token saved")
        elif status == "rejected":
            logger.warning("❌ Device pairing rejected by operator")

    def _schedule_reconnect(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self) -> None:
        base_delay = 1.0
        attempt = 0
        pairing_pending = False
        while not self._closing:
            if pairing_pending:
                # During pairing, poll every 10s (operator needs time to approve)
                delay = 10.0
            else:
                delay = min(base_delay * (2 ** attempt), self._config.reconnect_max_delay)
            jitter = delay * 0.3 * (2 * random.random() - 1)
            actual_delay = max(0.1, delay + jitter)
            if pairing_pending:
                logger.info("OpenClaw WS retrying connect in %.0fs (waiting for device approval)", actual_delay)
            else:
                logger.info("OpenClaw WS reconnecting in %.1fs (attempt %d)", actual_delay, attempt + 1)
            await asyncio.sleep(actual_delay)
            if self._closing:
                break
            try:
                await self._do_connect()
                if self._connected:
                    if pairing_pending:
                        logger.info("✅ Device pairing approved — connected successfully!")
                    else:
                        logger.info("OpenClaw WS reconnected after %d attempts", attempt + 1)
                    return
                pairing_pending = False
            except _PairingPendingError:
                pairing_pending = True
                # Don't increment attempt — keep retrying at pairing interval
                continue
            except Exception as e:
                pairing_pending = False
                logger.warning("OpenClaw WS reconnect attempt %d failed: %s", attempt + 1, e)
            attempt += 1

    async def disconnect(self) -> None:
        self._closing = True
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False
        self._connected_event.clear()
        logger.info("OpenClaw WS disconnected")

    async def _request(self, method: str, params: dict) -> dict:
        # If the WS is currently down (e.g. gateway service-restart 1012),
        # wait briefly for the reconnect loop to bring it back instead of
        # failing the caller instantly.  This smooths over routine gateway
        # restarts that take ~5–15s to recover.
        if (not self._ws or not self._connected) and not self._closing:
            wait_s = max(0.0, self._connect_wait_timeout)
            if wait_s > 0:
                logger.info(
                    "OpenClaw WS not connected; waiting up to %.1fs for reconnect (method=%s)",
                    wait_s, method,
                )
                try:
                    await asyncio.wait_for(self._connected_event.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
        if not self._ws or not self._connected:
            raise RuntimeError("OpenClaw WS not connected")

        req_id = uuid.uuid4().hex
        msg = {"type": "req", "id": req_id, "method": method, "params": params}

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending_requests[req_id] = fut

        await self._ws.send(json.dumps(msg))
        try:
            res = await asyncio.wait_for(fut, timeout=30.0)
            if not res.get("ok"):
                err = ((res or {}).get("error") or {}).get("message") or str(res)
                raise RuntimeError(err)
            return res
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise RuntimeError(f"Timed out waiting for gateway response: {method}")

    async def send_user_message(self, session_key: str, text: str, attachments: list | None = None) -> str:
        params: dict = {
            "sessionKey": session_key,
            "message": text,
            "idempotencyKey": f"idem_{uuid.uuid4().hex}",
        }
        if attachments:
            params["attachments"] = attachments

        res = await self._request("chat.send", params)
        payload = res.get("payload") or {}
        run_id = str(payload.get("runId") or payload.get("run_id") or params["idempotencyKey"])
        return run_id

    async def send_and_stream(
        self,
        session_key: str,
        text: str,
        *,
        attachments: list | None = None,
        timeout: float | None = None,
        on_run_started: Callable[[str], None] | None = None,
    ) -> AsyncIterator[dict]:
        """Send a message via chat.send and yield raw WS events for the resulting run.

        Events are yielded until a lifecycle end/error event is received,
        WS disconnects, or optional timeout. The caller gets the full stream
        of agent events (assistant deltas, tool events, lifecycle, errors).
        """
        run_id = await self.send_user_message(session_key, text, attachments=attachments)
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=2000)
        self._run_queues[run_id] = queue

        # Ensure a cancel event exists for this session so cancel_session()
        # can signal it.  We grab the reference once; cancel_session() sets
        # the same Event object.
        if session_key not in self._session_cancel_events:
            self._session_cancel_events[session_key] = asyncio.Event()
        cancel_event = self._session_cancel_events[session_key]

        try:
            # Expose the exact acknowledged run before waiting for its first
            # event, so callers can abort a silent/long-running gateway turn.
            if on_run_started is not None:
                on_run_started(run_id)
            deadline = (asyncio.get_event_loop().time() + timeout) if timeout else None
            while True:
                # Check if session was cancelled (chat.abort)
                if cancel_event and cancel_event.is_set():
                    logger.info("send_and_stream cancelled via chat.abort for run %s session=%s", run_id, session_key)
                    return

                if deadline is not None:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        logger.warning("send_and_stream timeout for run %s", run_id)
                        return
                    wait = min(remaining, 5.0)
                else:
                    wait = 5.0

                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=wait)
                except asyncio.TimeoutError:
                    if not self._connected:
                        logger.warning("WS disconnected during send_and_stream for run %s", run_id)
                        return
                    continue

                if raw is None:
                    return

                yield raw

                # Check if this is a lifecycle end/error -> run done
                payload = raw.get("payload") or {}
                if raw.get("type") == "event":
                    if raw.get("event") == "agent":
                        stream = payload.get("stream")
                        data = payload.get("data") or {}
                        if stream == "lifecycle" and data.get("phase") in ("end", "error"):
                            return
                        if stream == "error":
                            return
                    if raw.get("event") == "chat":
                        state = str(payload.get("state") or "").lower()
                        if state == "final":
                            return
        finally:
            self._run_queues.pop(run_id, None)

    async def get_chat_history(self, session_key: str, *, limit: int = 50) -> list[dict]:
        """Fetch chat history for a session via the gateway."""
        res = await self._request("chat.history", {"sessionKey": session_key, "limit": limit})
        payload = res.get("payload") or {}
        return payload.get("messages") or payload.get("history") or []

    def on_event(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._event_callback = callback

    def cancel_session(self, session_key: str) -> None:
        """Signal any active send_and_stream for this session to stop.

        Called when chat.abort is received so the bridge releases the
        session lock and allows the next send to proceed.
        """
        event = self._session_cancel_events.get(session_key)
        if event is None:
            event = asyncio.Event()
            self._session_cancel_events[session_key] = event
        event.set()
        logger.info("Session cancel signalled: %s", session_key)

    def clear_session_cancel(self, session_key: str) -> None:
        """Reset the cancel event for a session before a new send."""
        event = self._session_cancel_events.get(session_key)
        if event is not None:
            event.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscribed_sessions(self) -> set[str]:
        return set(self._subscribed_sessions)

    @property
    def stream_seq(self) -> int:
        return self._stream_seq
