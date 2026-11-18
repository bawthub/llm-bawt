"""Encrypted Talk bot signing secrets and the explicit pasted-secret reconnect.

A reconnect stores the pasted secret as a *candidate*, proves it with one signed
outbound message, then commits in a single transaction: the candidate is
consumed only if it is still exactly the row this request wrote (a newer
reconnect or a disconnect replaces or deletes it), the encrypted active record
is written, and the bot profile's mapping is rewritten without the legacy
plaintext ``secret``. Readers prefer the encrypted record and fall back to the
legacy plaintext, so no reader ever sees neither.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone

import httpx
from sqlalchemy import delete, select, update

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.runtime_settings import BotProfile, RuntimeSetting
from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore

from llm_bawt.service.commit_wake import notify_in_transaction

from .setup import SetupError

ACTIVE = "nextcloud-talk-bot:"
CANDIDATE = "nextcloud-talk-bot-candidate:"
BOOTSTRAP = "nextcloud-talk-bot-bootstrap:"
# Non-secret per-binding delivery health, deliberately outside the credential
# rows: plain timestamps, never secret material.
HEALTH = "nextcloud_talk_health:"
MAX_SECRET = 1024
VERIFY_MESSAGE = "BawtHub reconnected this room."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def binding_ref(room_token: str | None, secret: str | None) -> str | None:
    """Identity of one active binding (room + signing secret), never the secret itself."""
    if not room_token or not secret:
        return None
    return hashlib.sha256(f"{room_token}\0{secret}".encode()).hexdigest()[:16]


def row_ref(raw: str | None) -> str | None:
    """Stable reference to one exact stored pending row (hash of ciphertext JSON)."""
    return hashlib.sha256(raw.encode()).hexdigest()[:16] if raw else None


def sign(secret: str, random: str, message: str) -> str:
    return hmac.new(secret.encode(), (random + message).encode(), hashlib.sha256).hexdigest()


class SigningStore(IntegrationStore):
    def active_secret(self, bot_id: str) -> str | None:
        rec = self.load(ACTIVE + bot_id)
        value = rec.secret.get("signing_secret") if rec and rec.status == "connected" else None
        return value if isinstance(value, str) and value else None

    def has_active(self, bot_id: str) -> bool:
        return self.active_secret(bot_id) is not None

    def stage(self, bot_id: str, secret: str, mapping: dict) -> tuple[str, str]:
        """Write the candidate; returns (generation, exact stored row)."""
        generation = secrets.token_urlsafe(16)
        record = ConnectionRecord(provider=CANDIDATE + bot_id, status="pending", auth_method="pasted_secret",
                                  meta={**mapping, "generation": generation},
                                  secret={"signing_secret": secret})
        self.save(record)
        snap = self.snapshot(CANDIDATE + bot_id)
        if not snap.record or snap.record.meta.get("generation") != generation:
            raise SetupError("reconnect_superseded", "A newer reconnect or a disconnect replaced this one. Start again.", 409)
        return generation, snap.raw

    def discard(self, bot_id: str, raw: str) -> None:
        snap = self.snapshot(CANDIDATE + bot_id)
        if snap.exists and snap.raw == raw:
            self.remove(CANDIDATE + bot_id, snap)

    def commit(self, bot_id: str, raw: str, secret: str, mapping: dict, bootstrap_raw: str | None = None) -> None:
        """Atomically promote the verified candidate, versioned on its exact row.

        With ``bootstrap_raw`` the exact pending bootstrap generation is consumed
        in the same transaction, so a newer ``start_bootstrap`` (or a disconnect)
        during verification rolls the whole promotion back.
        """
        record = ConnectionRecord(provider=ACTIVE + bot_id, status="connected", auth_method="pasted_secret",
                                  meta=dict(mapping), secret={"signing_secret": secret})
        profile = {"bot_id": mapping["nextcloud_bot_id"], "conversation_token": mapping["room_token"],
                   "user_id": mapping["user_id"], "actor_id": mapping.get("actor_id"),
                   "enabled": True, "signing": "encrypted"}
        with self._store.engine.begin() as conn:
            # The profile row is the binding's lock: health notes take it first,
            # so they serialize with this promotion and with disconnect.
            # Readers fall back to legacy plaintext only while no encrypted
            # record exists; it is dropped in the same commit that adds one.
            if conn.execute(update(BotProfile.__table__).where(BotProfile.slug == bot_id)
                            .values(nextcloud_config=profile)).rowcount != 1:
                raise SetupError("bot_not_found", "Bot does not exist.", 404)
            consumed = conn.execute(delete(RuntimeSetting.__table__).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(CANDIDATE + bot_id), RuntimeSetting.value_json == raw,
            )).rowcount
            if consumed != 1:
                raise SetupError("reconnect_superseded", "A newer reconnect or a disconnect replaced this one. Start again.", 409)
            if bootstrap_raw is not None and conn.execute(delete(RuntimeSetting.__table__).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key == self._key(BOOTSTRAP + bot_id), RuntimeSetting.value_json == bootstrap_raw,
            )).rowcount != 1:
                raise SetupError("bootstrap_superseded", "A newer setup or a disconnect replaced this one. Start again.", 409)
            self.save_in(conn, record)
            prior = self._read_health(conn, bot_id)
            proven = prior.get("pending_ref") in {row_ref(raw), row_ref(bootstrap_raw)} - {None}
            self._write_health(conn, bot_id, {"outbound_at": _now(),
                                              "inbound_at": prior.get("pending_inbound_at") if proven else None})
            notify_in_transaction(conn, "nextcloud_talk")

    def start_bootstrap(self, bot_id: str, mapping: dict) -> str:
        """Generate and store a fresh signing secret; the caller reveals it exactly once.

        Restarting replaces any earlier pending secret, so only the newest
        generated value can complete.
        """
        secret = secrets.token_hex(32)  # 64 chars; Talk requires 40-128
        self.save(ConnectionRecord(provider=BOOTSTRAP + bot_id, status="pending", auth_method="occ_bootstrap",
                                   meta=dict(mapping), secret={"signing_secret": secret}))
        return secret

    def pending_bootstrap(self, bot_id: str):
        """(secret, mapping, exact row) for the pending bootstrap, or None."""
        snap = self.snapshot(BOOTSTRAP + bot_id)
        rec = snap.record
        value = rec.secret.get("signing_secret") if rec else None
        if not isinstance(value, str) or not value:
            return None
        return value, dict(rec.meta), snap.raw

    def has_bootstrap(self, bot_id: str) -> bool:
        return self.pending_bootstrap(bot_id) is not None

    def clear_in(self, conn, bot_id: str) -> None:
        conn.execute(delete(RuntimeSetting.__table__).where(
            RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
            RuntimeSetting.key.in_([self._key(p + bot_id) for p in (ACTIVE, CANDIDATE, BOOTSTRAP)] + [HEALTH + bot_id]),
        ))

    # -- delivery health -------------------------------------------------
    def health(self, bot_id: str) -> dict:
        value = self._store.get_value("global", "*", HEALTH + bot_id) if self.available else None
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _read_health(conn, bot_id: str) -> dict:
        """Read within the caller's transaction (a separate session could roll it back)."""
        current = conn.execute(select(RuntimeSetting.value_json).where(
            RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
            RuntimeSetting.key == HEALTH + bot_id)).scalar()
        try:
            value = json.loads(current) if current else {}
        except ValueError:
            value = {}
        return value if isinstance(value, dict) else {}

    def _write_health(self, conn, bot_id: str, value: dict) -> None:
        key = HEALTH + bot_id
        conn.execute(delete(RuntimeSetting.__table__).where(
            RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*", RuntimeSetting.key == key))
        conn.execute(RuntimeSetting.__table__.insert().values(
            scope_type="global", scope_id="*", key=key, value_json=json.dumps(value),
            updated_at=datetime.now(timezone.utc)))

    def _active_binding_in(self, conn, bot_id: str, config: dict) -> str | None:
        """binding_ref of the bot's current binding, read inside ``conn``'s transaction."""
        raw = conn.execute(select(RuntimeSetting.value_json).where(
            RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
            RuntimeSetting.key == self._key(ACTIVE + bot_id))).scalar()
        secret = None
        if raw:
            try:
                data = json.loads(raw)
                rec = self._record(self._key(ACTIVE + bot_id), data) if isinstance(data, dict) else None
            except ValueError:
                rec = None
            if rec and rec.status == "connected":
                secret = rec.secret.get("signing_secret")
        secret = secret or config.get("secret")  # legacy plaintext binding
        return binding_ref(config.get("conversation_token"), secret if isinstance(secret, str) else None)

    def note(self, bot_id: str, *, binding: str | None = None, pending_raw: str | None = None, **fields) -> bool:
        """Merge fields into health only for the exact binding (or pending row) that produced them.

        The bot's profile row is locked first, so notes serialize with each
        other, with promotion and with disconnect; a stale operation (older
        binding, replaced pending row, disconnected bot) changes nothing.
        """
        if (binding is None) == (pending_raw is None):
            raise ValueError("note needs exactly one of binding or pending_raw")
        with self._store.engine.begin() as conn:
            row = conn.execute(select(BotProfile.nextcloud_config).where(BotProfile.slug == bot_id)
                               .with_for_update()).first()
            if row is None:
                return False
            config = row[0]
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except ValueError:
                    config = None
            config = config if isinstance(config, dict) else {}
            if binding is not None:
                if not config or self._active_binding_in(conn, bot_id, config) != binding:
                    return False
            else:
                still = conn.execute(select(RuntimeSetting.id).where(
                    RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                    RuntimeSetting.key.in_([self._key(CANDIDATE + bot_id), self._key(BOOTSTRAP + bot_id)]),
                    RuntimeSetting.value_json == pending_raw)).first()
                if still is None:
                    return False
            self._write_health(conn, bot_id, {**self._read_health(conn, bot_id), **fields})
            notify_in_transaction(conn, "nextcloud_talk")
            return True

    def match_pending(self, room_token: str, signature: str, random: str, body: bytes) -> str | None:
        """Bot whose pending (unverified) secret for this room signed the callback, if any.

        Records pending inbound proof; the caller must never start a model turn.
        """
        if not self.available or not room_token:
            return None
        with self._store.engine.connect() as conn:
            rows = conn.execute(select(RuntimeSetting.key, RuntimeSetting.value_json).where(
                RuntimeSetting.scope_type == "global", RuntimeSetting.scope_id == "*",
                RuntimeSetting.key.like(self._key("nextcloud-talk-bot-%")))).all()
        for key, raw in rows:
            for prefix in (CANDIDATE, BOOTSTRAP):
                if not key.startswith(self._key(prefix)):
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    continue
                rec = self._record(key, data) if isinstance(data, dict) else None
                secret = rec.secret.get("signing_secret") if rec else None
                if not secret or rec.meta.get("room_token") != room_token:
                    continue
                expected = hmac.new(secret.encode(), random.encode() + body, hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected, (signature or "").lower()):
                    bot_id = key[len(self._key(prefix)):]
                    self.note(bot_id, pending_raw=raw, pending_inbound_at=_now(), pending_ref=row_ref(raw))
                    return bot_id
        return None


def default_store() -> "SigningStore":
    from llm_bawt.utils.config import Config
    return SigningStore(Config())


def note_health(bot_id: str, room_token: str, secret: str, **fields) -> None:
    """Best-effort health write for the binding (room + secret) an operation used; never raises."""
    try:
        binding = binding_ref(room_token, secret)
        store = default_store()
        if binding and store.available:
            store.note(bot_id, binding=binding, **fields)
    except Exception:  # noqa: BLE001 - health must not break delivery
        import logging
        logging.getLogger(__name__).warning("Could not record Talk health for %s", bot_id)


def match_pending(room_token: str, signature: str, random: str, body: bytes) -> str | None:
    try:
        return default_store().match_pending(room_token, signature, random, body)
    except Exception:  # noqa: BLE001 - treat as no pending match
        return None


def resolve_secret(bot_id: str, legacy: str) -> str:
    """Signing secret for webhook/outbox readers: encrypted record, else legacy."""
    try:
        store = default_store()
        found = store.active_secret(bot_id) if store.available else None
    except Exception:  # noqa: BLE001 - never echo; legacy fallback stays usable
        found = None
    return found or legacy or ""


def verify_outbound(client, instance: str, room_token: str, secret: str) -> None:
    """Prove the secret with one signed bot message; Nextcloud returns 201 only for a valid signature."""
    random = secrets.token_hex(32)
    try:
        response = client.post(
            f"{instance}/ocs/v2.php/apps/spreed/api/v1/bot/{room_token}/message",
            headers={"Content-Type": "application/json", "Accept": "application/json", "OCS-APIRequest": "true",
                     "X-Nextcloud-Talk-Bot-Random": random,
                     "X-Nextcloud-Talk-Bot-Signature": sign(secret, random, VERIFY_MESSAGE)},
            content=json.dumps({"message": VERIFY_MESSAGE}).encode())
    except httpx.HTTPError:
        raise SetupError("nextcloud_unavailable", "Could not reach Nextcloud to verify the secret. Nothing was changed.", 502) from None
    if response.status_code == 201:
        return
    if response.status_code in (401, 403):
        raise SetupError("signing_rejected", "Nextcloud rejected the signing secret for this room. Nothing was changed.", 400)
    if response.status_code == 404:
        raise SetupError("bot_not_in_room", "The bot is not enabled in this room. Enable it in Talk, then try again.", 400)
    raise SetupError("nextcloud_unavailable", "Nextcloud could not verify the secret. Nothing was changed.", 502)


def require_durable_key() -> None:
    if CredentialStore.durable_key_problem():
        raise SetupError("storage_unavailable", "Secure integration storage is unavailable.", 503)
