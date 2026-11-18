"""Local Talk binding views and non-destructive disconnects.

Uses the existing bot_profiles.nextcloud_config transport mapping. Installing or
recovering a remote signing secret is deliberately not implied by account login.
Bots tagged ``private`` (TASK-1329) are outside setup scope: refused and never
listed. Without readable tags every bot is refused (503) and none is listed.
"""
import json
import shlex
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import update

from llm_bawt.service.commit_wake import notify_in_transaction
from sqlmodel import Session, select

from llm_bawt.bot_tags import BehaviorTag, BotTags, TagSnapshot
from llm_bawt.runtime_settings import BotProfile, BotProfileStore
from .setup import SetupError, PROVISIONING_REASON, canonical_instance
from .signing import SigningStore, require_durable_key, verify_outbound


OCC_NOTES = [
    "Run these as the web server user from the Nextcloud directory (for example `sudo -u www-data php occ ...`, "
    "or `docker exec -u www-data <container> php occ ...`).",
    "--feature needs Talk 18 or newer; on Talk 17 drop the --feature flags. Check `occ talk:bot:install --help` "
    "on your server, as options vary by version.",
    "Copy the bot id printed by `talk:bot:list`, then finish setup here. The secret is not shown again.",
]


def public_webhook_url(value: str) -> str:
    try:
        p = urlsplit((value or "").strip())
        if (p.scheme not in ("http", "https") or not p.hostname or p.username is not None
                or p.password is not None or p.query or p.fragment or len(value) > 2048
                or any(c.isspace() or ord(c) < 32 for c in value.strip())):
            raise ValueError
        return urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    except (ValueError, TypeError):
        raise SetupError("invalid_webhook_url", "Use the BawtHub webhook URL Nextcloud can reach, without credentials, query or fragment.") from None


def occ_commands(bot_name: str, secret: str, webhook_url: str, room_token: str) -> list[str]:
    q = shlex.quote
    return [
        f"occ talk:bot:install --feature webhook --feature response {q(bot_name)} {q(secret)} {q(webhook_url)}",
        "occ talk:bot:list",
        f"occ talk:bot:setup <bot-id> {q(room_token)}",
    ]


class NextcloudBindings:
    def __init__(self, config, *, store=None, reload=None, signing=None, client_factory=httpx.Client,
                 known_user=None):
        self.config = config
        self.known_user = known_user
        self.store = store if store is not None else BotProfileStore(config)
        self.reload = reload
        self.signing = signing if signing is not None else SigningStore(config)
        self.client_factory = client_factory

    def _tags(self) -> TagSnapshot:
        # Never BotTags(None): that would resolve the process-wide engine
        # instead of this store's.
        engine = self.store.engine
        return BotTags(engine).snapshot() if engine is not None else TagSnapshot.unavailable()

    def _bot_id(self, bot_id: str) -> str:
        bot_id = bot_id.strip().lower()
        tags = self._tags()
        if not tags.available:
            raise SetupError("policy_unavailable", "Bot tags are unavailable; try again shortly.", 503)
        if tags.has(bot_id, BehaviorTag.PRIVATE):
            raise SetupError("protected_bot", "This bot is outside the supported setup scope.", 403)
        return bot_id

    def list(self) -> list[dict]:
        if self.store.engine is None:
            return []
        # Snapshot first: a bot created after it is unknown, so private and hidden.
        private = self._tags().has
        with Session(self.store.engine) as session:
            rows = session.exec(select(BotProfile.slug, BotProfile.nextcloud_config)).all()
        rows = [(slug, data) for slug, data in rows if not private(slug, BehaviorTag.PRIVATE)]
        return [self._row(slug, data) for slug, data in rows if isinstance(data, dict) and data]

    def _health(self, slug: str) -> dict:
        try:
            return self.signing.health(slug) if self.signing.available else {}
        except Exception:  # noqa: BLE001 - a status read never fails on this
            return {}

    def _row(self, slug: str, data: dict) -> dict:
        health = self._health(slug)
        inbound, outbound = health.get("inbound_at"), health.get("outbound_at")
        failed = health.get("outbound_failed_at")
        outbound_ok = bool(outbound) and not (failed and failed > outbound)
        identity = bool(data.get("user_id") and data.get("actor_id"))
        # Separate facts: an outbound 201 never implies the callback reaches us.
        state = ("outbound_failing" if failed and not outbound_ok else
                 "verified" if inbound and outbound_ok else
                 "awaiting_inbound" if outbound_ok else "mapped_unverified")
        return {"bot_id": slug, "room_token": data.get("conversation_token"),
                "nextcloud_bot_id": data.get("bot_id"), "user_id": data.get("user_id"),
                "actor_id": data.get("actor_id"), "identity_mapped": identity,
                "enabled": data.get("enabled", True),
                "has_signing_secret": bool(data.get("secret")) or self._encrypted(slug),
                "secret_storage": "encrypted" if self._encrypted(slug) else ("legacy" if data.get("secret") else None),
                "bootstrap_pending": self._bootstrap_pending(slug), "state": state,
                "inbound_verified": bool(inbound), "outbound_verified": outbound_ok,
                "last_inbound_at": inbound, "last_outbound_at": outbound,
                "last_outbound_failed_at": failed, "pending_inbound_at": health.get("pending_inbound_at")}

    def connect(self, bot_id: str, room_token: str, nextcloud_bot_id: int, user_id: str) -> dict:
        self._bot_id(bot_id)
        # Ordinary Nextcloud account APIs redact the installed bot secret. Never
        # write a binding which cannot validate signatures or return a reply.
        raise SetupError("provisioning_unavailable", PROVISIONING_REASON, 503)

    def _encrypted(self, slug: str) -> bool:
        try:
            return self.signing.available and self.signing.has_active(slug)
        except Exception:  # noqa: BLE001 - a status read never fails on this
            return False

    def _bootstrap_pending(self, slug: str) -> bool:
        try:
            return self.signing.available and self.signing.has_bootstrap(slug)
        except Exception:  # noqa: BLE001 - a status read never fails on this
            return False

    def _identity(self, user_id: str, actor_id: str) -> tuple[str, str]:
        """Server-side room identity: the Nextcloud user allowed here and the app user it maps to."""
        user_id, actor_id = (user_id or "").strip(), (actor_id or "").strip()
        if not actor_id or len(actor_id) > 128 or "/" in actor_id or any(c.isspace() for c in actor_id):
            raise SetupError("invalid_actor", "Enter the Nextcloud user ID allowed to talk to this bot.")
        if not user_id or not self._user_exists(user_id):
            raise SetupError("unknown_user", "Choose an existing BawtHub user for this room.")
        return user_id, actor_id

    def _user_exists(self, user_id: str) -> bool:
        if self.known_user is not None:
            return bool(self.known_user(user_id))
        from llm_bawt.profiles import EntityProfile, EntityType
        with Session(self.store.engine) as session:
            row = session.exec(select(EntityProfile.entity_id).where(
                EntityProfile.entity_type == EntityType.USER, EntityProfile.entity_id == user_id,
                EntityProfile.is_service == False)).first()  # noqa: E712
        return row is not None

    def reconnect(self, bot_id: str, room_token: str, nextcloud_bot_id: int, user_id: str, secret: str,
                  actor_id: str = "", *, bootstrap_raw: str | None = None) -> dict:
        """Explicit reconnect with a pasted signing secret; active config changes only after verification."""
        bot_id = self._bot_id(bot_id)
        user_id, actor_id = self._identity(user_id, actor_id)
        if not isinstance(secret, str) or not secret.strip() or len(secret) > 1024:
            raise SetupError("invalid_secret", "Paste the bot's signing secret.")
        secret = secret.strip()
        self._require_storage(bot_id)
        instance = canonical_instance(getattr(self.config, "NEXTCLOUD_URL", "") or "")
        mapping = {"room_token": room_token, "nextcloud_bot_id": nextcloud_bot_id, "user_id": user_id,
                   "actor_id": actor_id}
        _, raw = self.signing.stage(bot_id, secret, mapping)
        try:
            with self.client_factory(timeout=20, follow_redirects=False, trust_env=False) as http:
                verify_outbound(http, instance, room_token, secret)
        except BaseException:
            self.signing.discard(bot_id, raw)
            raise
        try:
            self.signing.commit(bot_id, raw, secret, mapping, bootstrap_raw=bootstrap_raw)
        except BaseException:
            self.signing.discard(bot_id, raw)
            raise
        self._reload()
        return {"reconnected": True, "bot_id": bot_id, "room_token": room_token, "outbound_verified": True}

    def _require_storage(self, bot_id: str) -> None:
        if self.store.engine is None or not self.signing.available:
            raise SetupError("storage_unavailable", "Bot configuration storage is unavailable.", 503)
        require_durable_key()
        with Session(self.store.engine) as session:
            if session.exec(select(BotProfile.slug).where(BotProfile.slug == bot_id)).first() is None:
                raise SetupError("bot_not_found", "Bot does not exist.", 404)

    def bootstrap(self, bot_id: str, room_token: str, user_id: str, webhook_url: str, bot_name: str,
                  actor_id: str = "") -> dict:
        """Generate a signing secret and the occ commands that install the bot with it.

        The secret is revealed in this response only (no-store); status,
        listings and errors never carry it. Active config is untouched until
        :meth:`complete_bootstrap` verifies a signed send.
        """
        bot_id = self._bot_id(bot_id)
        webhook_url = public_webhook_url(webhook_url)
        user_id, actor_id = self._identity(user_id, actor_id)
        self._require_storage(bot_id)
        mapping = {"room_token": room_token, "user_id": user_id, "actor_id": actor_id, "webhook_url": webhook_url}
        secret = self.signing.start_bootstrap(bot_id, mapping)
        return {"bot_id": bot_id, "room_token": room_token, "secret": secret, "secret_shown_once": True,
                "commands": occ_commands(bot_name, secret, webhook_url, room_token),
                "notes": OCC_NOTES}

    def complete_bootstrap(self, bot_id: str, nextcloud_bot_id: int) -> dict:
        """Verify the generated secret against the installed bot, then activate it."""
        bot_id = self._bot_id(bot_id)
        self._require_storage(bot_id)
        pending = self.signing.pending_bootstrap(bot_id)
        if pending is None:
            raise SetupError("no_bootstrap", "No pending setup for this bot. Start setup again.", 409)
        secret, mapping, raw = pending
        return self.reconnect(bot_id, mapping["room_token"], nextcloud_bot_id, mapping["user_id"], secret,
                              mapping.get("actor_id", ""), bootstrap_raw=raw)

    def _reload(self) -> None:
        if self.reload:
            self.reload()
        else:
            from llm_bawt.bots import _load_bots_config
            _load_bots_config()

    def disconnect(self, bot_id: str) -> dict:
        bot_id = self._bot_id(bot_id)
        if self.store.engine is None:
            raise SetupError("storage_unavailable", "Bot configuration storage is unavailable.", 503)
        # BotProfileStore.upsert intentionally ignores None on partial updates;
        # clear this column explicitly without rewriting unrelated profile data.
        # One transaction: a reconnect still verifying loses its candidate and
        # cannot commit afterwards.
        with self.store.engine.begin() as conn:
            prior = conn.execute(select(BotProfile.nextcloud_config).where(BotProfile.slug == bot_id)
                                 .with_for_update()).first()
            result = conn.execute(update(BotProfile.__table__).where(BotProfile.slug == bot_id).values(nextcloud_config=None))
            if result.rowcount != 1:
                raise SetupError("bot_not_found", "Bot does not exist.", 404)
            if self.signing.available:
                self.signing.clear_in(conn, bot_id)
            notify_in_transaction(conn, "nextcloud_talk")
        self._reload()
        prior = prior[0] if prior else None
        if isinstance(prior, str):
            try:
                prior = json.loads(prior)
            except ValueError:
                prior = None
        prior = prior if isinstance(prior, dict) else {}
        # Local routing has stopped; the caller reports the remote bot-disable separately.
        return {"disconnected": True, "remote_rooms_preserved": True,
                "room_token": prior.get("conversation_token"), "nextcloud_bot_id": prior.get("bot_id")}
