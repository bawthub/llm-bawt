"""Local Talk binding views and non-destructive disconnects.

Uses the existing bot_profiles.nextcloud_config transport mapping. Installing or
recovering a remote signing secret is deliberately not implied by account login.
"""
from sqlalchemy import update
from sqlmodel import Session, select

from llm_bawt.runtime_settings import BotProfile, BotProfileStore
from .setup import SetupError, PROVISIONING_REASON


class NextcloudBindings:
    def __init__(self, config, *, store=None, reload=None):
        self.store = store if store is not None else BotProfileStore(config)
        self.reload = reload

    @staticmethod
    def _bot_id(bot_id: str) -> str:
        bot_id = bot_id.strip().lower()
        if bot_id == "mira":
            raise SetupError("protected_bot", "This bot is outside the supported setup scope.", 403)
        return bot_id

    def list(self) -> list[dict]:
        if self.store.engine is None:
            return []
        with Session(self.store.engine) as session:
            rows = session.exec(select(BotProfile.slug, BotProfile.nextcloud_config).where(BotProfile.slug != "mira")).all()
        return [{"bot_id": slug, "room_token": data.get("conversation_token"),
                 "nextcloud_bot_id": data.get("bot_id"), "user_id": data.get("user_id"),
                 "enabled": data.get("enabled", True), "has_signing_secret": bool(data.get("secret")),
                 "state": "mapped_unverified", "inbound_verified": False, "outbound_verified": False,
                 "last_inbound_at": None, "last_outbound_at": None}
                for slug, data in rows if isinstance(data, dict) and data]

    def connect(self, bot_id: str, room_token: str, nextcloud_bot_id: int, user_id: str) -> dict:
        self._bot_id(bot_id)
        # Ordinary Nextcloud account APIs redact the installed bot secret. Never
        # write a binding which cannot validate signatures or return a reply.
        raise SetupError("provisioning_unavailable", PROVISIONING_REASON, 503)

    def disconnect(self, bot_id: str) -> dict:
        bot_id = self._bot_id(bot_id)
        if self.store.engine is None:
            raise SetupError("storage_unavailable", "Bot configuration storage is unavailable.", 503)
        # BotProfileStore.upsert intentionally ignores None on partial updates;
        # clear this column explicitly without rewriting unrelated profile data.
        with Session(self.store.engine) as session:
            result = session.execute(update(BotProfile).where(BotProfile.slug == bot_id).values(nextcloud_config=None))
            if result.rowcount != 1:
                raise SetupError("bot_not_found", "Bot does not exist.", 404)
            session.commit()
        if self.reload:
            self.reload()
        else:
            from llm_bawt.bots import _load_bots_config
            _load_bots_config()
        return {"disconnected": True, "remote_rooms_preserved": True}
