"""Client for Nextcloud Talk provisioner service.

The provisioner URL and bearer token come from the encrypted
``integration_connection:talk-provisioner`` record (the Integrations
credentials form), resolved on every :func:`get_provisioner_client` call
(TASK-1476). The legacy env settings apply only while no record exists.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from llm_bawt.service.connection_secrets import ConnectionUnavailable, resolve_integration

log = logging.getLogger(__name__)

INTEGRATION_ID = "talk-provisioner"
# Deprecated settings, read in this order while no record exists.
_LEGACY_URL_VARS = ("LLM_BAWT_TALK_PROVISIONER_URL", "TALK_PROVISIONER_URL")
_LEGACY_TOKEN_VARS = ("LLM_BAWT_TALK_PROVISIONER_TOKEN", "TALK_PROVISIONER_TOKEN")


class TalkProvisionerUnavailable(ValueError):
    """A stored connection exists (or the store is down) but can't be used.

    ``code`` is ``not_connected``, ``unreadable`` or ``store_unavailable``.
    A ``ValueError`` so the admin route reports it as a 400; the message
    never carries a value.
    """

    def __init__(self, code: str):
        super().__init__(f"Talk provisioner connection unavailable ({code}); reconnect it under Integrations")
        self.code = code


@dataclass
class ProvisionResult:
    """Result from provisioning a Nextcloud Talk room and bot."""
    nextcloud_base_url: str
    room_token: str
    room_url: str
    bot_id: int
    bot_name: str
    bot_secret: str
    webhook_url: str

    def __repr__(self):
        """Redact secret in repr."""
        return (
            f"ProvisionResult("
            f"bot_id={self.bot_id}, "
            f"bot_name='{self.bot_name}', "
            f"room_token='{self.room_token}', "
            f"nextcloud_base_url='{self.nextcloud_base_url}', "
            f"bot_secret='***REDACTED***')"
        )


class ProvisionerClient:
    """Client for the Nextcloud Talk provisioner service."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = 30.0,
    ):
        if not token:
            raise ValueError("A Talk provisioner bearer token is required")

        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout

    def _get_headers(self) -> dict:
        """Get request headers with auth."""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }

    async def provision_talk_room_and_bot(
        self,
        room_name: str,
        bot_name: str,
        webhook_url: Optional[str] = None,
        owner_user_id: str = "user",
        room_description: Optional[str] = None,
        bot_description: Optional[str] = None,
    ) -> ProvisionResult:
        """
        Provision a Nextcloud Talk room with a bot.

        Args:
            room_name: Name for the Talk room
            bot_name: Name for the bot
            webhook_url: Webhook URL for the bot (optional - provisioner may have default)
            owner_user_id: Nextcloud user ID who will own the room
            room_description: Optional room description
            bot_description: Optional bot description

        Returns:
            ProvisionResult with room and bot details

        Raises:
            httpx.HTTPError: If provisioning fails
            ValueError: If response is invalid
        """
        url = f"{self.base_url}/provision/nextcloud-talk"

        payload = {
            "roomName": room_name,
            "botName": bot_name,
            "ownerUserId": owner_user_id,
        }

        if webhook_url:
            payload["webhookUrl"] = webhook_url

        if room_description:
            payload["roomDescription"] = room_description
        if bot_description:
            payload["botDescription"] = bot_description

        log.info(f"Provisioning Nextcloud Talk room: {room_name}")
        log.debug(f"Provisioner URL: {url}")
        # DO NOT log payload - it will contain the token in response

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                data = response.json()

                # Validate response
                required_fields = [
                    'nextcloudBaseUrl',
                    'roomToken',
                    'roomUrl',
                    'botId',
                    'botName',
                    'botSecret',
                    'webhookUrl',
                ]

                missing = [f for f in required_fields if f not in data]
                if missing:
                    raise ValueError(f"Provisioner response missing fields: {missing}")

                result = ProvisionResult(
                    nextcloud_base_url=data['nextcloudBaseUrl'],
                    room_token=data['roomToken'],
                    room_url=data['roomUrl'],
                    bot_id=data['botId'],
                    bot_name=data['botName'],
                    bot_secret=data['botSecret'],
                    webhook_url=data['webhookUrl'],
                )

                log.info(
                    f"✓ Provisioned: bot_id={result.bot_id}, "
                    f"room_token={result.room_token}"
                )

                return result

            except httpx.HTTPStatusError as e:
                log.error(
                    f"Provisioner returned {e.response.status_code}: "
                    f"{e.response.text}"
                )
                raise
            except httpx.RequestError as e:
                log.error(f"Provisioner request failed: {e}")
                raise

    async def rename_room(self, room_token: str, new_name: str) -> bool:
        """
        Rename a Nextcloud Talk room.

        Args:
            room_token: The room token
            new_name: New name for the room

        Returns:
            True if successful

        Raises:
            httpx.HTTPError: If rename fails
        """
        url = f"{self.base_url}/room/rename"

        payload = {
            "token": room_token,
            "name": new_name,
        }

        log.info(f"Renaming room {room_token} to '{new_name}'")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                log.info(f"✓ Renamed room {room_token}")
                return True

            except httpx.HTTPStatusError as e:
                log.error(
                    f"Room rename failed {e.response.status_code}: "
                    f"{e.response.text}"
                )
                raise
            except httpx.RequestError as e:
                log.error(f"Room rename request failed: {e}")
                raise


def _first_env(names: tuple[str, ...]) -> Optional[str]:
    return next((value for name in names if (value := os.getenv(name)) and value.strip()), None)


def get_provisioner_client(config=None) -> ProvisionerClient:
    """A client on the credentials stored now (the record first, else legacy settings).

    Raises :class:`TalkProvisionerUnavailable` for a record that exists but
    can't be used (never falling back to env) and ``ValueError`` when nothing
    is configured.
    """
    if config is None:
        from llm_bawt.utils.config import Config  # noqa: PLC0415

        config = Config()
    legacy = {
        "url": _first_env(_LEGACY_URL_VARS) or getattr(config, "TALK_PROVISIONER_URL", None),
        "token": _first_env(_LEGACY_TOKEN_VARS) or getattr(config, "TALK_PROVISIONER_TOKEN", None),
    }
    try:
        found = resolve_integration(config, INTEGRATION_ID, legacy,
                                    legacy_names=(*_LEGACY_URL_VARS, *_LEGACY_TOKEN_VARS))
    except ConnectionUnavailable as e:
        raise TalkProvisionerUnavailable(e.code) from None
    if found is None:
        raise ValueError("The Talk provisioner is not configured; connect it under Integrations")
    return ProvisionerClient(base_url=found.meta["url"], token=found.secret["token"])
