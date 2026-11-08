"""X public-post search credentials, separate from xAI model credentials."""

from ...integrations.x_api import XApiError, request_x
from .api_key import ApiKeyAdapter
from .base import HEALTH_BROKEN, HEALTH_UNCONFIGURED, health_block


class XAdapter(ApiKeyAdapter):
    id = "x"
    label = "X (Twitter)"

    def descriptor(self) -> dict:
        return {
            **super().descriptor(),
            "category": "search",
            "credential_label": "Bearer token",
            "setup_url": "https://console.x.com/",
            "description": "Optional, read-only search of public X posts from the last seven days. Separate from xAI/Grok. Searches spend X API credits; automatic web search does not call X.",
            "credential_help": "Generate an app-only Bearer Token in the X Developer Console. Set a spending limit and fund API credits. Connection checks usage access without searching posts; search access also depends on your X plan and credit balance.",
        }

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        try:
            data = request_x(key, "usage/tweets")
        except XApiError as exc:
            return None, str(exc)
        if not isinstance(data.get("data"), dict) or not data["data"].get("project_id"):
            return None, "X returned an invalid usage response; bearer token was not saved."
        return "X API app", None

    def health(self) -> dict:
        # Local only: polling must never spend credits or exhaust upstream quotas.
        record = self.store.load(self.id)
        if not record:
            return health_block(HEALTH_UNCONFIGURED, detail="not connected")
        if record.status != "connected" or not record.secret.get("api_key"):
            return health_block(HEALTH_BROKEN, detail="X bearer token is unavailable; reconnect.", fix="reconnect")
        from .api_key_health import api_key_health

        return api_key_health(record, detail="Bearer token stored; search availability depends on X API access and credits.")
