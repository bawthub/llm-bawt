"""Live, global wire-protocol selection for direct voice clients."""

from __future__ import annotations

import hashlib
import hmac
import json

VOICE_AUTH_MODE_KEY = "voice_backend_auth_mode"
VOICE_AUTH_MODES = ("legacy", "derived-v1")
VOICE_DERIVATION_LABEL = b"bawthub/voice-proxy/v1"


class VoiceCredentialUnavailable(RuntimeError):
    """The outgoing voice credential cannot be selected safely."""


def read_voice_auth_mode() -> str:
    """Read the exact global row on every call, including in the video bridge.

    Only an absent row selects the registry's legacy default. The general
    runtime resolver caches values and treats unreadable JSON as absent, so
    neither it nor get_value() is suitable for this wire migration.
    """
    from sqlalchemy import text

    from llm_bawt.setting_definitions import setting_default
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    try:
        engine = get_shared_engine(Config())
        if engine is None:
            raise VoiceCredentialUnavailable("Voice credential setting is unavailable")
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT value_json FROM runtime_settings "
                    "WHERE scope_type = 'global' AND scope_id = '*' AND key = :key"
                ),
                {"key": VOICE_AUTH_MODE_KEY},
            ).one_or_none()
        mode = setting_default(VOICE_AUTH_MODE_KEY) if row is None else json.loads(row[0])
        if mode not in VOICE_AUTH_MODES:
            raise VoiceCredentialUnavailable("Voice credential setting is invalid")
        return mode
    except Exception:
        # No DB URL, stored value or credential belongs in a downstream error.
        raise VoiceCredentialUnavailable("Voice credential setting is unavailable or invalid") from None


def derive_voice_credential(seed: str) -> str:
    """Match BawtHub's TS/Python v1 protocol; never return the service seed."""
    seed = seed.strip()
    if len(seed) < 32:
        raise VoiceCredentialUnavailable("Voice credential seed is unavailable")
    return hmac.new(seed.encode("utf-8"), VOICE_DERIVATION_LABEL, hashlib.sha256).hexdigest()
