"""The OpenClaw gateway token for the app's own gateway calls (TASK-1476).

Record first: the encrypted ``integration_connection:openclaw`` record wins,
the deprecated ``LLM_BAWT_OPENCLAW_GATEWAY_TOKEN`` setting applies only when
no record exists, and a record that exists but can't be used raises
:class:`~llm_bawt.service.connection_secrets.ConnectionUnavailable`. Resolve
per use so a saved or rotated token applies without a restart.
"""

from __future__ import annotations

from llm_bawt.service.connection_secrets import resolve_integration

LEGACY_SETTING = "LLM_BAWT_OPENCLAW_GATEWAY_TOKEN"


def gateway_token(config) -> str:
    """The token to present, or ``""`` when none is configured anywhere."""
    found = resolve_integration(
        config, "openclaw", {"token": getattr(config, "OPENCLAW_GATEWAY_TOKEN", "")},
        legacy_names=(LEGACY_SETTING,),
    )
    return found.secret["token"] if found is not None else ""


__all__ = ["LEGACY_SETTING", "gateway_token"]
