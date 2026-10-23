"""ChatGPT-subscription model catalog: request shape and response parser.

Shared by the app-side ChatGPT discovery provider
(``llm_bawt.service.model_discovery``) and the Codex bridge's ``/models``
endpoint (``codex_bridge.__main__``), so both describe the same backend call
and shape its rows identically (TASK-1296). It lives in ``agent_bridge``
because the bridge venv cannot import ``llm_bawt``; this package is the only
one both processes share. Pure functions: no I/O and no credential handling.
"""

from __future__ import annotations

from typing import Any

#: ``client_version`` the ChatGPT backend's ``/models`` gate is told. The
#: backend hides models whose ``minimal_client_version`` exceeds it (gpt-5.6
#: needs >= 0.144.0), so this is the floor a listing may claim, not a
#: statement about any installed binary. The app pins it as
#: ``Config.CODEX_CLIENT_VERSION`` (env ``LLM_BAWT_CODEX_CLIENT_VERSION``);
#: the Codex bridge uses it only when the real binary can't be probed.
CHATGPT_MODELS_CLIENT_VERSION = "0.144.1"

#: Upstream visibility value for models the backend lists but does not offer.
CHATGPT_MODEL_HIDDEN = "hide"


def chatgpt_models_request(
    *,
    base_url: str,
    bearer: str,
    account_id: str,
    client_version: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return ``(url, params, headers)`` for the backend's ``/models`` call.

    ``base_url`` is the ChatGPT Codex backend (``.../backend-api/codex``), not
    the platform API. ``chatgpt-account-id`` is mandatory: without it the
    backend answers 401 even with a valid bearer.
    """
    url = f"{base_url.rstrip('/')}/models"
    params = {"client_version": client_version}
    headers = {
        "Authorization": f"Bearer {bearer}",
        "chatgpt-account-id": account_id,
    }
    return url, params, headers


def parse_chatgpt_models(payload: object) -> list[dict[str, Any]]:
    """Shape the backend's ``/models`` payload into catalog rows.

    Keeps ``id`` (the ``slug``), ``context_window`` and ``description``; drops
    hidden rows and anything without a slug. Visibility says the model is
    offered to the plan, nothing more: it is not evidence of tool support.
    """
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    out: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("visibility") == CHATGPT_MODEL_HIDDEN:
            continue
        slug = item.get("slug") or item.get("id")
        if not slug:
            continue
        out.append(
            {
                "id": str(slug),
                "context_window": item.get("context_window"),
                "description": item.get("description") or item.get("display_name") or "",
            }
        )
    return out
