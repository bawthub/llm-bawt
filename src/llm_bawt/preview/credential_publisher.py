"""Production owner of the revocable Preview grant and gateway-only bearer file.

Activation is deployment configuration, not a public route or generic setting:
the approved service configuration supplies an exact run, provider, endpoint,
expiry and export directory. An absent configuration withdraws any old grant at
startup. Only the production app runs this publisher; Preview never imports it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import text

from llm_bawt.bawthub_auth.token_export import publish
from llm_bawt.service.preview_credential import (
    GRANT_KEY, PreviewGrant, _provider_for_endpoint, _read_setting, issue_capability,
)

RUN_ENV = "PREVIEW_CREDENTIAL_RUN_ID"
PROVIDER_ENV = "PREVIEW_CREDENTIAL_PROVIDER"
ENDPOINT_ENV = "PREVIEW_CREDENTIAL_ENDPOINT_ID"
EXPIRY_ENV = "PREVIEW_CREDENTIAL_RUN_EXPIRES_AT"
EXPORT_ENV = "PREVIEW_CREDENTIAL_CAPABILITY_EXPORT_FILE"
EXPORT_UID_ENV = "PREVIEW_CREDENTIAL_CAPABILITY_EXPORT_UID"
ACTIVATION_ENVS = (RUN_ENV, PROVIDER_ENV, ENDPOINT_ENV, EXPIRY_ENV, EXPORT_ENV)
RENEW_SECONDS = 120  # shorter than the capability's 600-second lifetime


class ActivationUnavailable(ValueError):
    """Incomplete or invalid approved activation configuration."""


def configured_grant(env: Mapping[str, str], now: int) -> tuple[PreviewGrant, Path] | None:
    values = [(env.get(name) or "").strip() for name in ACTIVATION_ENVS]
    if not any(values):
        return None
    if not all(values):
        raise ActivationUnavailable("Preview credential activation incomplete")
    run_id, provider, endpoint, expiry, raw_file = values
    try:
        if str(UUID(run_id)) != run_id or not endpoint.isdecimal() or not expiry.isdecimal():
            raise ValueError("bad Preview activation fields")
        grant = PreviewGrant.parse({"run_id": run_id, "provider": provider,
                                    "endpoint_id": int(endpoint), "expires_at": int(expiry)}, now)
    except (ValueError, AttributeError, OverflowError) as exc:
        raise ActivationUnavailable("Preview credential activation invalid") from exc
    file = Path(raw_file)
    if not file.is_absolute() or file.is_dir():
        raise ActivationUnavailable("Preview credential export path invalid")
    return grant, file


def _upsert_grant(engine: Any, grant: PreviewGrant | None) -> None:
    with engine.begin() as conn:
        if grant is None:
            conn.execute(text("DELETE FROM runtime_settings WHERE scope_type='global' "
                              "AND scope_id='*' AND key=:key"), {"key": GRANT_KEY})
        else:
            conn.execute(text(
                "INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at) "
                "VALUES ('global', '*', :key, :value, now()) "
                "ON CONFLICT (scope_type, scope_id, key) "
                "DO UPDATE SET value_json=EXCLUDED.value_json, updated_at=now()"),
                {"key": GRANT_KEY, "value": json.dumps(grant.__dict__)})


def activate(engine: Any, catalog: Any, env: Mapping[str, str] | None = None,
             *, now: int | None = None) -> Path | None:
    """At production startup: write the exactly selected grant or revoke it.

    A partially supplied activation fails startup instead of selecting a model
    or falling back to another provider. The approved Compose configuration is
    the only grant writer; the settings API rejects this reserved key.
    """
    env = os.environ if env is None else env
    now = int(time.time()) if now is None else now
    selected = configured_grant(env, now)
    if selected is None:
        _upsert_grant(engine, None)
        raw_file = (env.get(EXPORT_ENV) or "").strip()
        if raw_file:
            publish(Path(raw_file), None, env, uid_env=EXPORT_UID_ENV)
        return None
    grant, file = selected
    _provider_for_endpoint(catalog, grant)
    # Check write permissions before authorizing the DB grant. A missing export
    # must not leave an active grant behind (which would serve a leaked bearer).
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.parent.is_dir():
        raise ActivationUnavailable("Preview credential export directory unavailable")
    _upsert_grant(engine, grant)
    try:
        if not refresh(engine, catalog, file, env, now=now):
            raise ActivationUnavailable("Preview credential capability export unavailable")
    except BaseException:
        _upsert_grant(engine, None)
        publish(file, None, env, uid_env=EXPORT_UID_ENV)
        raise
    return file


def refresh(engine: Any, catalog: Any, file: Path, env: Mapping[str, str],
            *, now: int | None = None) -> bool:
    """Read live DB grant every time; revoke stale file on any refusal."""
    now = int(time.time()) if now is None else now
    try:
        grant = PreviewGrant.parse(_read_setting(engine, GRANT_KEY), now)
        _provider_for_endpoint(catalog, grant)
        selected = configured_grant(env, now)
        if selected is None or selected[0] != grant or selected[1] != file:
            raise ActivationUnavailable("Preview credential grant changed")
        capability = issue_capability(grant, now=now)
        payload = json.dumps({**grant.__dict__, "capability": capability}, separators=(",", ":"))
        publish(file, payload, env, uid_env=EXPORT_UID_ENV)
        return True
    except Exception:
        publish(file, None, env, uid_env=EXPORT_UID_ENV)
        return False


async def renew_forever(engine: Any, catalog: Any, file: Path, env: Mapping[str, str]) -> None:
    """Renew owner-signed capability without exposing a signing key to Preview."""
    while True:
        await asyncio.sleep(RENEW_SECONDS)
        await asyncio.to_thread(refresh, engine, catalog, file, env)
