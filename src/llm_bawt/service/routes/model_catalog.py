"""Normalized model/access-path/endpoint CRUD and cascade routes."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from claude_code_bridge.proxy.transport_policy import (
    VALID_RESPONSES_TRANSPORTS,
    normalize_responses_transport,
)

from ...model_catalog import AccessPath, ProtocolCompatibility
from ..dependencies import get_model_catalog_engine, get_service

router = APIRouter(prefix="/v1/models/catalog", tags=["Models"])


class ModelWrite(BaseModel):
    vendor: str
    display_name: str
    description: str | None = None
    # TASK-616: required + positive. The catalog is the source of truth for
    # per-model context windows; a windowless model silently falls back to the
    # global default (model_context_window_default), which over-budgets small
    # local models and under-budgets large cloud ones. Reject at write time so
    # the global default stays a deliberate fallback, not a substitute for
    # incomplete catalog data. Enforced in the DB by models_context_window
    # (NOT NULL + CHECK > 0) in model_catalog_migration.py.
    default_context_window: int = Field(gt=0)
    default_tool_support: str | None = None
    # Claude Code auto-compact window. None = global agent_compact_threshold.
    # Bounds mirror the CLI's accepted CLAUDE_CODE_AUTO_COMPACT_WINDOW range.
    compact_threshold: int | None = Field(default=None, ge=100_000, le=1_000_000)


class AccessPathWrite(BaseModel):
    vendor: str
    protocol: Literal["chat-completions", "responses", "anthropic-messages"]
    base_url: str | None = None
    auth_mechanism: str
    engine_kind: str | None = None
    system_prompt_overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("system_prompt_overrides")
    @classmethod
    def normalize_system_prompt_overrides(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        unknown = sorted(set(value) - set(ProtocolCompatibility.HARNESS_PROTOCOLS))
        if unknown:
            raise ValueError(f"Unknown harness key(s): {', '.join(unknown)}")
        return {
            harness: body.strip()
            for harness, body in value.items()
            if body.strip()
        }


class EndpointWrite(BaseModel):
    endpoint_id: int | None = Field(
        default=None,
        description="Compare-and-set guard when updating an existing endpoint.",
    )
    upstream_model_id: str | None = None
    serving_config: dict[str, Any] = Field(default_factory=dict)
    context_window_override: int | None = Field(default=None, gt=0)
    tool_support_override: str | None = None
    pricing: dict[str, Any] | None = None

    @field_validator("serving_config")
    @classmethod
    def normalize_serving_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(value)
        effort = normalized.get("reasoning_effort")
        if effort is not None:
            if not isinstance(effort, str) or effort.strip().lower() not in {
                "low", "medium", "high", "xhigh", "max",
            }:
                raise ValueError(
                    "serving_config.reasoning_effort must be one of: "
                    "low, medium, high, xhigh, max"
                )
            normalized["reasoning_effort"] = effort.strip().lower()

        transport = normalized.get("responses_transport")
        if transport is not None:
            canonical_transport = normalize_responses_transport(transport)
            if canonical_transport is None:
                choices = ", ".join(sorted(VALID_RESPONSES_TRANSPORTS))
                raise ValueError(
                    "serving_config.responses_transport must be one of: "
                    f"{choices}"
                )
            normalized["responses_transport"] = canonical_transport
        return normalized


def _engine():
    engine = get_model_catalog_engine(get_service().config)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model catalog database unavailable")
    return engine


def _refresh_catalog() -> None:
    # Local import avoids a module-import cycle with the compatibility routes.
    from .models import reload_models_catalog

    reload_models_catalog()


def _dict_rows(result) -> list[dict[str, Any]]:
    return [dict(row) for row in result.mappings().all()]


def _normalized_harness(harness: str | None) -> str | None:
    normalized = ProtocolCompatibility.normalize_harness(harness)
    if normalized and normalized not in ProtocolCompatibility.HARNESS_PROTOCOLS:
        raise HTTPException(status_code=400, detail=f"Unknown harness '{harness}'")
    return normalized


def _check_endpoint_cas(existing_id: int | None, expected_id: int | None) -> None:
    if expected_id is None:
        return
    if existing_id is None:
        raise HTTPException(status_code=409, detail="Expected endpoint no longer exists")
    if existing_id != expected_id:
        raise HTTPException(
            status_code=409,
            detail=f"Endpoint changed: expected {expected_id}, found {existing_id}",
        )


def _delete_catalog_model(conn, model_key: str):
    """Delete one normalized model and cascade its unassigned endpoints."""
    return conn.execute(
        text("DELETE FROM models WHERE key = :key RETURNING id"),
        {"key": model_key},
    ).first()


@router.get("/harnesses")
def list_harnesses():
    """List harnesses and the protocol rule used by the cascade UI."""
    return {
        "harnesses": [
            {"key": key, "protocol": protocol, "proxies": key == "claude-proxy"}
            for key, protocol in ProtocolCompatibility.HARNESS_PROTOCOLS.items()
        ]
    }


@router.get("/models")
def list_catalog_models(access_path: str | None = None):
    sql = """
        SELECT DISTINCT m.*
        FROM models m
        LEFT JOIN model_endpoints e ON e.model_id = m.id
        LEFT JOIN access_paths a ON a.id = e.access_path_id
        WHERE (:access_path IS NULL OR a.key = :access_path)
        ORDER BY m.key
    """
    with _engine().connect() as conn:
        return {"models": _dict_rows(conn.execute(text(sql), {"access_path": access_path}))}


@router.put("/models/{model_key}")
def put_catalog_model(model_key: str, request: ModelWrite):
    sql = """
        INSERT INTO models
            (key, vendor, display_name, description, default_context_window,
             default_tool_support, compact_threshold, created_at, updated_at)
        VALUES
            (:key, :vendor, :display_name, :description, :default_context_window,
             :default_tool_support, :compact_threshold, NOW(), NOW())
        ON CONFLICT (key) DO UPDATE SET
            vendor = EXCLUDED.vendor,
            display_name = EXCLUDED.display_name,
            description = EXCLUDED.description,
            default_context_window = EXCLUDED.default_context_window,
            default_tool_support = EXCLUDED.default_tool_support,
            -- Only overwrite when the caller sent the field, so older PUT
            -- callers that don't know it can't silently clear it.
            compact_threshold = CASE WHEN :compact_threshold_sent
                THEN EXCLUDED.compact_threshold ELSE models.compact_threshold END,
            updated_at = NOW()
        RETURNING *
    """
    with _engine().begin() as conn:
        params = {
            "key": model_key,
            **request.model_dump(),
            "compact_threshold_sent": "compact_threshold" in request.model_fields_set,
        }
        row = dict(conn.execute(text(sql), params).mappings().one())
    _refresh_catalog()
    return row


@router.delete("/models/{model_key}")
def delete_catalog_model(model_key: str):
    try:
        with _engine().begin() as conn:
            bot = conn.execute(text("""
                SELECT b.slug FROM bot_profiles b
                JOIN model_endpoints e ON e.id = b.endpoint_id
                JOIN models m ON m.id = e.model_id
                WHERE m.key = :key LIMIT 1
            """), {"key": model_key}).scalar()
            if bot:
                raise HTTPException(status_code=409, detail=f"Model is used by bot '{bot}'")
            row = _delete_catalog_model(conn, model_key)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Model is still in use: {exc}") from exc
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    _refresh_catalog()
    return {"ok": True, "key": model_key}


@router.get("/access-paths")
def list_access_paths(harness: str | None = Query(default=None)):
    normalized = _normalized_harness(harness)
    with _engine().connect() as conn:
        rows = _dict_rows(conn.execute(text("SELECT * FROM access_paths ORDER BY key")))
    if normalized:
        rows = [
            row
            for row in rows
            if ProtocolCompatibility.is_compatible(
                normalized,
                AccessPath(
                    row["id"], row["key"], row["vendor"], row["protocol"],
                    row["base_url"], row["auth_mechanism"], row["engine_kind"],
                ),
            )
        ]
    return {"access_paths": rows}


@router.put("/access-paths/{access_key}")
def put_access_path(access_key: str, request: AccessPathWrite):
    sql = """
        INSERT INTO access_paths
            (key, vendor, protocol, base_url, auth_mechanism, engine_kind,
             system_prompt_overrides, created_at, updated_at)
        VALUES
            (:key, :vendor, :protocol, :base_url, :auth_mechanism, :engine_kind,
             CAST(:system_prompt_overrides AS jsonb), NOW(), NOW())
        ON CONFLICT (key) DO UPDATE SET
            vendor = EXCLUDED.vendor,
            protocol = EXCLUDED.protocol,
            base_url = EXCLUDED.base_url,
            auth_mechanism = EXCLUDED.auth_mechanism,
            engine_kind = EXCLUDED.engine_kind,
            system_prompt_overrides = EXCLUDED.system_prompt_overrides,
            updated_at = NOW()
        RETURNING *
    """
    try:
        with _engine().begin() as conn:
            params = {"key": access_key, **request.model_dump()}
            params["system_prompt_overrides"] = json.dumps(
                request.system_prompt_overrides
            )
            row = dict(conn.execute(text(sql), params).mappings().one())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Access path update rejected: {exc}") from exc
    _refresh_catalog()
    return row


@router.delete("/access-paths/{access_key}")
def delete_access_path(access_key: str):
    try:
        with _engine().begin() as conn:
            row = conn.execute(
                text("DELETE FROM access_paths WHERE key = :key RETURNING id"),
                {"key": access_key},
            ).first()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Access path is still in use: {exc}") from exc
    if not row:
        raise HTTPException(status_code=404, detail="Access path not found")
    _refresh_catalog()
    return {"ok": True, "key": access_key}


@router.get("/endpoints")
def list_endpoints(
    harness: str | None = None,
    access_path: str | None = None,
    model: str | None = None,
):
    normalized = _normalized_harness(harness)
    sql = """
        SELECT e.*, m.key AS model_key, m.vendor AS model_vendor,
               m.display_name, m.description,
               m.default_context_window, m.default_tool_support,
               m.compact_threshold,
               a.key AS access_path_key, a.vendor AS access_vendor,
               a.protocol, a.base_url, a.auth_mechanism, a.engine_kind
        FROM model_endpoints e
        JOIN models m ON m.id = e.model_id
        JOIN access_paths a ON a.id = e.access_path_id
        WHERE (:access_path IS NULL OR a.key = :access_path)
          AND (:model IS NULL OR m.key = :model)
        ORDER BY m.key, a.key
    """
    with _engine().connect() as conn:
        rows = _dict_rows(conn.execute(text(sql), {"access_path": access_path, "model": model}))
    if normalized:
        rows = [
            row for row in rows
            if ProtocolCompatibility.is_compatible(
                normalized,
                AccessPath(
                    row["access_path_id"], row["access_path_key"], row["access_vendor"],
                    row["protocol"], row["base_url"], row["auth_mechanism"],
                    row["engine_kind"],
                ),
            )
        ]
    return {"endpoints": rows}


@router.put("/endpoints/{model_key}/{access_key}")
def put_endpoint(model_key: str, access_key: str, request: EndpointWrite):
    """Upsert exactly one (model, access-path) endpoint with CAS protection."""
    with _engine().begin() as conn:
        ids = conn.execute(
            text("""
                SELECT m.id AS model_id, a.id AS access_path_id
                FROM models m CROSS JOIN access_paths a
                WHERE m.key = :model_key AND a.key = :access_key
            """),
            {"model_key": model_key, "access_key": access_key},
        ).mappings().first()
        if not ids:
            raise HTTPException(status_code=404, detail="Model or access path not found")
        existing = conn.execute(
            text("""
                SELECT id FROM model_endpoints
                WHERE model_id = :model_id AND access_path_id = :access_path_id
                FOR UPDATE
            """),
            ids,
        ).mappings().first()
        _check_endpoint_cas(existing["id"] if existing else None, request.endpoint_id)
        params = {
            **ids,
            "upstream_model_id": request.upstream_model_id,
            "serving_config": json.dumps(request.serving_config),
            "context_window_override": request.context_window_override,
            "tool_support_override": request.tool_support_override,
            "pricing": json.dumps(request.pricing) if request.pricing is not None else None,
        }
        row = conn.execute(text("""
            INSERT INTO model_endpoints
                (model_id, access_path_id, upstream_model_id, serving_config,
                 context_window_override, tool_support_override, pricing,
                 created_at, updated_at)
            VALUES
                (:model_id, :access_path_id, :upstream_model_id,
                 CAST(:serving_config AS jsonb), :context_window_override,
                 :tool_support_override, CAST(:pricing AS jsonb), NOW(), NOW())
            ON CONFLICT (model_id, access_path_id) DO UPDATE SET
                upstream_model_id = EXCLUDED.upstream_model_id,
                serving_config = EXCLUDED.serving_config,
                context_window_override = EXCLUDED.context_window_override,
                tool_support_override = EXCLUDED.tool_support_override,
                pricing = EXCLUDED.pricing,
                updated_at = NOW()
            RETURNING *
        """), params).mappings().one()
    _refresh_catalog()
    return dict(row)


@router.delete("/endpoints/{endpoint_id}")
def delete_endpoint(endpoint_id: int):
    with _engine().begin() as conn:
        bot = conn.execute(
            text("SELECT slug FROM bot_profiles WHERE endpoint_id = :id LIMIT 1"),
            {"id": endpoint_id},
        ).scalar()
        if bot:
            raise HTTPException(status_code=409, detail=f"Endpoint is used by bot '{bot}'")
        row = conn.execute(
            text("DELETE FROM model_endpoints WHERE id = :id RETURNING id"),
            {"id": endpoint_id},
        ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    _refresh_catalog()
    return {"ok": True, "id": endpoint_id}
