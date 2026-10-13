"""Prompt template management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ...prompt_registry import COMPACT_SUMMARY_KEY, PromptResolver, PromptTemplate, extract_placeholders
from ..dependencies import get_service
from ..schemas import (
    PersonaCreateRequest,
    PersonaResponse,
    PersonaUpdateRequest,
    PromptTemplateIndexResponse,
    PromptTemplateListResponse,
    PromptTemplateResponse,
    PromptTemplateSeedResponse,
    PromptTemplateSummary,
    PromptTemplateUpsertRequest,
    PromptTemplateValidateRequest,
    PromptTemplateValidateResponse,
    PromptTemplateVersionResponse,
    PromptTemplateVersionsResponse,
)

router = APIRouter()


def _row_to_response(row: PromptTemplate) -> PromptTemplateResponse:
    from ...prompt_registry import _parse_json_dict, _parse_json_list

    return PromptTemplateResponse(
        key=row.key,
        title=row.title or row.key,
        category=row.category or "custom",
        format=row.format or "plain_text",
        body=row.body,
        scope_type=row.scope_type,
        scope_id=row.scope_id,
        source="db_override",
        required_vars=_parse_json_list(row.required_vars_json),
        placeholders=extract_placeholders(row.body),
        metadata=_parse_json_dict(row.metadata_json),
        updated_at=row.updated_at,
    )


def _resolved_to_response(resolved) -> PromptTemplateResponse:
    return PromptTemplateResponse(
        key=resolved.key,
        title=resolved.title,
        category=resolved.category,
        format=resolved.format,
        body=resolved.body,
        scope_type=resolved.scope_type,
        scope_id=resolved.scope_id,
        source=resolved.source,
        required_vars=list(resolved.required_vars),
        placeholders=extract_placeholders(resolved.body),
        metadata=dict(resolved.metadata),
        updated_at=resolved.updated_at,
    )


def _response_to_summary(item: PromptTemplateResponse) -> PromptTemplateSummary:
    """Strip body/placeholders/metadata/required_vars off a full response."""
    return PromptTemplateSummary(
        key=item.key,
        title=item.title,
        category=item.category,
        format=item.format,
        scope_type=item.scope_type,
        scope_id=item.scope_id,
        source=item.source,
        body_length=len(item.body or ""),
        updated_at=item.updated_at,
    )


def _collect_prompts(
    *,
    category: str | None,
    scope_type: str,
    scope_id: str | None,
    include_defaults: bool,
) -> tuple[list[PromptTemplateResponse], dict[str, Any]]:
    """Shared resolution path for the list + index endpoints.

    Returns (prompts, filters_echo) so callers can wrap the list in either the
    full-body response model or the summary-only index model with identical
    semantics. Pulled out so /v1/prompts and /v1/prompts/index can never drift.
    """
    service = get_service()
    resolver = PromptResolver(service.config)

    prompts: list[PromptTemplateResponse] = []
    seen_keys: set[tuple[str, str, str]] = set()
    normalized_scope_type = scope_type.strip().lower() if scope_type else "global"
    normalized_scope_id = scope_id

    if include_defaults:
        for key, definition in sorted(resolver.definitions().items()):
            if category and definition.category != category.strip().lower():
                continue
            resolved = resolver.resolve(key, normalized_scope_type, normalized_scope_id)
            if resolved is None:
                continue
            item = _resolved_to_response(resolved)
            prompts.append(item)
            seen_keys.add((item.key, item.scope_type, item.scope_id))

    rows: list[PromptTemplate] = []
    if include_defaults:
        rows.extend(
            resolver.store.list_rows(
                category=category,
                scope_type="global",
                scope_id="*",
            )
        )
        if normalized_scope_type == "bot":
            rows.extend(
                resolver.store.list_rows(
                    category=category,
                    scope_type="bot",
                    scope_id=normalized_scope_id,
                )
            )
    else:
        rows = resolver.store.list_rows(
            category=category,
            scope_type=normalized_scope_type,
            scope_id=normalized_scope_id,
        )

    for row in rows:
        marker = (row.key, row.scope_type, row.scope_id)
        if marker in seen_keys:
            continue
        prompts.append(_row_to_response(row))

    filters = {
        "category": category,
        "scope_type": normalized_scope_type,
        "scope_id": normalized_scope_id,
        "include_defaults": include_defaults,
    }
    return prompts, filters


@router.get("/v1/prompts", response_model=PromptTemplateListResponse, tags=["Admin"])
def list_prompts(
    category: str | None = Query(None, description="Filter by prompt category"),
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped resolution"),
    include_defaults: bool = Query(True, description="Include code-default prompts in the listing"),
):
    """List prompt templates with full bodies. Heavy.

    Returns every prompt's full `body`, `placeholders`, `metadata`, and
    `required_vars`. For agent or admin-UI enumeration where you only need
    names + scope + source, use `GET /v1/prompts/index` — same filters,
    bodies stripped.
    """
    try:
        prompts, filters = _collect_prompts(
            category=category,
            scope_type=scope_type,
            scope_id=scope_id,
            include_defaults=include_defaults,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PromptTemplateListResponse(
        prompts=prompts,
        total_count=len(prompts),
        filters=filters,
    )


@router.get("/v1/prompts/index", response_model=PromptTemplateIndexResponse, tags=["Admin"])
def list_prompts_index(
    category: str | None = Query(None, description="Filter by prompt category"),
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped resolution"),
    include_defaults: bool = Query(True, description="Include code-default prompts in the listing"),
):
    """List prompt templates by name only — no bodies. Context-safe for agents.

    Mirrors `GET /v1/prompts`'s resolution and filters but returns
    `PromptTemplateSummary` entries (key, title, category, scope, source,
    format, body_length, updated_at). Fetch full body for a chosen prompt
    via `GET /v1/prompts/{key}` once the agent has picked one.
    """
    try:
        prompts, filters = _collect_prompts(
            category=category,
            scope_type=scope_type,
            scope_id=scope_id,
            include_defaults=include_defaults,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    summaries = [_response_to_summary(item) for item in prompts]
    return PromptTemplateIndexResponse(
        prompts=summaries,
        total_count=len(summaries),
        filters=filters,
    )


@router.post("/v1/prompts/seed-defaults", response_model=PromptTemplateSeedResponse, tags=["Admin"])
def seed_prompt_defaults():
    """Insert any missing built-in prompt templates into the DB."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    result = resolver.store.seed_defaults()
    return PromptTemplateSeedResponse(**result)


@router.get("/v1/prompts/{key}/versions", response_model=PromptTemplateVersionsResponse, tags=["Admin"])
def list_prompt_versions(
    key: str,
    scope_type: str = Query("global", description="Scope type: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """List saved versions for one prompt key/scope."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    try:
        versions = resolver.store.list_versions(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PromptTemplateVersionsResponse(
        key=key,
        scope_type=scope_type,
        scope_id=scope_id or ("*" if scope_type == "global" else ""),
        versions=[
            PromptTemplateVersionResponse(
                version=row.version,
                body=row.body,
                change_note=row.change_note,
                created_by=row.created_by,
                created_at=row.created_at,
            )
            for row in versions
        ],
        total_count=len(versions),
    )


@router.get("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
def get_prompt(
    key: str,
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
    exact: bool = Query(False, description="If true, fetch the exact DB row only (no fallback)"),
):
    """Get one prompt template, optionally with code-default fallback resolution."""
    service = get_service()
    resolver = PromptResolver(service.config)
    # This key guides live compaction. If DB access is unavailable, serving the
    # code default would silently override a saved edit. Fail closed instead;
    # the bridge can compact without additional guidance for this occurrence.
    if key == COMPACT_SUMMARY_KEY and resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Compaction prompt DB unavailable")

    if exact:
        if resolver.store.engine is None:
            raise HTTPException(status_code=503, detail="Prompt template DB unavailable")
        try:
            row = resolver.store.get_exact(key, scope_type, scope_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if row is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        return _row_to_response(row)

    try:
        resolved = resolver.resolve(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if resolved is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    return _resolved_to_response(resolved)


@router.put("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
@router.patch("/v1/prompts/{key}", response_model=PromptTemplateResponse, tags=["Admin"])
def upsert_prompt(key: str, request: PromptTemplateUpsertRequest):
    """Create or update one prompt template override."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    validation = resolver.validate(key=key, body=request.body, required_vars=request.required_vars or None)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"errors": validation["errors"], **validation})

    definition = resolver.definition_for(key)
    try:
        row = resolver.store.upsert(
            key=key,
            body=request.body,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            title=request.title or (definition.title if definition else key),
            category=(request.category or (definition.category if definition else "custom")).strip().lower(),
            format=request.format or (definition.format if definition else "plain_text"),
            required_vars=validation["required_vars"],
            metadata=request.metadata,
            updated_by=request.updated_by,
            change_note=request.change_note,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _row_to_response(row)


@router.post("/v1/prompts/{key}/validate", response_model=PromptTemplateValidateResponse, tags=["Admin"])
def validate_prompt(key: str, request: PromptTemplateValidateRequest):
    """Validate a prompt template body and optionally render a preview."""
    service = get_service()
    resolver = PromptResolver(service.config)

    body = request.body
    if body is None:
        resolved = resolver.resolve(key)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        body = resolved.body

    validation = resolver.validate(key=key, body=body, required_vars=request.required_vars)
    rendered_preview = None
    if validation["valid"] and request.variables:
        try:
            rendered_preview = resolver.render(key=key, variables=request.variables, body_override=body)
        except KeyError as e:
            validation["valid"] = False
            validation["errors"].append(f"Missing preview variable: {e.args[0]}")

    return PromptTemplateValidateResponse(
        valid=validation["valid"],
        required_vars=validation["required_vars"],
        placeholders=validation["placeholders"],
        missing_required=validation["missing_required"],
        unknown_placeholders=validation["unknown_placeholders"],
        rendered_preview=rendered_preview,
        errors=validation["errors"],
    )


@router.post("/v1/prompts/{key}/preview", response_model=PromptTemplateValidateResponse, tags=["Admin"])
def preview_prompt(
    key: str,
    request: PromptTemplateValidateRequest,
    scope_type: str = Query("global", description="Resolution scope: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """Render a prompt template with variables to preview the final prompt text."""
    service = get_service()
    resolver = PromptResolver(service.config)

    body = request.body
    if body is None:
        try:
            resolved = resolver.resolve(key, scope_type, scope_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if resolved is None:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        body = resolved.body

    validation = resolver.validate(key=key, body=body, required_vars=request.required_vars)
    if not validation["valid"]:
        return PromptTemplateValidateResponse(
            valid=False,
            required_vars=validation["required_vars"],
            placeholders=validation["placeholders"],
            missing_required=validation["missing_required"],
            unknown_placeholders=validation["unknown_placeholders"],
            rendered_preview=None,
            errors=validation["errors"],
        )

    try:
        preview = resolver.render(key=key, variables=request.variables, body_override=body)
    except KeyError as e:
        return PromptTemplateValidateResponse(
            valid=False,
            required_vars=validation["required_vars"],
            placeholders=validation["placeholders"],
            missing_required=validation["missing_required"],
            unknown_placeholders=validation["unknown_placeholders"],
            rendered_preview=None,
            errors=[f"Missing preview variable: {e.args[0]}"],
        )

    return PromptTemplateValidateResponse(
        valid=True,
        required_vars=validation["required_vars"],
        placeholders=validation["placeholders"],
        missing_required=validation["missing_required"],
        unknown_placeholders=validation["unknown_placeholders"],
        rendered_preview=preview,
        errors=[],
    )


@router.post("/v1/prompts/{key}/reset", response_model=PromptTemplateResponse, tags=["Admin"])
def reset_prompt(
    key: str,
    scope_type: str = Query("global", description="Scope type: global or bot"),
    scope_id: str | None = Query(None, description="Scope ID for bot-scoped prompts"),
):
    """Delete a DB override and fall back to the code-default prompt."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")

    try:
        deleted = resolver.store.reset(key, scope_type, scope_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resolved = resolver.resolve(key, scope_type, scope_id)
    if resolved is None:
        if not deleted:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        raise HTTPException(status_code=404, detail="Prompt template reset but no code default exists")
    return _resolved_to_response(resolved)


# ── Personas (TASK-477) ────────────────────────────────────────────────
# Switchable, shareable system-prompt overrides. Backed by prompt_templates
# rows (category='persona', global scope) so they inherit versioning.

def _persona_to_response(row: PromptTemplate) -> PersonaResponse:
    return PersonaResponse(
        id=row.id,
        key=row.key,
        title=row.title or row.key,
        body=row.body,
        updated_at=row.updated_at,
        created_at=row.created_at,
    )


def _persona_store():
    """Return a prompt-template store, or 503 if the DB is unavailable."""
    service = get_service()
    resolver = PromptResolver(service.config)
    if resolver.store.engine is None:
        raise HTTPException(status_code=503, detail="Prompt template DB unavailable")
    return resolver.store


@router.get("/v1/personas", tags=["System"])
def list_personas():
    """List all global (shareable) personas."""
    store = _persona_store()
    return {"personas": [_persona_to_response(r) for r in store.list_personas()]}


@router.post("/v1/personas", tags=["System"])
def create_persona(request: PersonaCreateRequest):
    """Create a new global persona."""
    store = _persona_store()
    title = (request.title or "").strip()
    body = request.body or ""
    if not title:
        raise HTTPException(status_code=400, detail="Persona title is required")
    if not body.strip():
        raise HTTPException(status_code=400, detail="Persona body cannot be empty")
    row = store.create_persona(title=title, body=body, updated_by="ui")
    return _persona_to_response(row)


@router.get("/v1/personas/{persona_id}", tags=["System"])
def get_persona(persona_id: int):
    """Get one persona by id."""
    from ...prompt_registry import PERSONA_CATEGORY

    store = _persona_store()
    row = store.get_by_id(persona_id)
    if row is None or row.category != PERSONA_CATEGORY:
        raise HTTPException(status_code=404, detail="Persona not found")
    return _persona_to_response(row)


@router.put("/v1/personas/{persona_id}", tags=["System"])
@router.patch("/v1/personas/{persona_id}", tags=["System"])
def update_persona(persona_id: int, request: PersonaUpdateRequest):
    """Edit a persona (creates a new version). Shared: affects every bot using it."""
    store = _persona_store()
    if request.body is not None and not request.body.strip():
        raise HTTPException(status_code=400, detail="Persona body cannot be empty")
    row = store.update_persona(
        persona_id,
        body=request.body,
        title=request.title,
        updated_by="ui",
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return _persona_to_response(row)


@router.delete("/v1/personas/{persona_id}", tags=["System"])
def delete_persona(persona_id: int):
    """Delete a persona. Does not clear it from bots still referencing it."""
    store = _persona_store()
    deleted = store.delete_persona(persona_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"deleted": True, "id": persona_id}
