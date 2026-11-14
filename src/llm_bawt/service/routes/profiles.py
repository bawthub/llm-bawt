"""Unified profile routes."""

from sqlalchemy import func
from sqlmodel import Session, select

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import attribute_to_response, get_profile_manager, get_service
from ..logging import get_service_logger
from ..http_errors import internal_error
from ..schemas import (
    ProfileAttributeUpdateRequest,
    ProfileAttributeListResponse,
    ProfileAttributeUpsertRequest,
    ProfileDetail,
    ProfileListResponse,
    ProfileUpdateRequest,
    UserProfileAttribute,
)

router = APIRouter()
log = get_service_logger(__name__)


def _get_profile_detail(manager, entity_type_enum, entity_type: str, entity_id: str) -> ProfileDetail:
    profile = manager.get_profile(entity_type_enum, entity_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} '{entity_id}' not found")

    attributes = manager.get_all_attributes(entity_type_enum, entity_id)
    return ProfileDetail(
        entity_type=entity_type,
        entity_id=entity_id,
        email=profile.email,
        is_service=profile.is_service,
        display_name=profile.display_name,
        description=profile.description,
        color=profile.color,
        avatar=profile.avatar,
        avatar_render=profile.avatar_render,
        summary=profile.summary,
        summary_updated_at=profile.summary_updated_at.isoformat() if profile.summary_updated_at else None,
        attributes=[attribute_to_response(attr) for attr in attributes],
        created_at=profile.created_at.isoformat() if profile.created_at else None,
    )


def _normalize_entity_type(entity_type: str | None) -> str | None:
    if entity_type is None:
        return None

    normalized = entity_type.lower()
    if normalized not in ("user", "bot"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'bot'.",
        )
    return normalized


@router.get("/v1/profiles", response_model=ProfileListResponse, tags=["Profiles"])
def list_profiles(entity_type: str | None = None, entity_id: str | None = None):
    """List profiles.

    - /v1/profiles: list all user + bot profiles
    - /v1/profiles?entity_type=user|bot: list by type
    - entity_id narrows the result to that entity's profiles (TASK-2279)
    """
    if entity_id is not None and not entity_id.strip():
        raise HTTPException(status_code=400, detail="entity_id must be nonblank")
    service = get_service()

    try:
        from ...profiles import EntityType

        manager = get_profile_manager(service.config)
        normalized_entity_type = _normalize_entity_type(entity_type)

        result_profiles: list[ProfileDetail] = []
        wanted = entity_id.strip().lower() if entity_id is not None else None

        def _kept(profiles):
            if wanted is None:
                return profiles
            return [p for p in profiles if str(p.entity_id).strip().lower() == wanted]

        if normalized_entity_type:
            target_type = EntityType.USER if normalized_entity_type == "user" else EntityType.BOT
            typed_profiles = _kept(manager.list_profiles(target_type))
            result_profiles = [
                _get_profile_detail(manager, target_type, normalized_entity_type, profile.entity_id)
                for profile in typed_profiles
            ]
        else:
            user_profiles = _kept(manager.list_profiles(EntityType.USER))
            bot_profiles = _kept(manager.list_profiles(EntityType.BOT))

            result_profiles.extend(
                _get_profile_detail(manager, EntityType.USER, "user", profile.entity_id)
                for profile in user_profiles
            )
            result_profiles.extend(
                _get_profile_detail(manager, EntityType.BOT, "bot", profile.entity_id)
                for profile in bot_profiles
            )

        return ProfileListResponse(profiles=result_profiles, total_count=len(result_profiles))
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, "Failed to list profiles") from e


@router.get("/v1/profiles/attribute", response_model=ProfileAttributeListResponse, tags=["Profiles"])
def list_profile_attributes(
    entity_type: str | None = Query(None, description="Filter by entity type: user or bot"),
    entity_id: str | None = Query(None, description="Filter by entity ID"),
    category: str | None = Query(None, description="Filter by attribute category"),
    key: str | None = Query(None, description="Filter by attribute key"),
    source: str | None = Query(None, description="Filter by attribute source"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List profile attributes with filters and pagination."""
    service = get_service()

    try:
        from ...profiles import EntityType, ProfileAttribute

        manager = get_profile_manager(service.config)
        normalized_entity_type = _normalize_entity_type(entity_type)

        conditions: list = []
        if normalized_entity_type:
            target_type = EntityType.USER if normalized_entity_type == "user" else EntityType.BOT
            conditions.append(ProfileAttribute.entity_type == target_type)
        if entity_id:
            conditions.append(ProfileAttribute.entity_id == entity_id.strip().lower())
        if category:
            conditions.append(ProfileAttribute.category == category.strip().lower())
        if key:
            conditions.append(ProfileAttribute.key == key.strip().lower())
        if source:
            conditions.append(ProfileAttribute.source == source.strip().lower())
        if min_confidence is not None:
            conditions.append(ProfileAttribute.confidence >= min_confidence)

        statement = select(ProfileAttribute)
        count_statement = select(func.count()).select_from(ProfileAttribute)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = statement.order_by(ProfileAttribute.updated_at.desc())

        with Session(manager.engine) as session:
            total_count = int(session.exec(count_statement).one() or 0)
            rows = session.exec(statement.offset(offset).limit(limit)).all()

        return ProfileAttributeListResponse(
            attributes=[attribute_to_response(attr) for attr in rows],
            total_count=total_count,
            filters={
                "entity_type": normalized_entity_type,
                "entity_id": entity_id,
                "category": category,
                "key": key,
                "source": source,
                "min_confidence": min_confidence,
                "limit": limit,
                "offset": offset,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, "Failed to list profile attributes") from e


@router.get("/v1/profiles/lookup", tags=["Profiles"])
def lookup_profile_by_email(email: str = Query(..., description="Email address to look up")):
    """Look up a user profile by email address. Returns entity_id mapping."""
    service = get_service()

    try:
        
        manager = get_profile_manager(service.config)
        profile = manager.get_profile_by_email(email)
        if not profile:
            return {"found": False, "email": email, "entity_id": None}

        return {
            "found": True,
            "email": email,
            "entity_id": profile.entity_id,
            "display_name": profile.display_name,
        }
    except Exception as e:
        raise internal_error(e, f"Failed to look up profile by email '{email}'") from e


@router.get("/v1/profiles/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
def get_profile(entity_id: str):
    """Get one profile by ID (auto-detect user vs bot)."""
    service = get_service()

    try:
        from ...profiles import EntityType

        manager = get_profile_manager(service.config)

        user_profile = manager.get_profile(EntityType.USER, entity_id)
        if user_profile:
            return _get_profile_detail(manager, EntityType.USER, "user", entity_id)

        bot_profile = manager.get_profile(EntityType.BOT, entity_id)
        if bot_profile:
            return _get_profile_detail(manager, EntityType.BOT, "bot", entity_id)

        raise HTTPException(status_code=404, detail=f"No profile found for entity '{entity_id}'")
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, f"Failed to get profile '{entity_id}'") from e


@router.get("/v1/profiles/{entity_type}/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
def get_typed_profile(entity_type: str, entity_id: str):
    """Get one profile by explicit entity type and ID."""
    service = get_service()

    try:
        from ...profiles import EntityType

        normalized_entity_type = _normalize_entity_type(entity_type)
        if normalized_entity_type is None:
            raise HTTPException(status_code=400, detail="entity_type is required")

        manager = get_profile_manager(service.config)
        target_type = EntityType.USER if normalized_entity_type == "user" else EntityType.BOT
        return _get_profile_detail(manager, target_type, normalized_entity_type, entity_id)
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, f"Failed to get {entity_type} profile '{entity_id}'") from e


@router.patch("/v1/profiles/attribute/{attribute_id}", response_model=UserProfileAttribute, tags=["Profiles"])
def update_profile_attribute(attribute_id: int, request: ProfileAttributeUpdateRequest):
    """Update a profile attribute by its ID."""
    service = get_service()

    try:
        
        if request.value is None and request.confidence is None and request.source is None:
            raise HTTPException(status_code=400, detail="Provide at least one field: value, confidence, or source")

        manager = get_profile_manager(service.config)
        updated = manager.update_attribute_by_id(
            attribute_id=attribute_id,
            value=request.value,
            confidence=request.confidence,
            source=request.source,
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"Attribute with ID {attribute_id} not found")

        return attribute_to_response(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, "Failed to update profile attribute") from e


@router.patch("/v1/profiles/{entity_type}/{entity_id}", response_model=ProfileDetail, tags=["Profiles"])
def update_typed_profile(entity_type: str, entity_id: str, request: ProfileUpdateRequest):
    """Update core profile fields by explicit entity type and ID."""
    service = get_service()

    try:
        from ...profiles import EntityType

        normalized_entity_type = _normalize_entity_type(entity_type)
        if normalized_entity_type is None:
            raise HTTPException(status_code=400, detail="entity_type is required")

        target_type = EntityType.USER if normalized_entity_type == "user" else EntityType.BOT
        manager = get_profile_manager(service.config)
        avatar_render = None
        if request.avatar is not None:
            from ...media.avatar import resolve_avatar_render

            avatar_render = resolve_avatar_render(request.avatar) or ""

        profile = manager.update_profile(
            target_type,
            entity_id,
            display_name=request.display_name,
            description=request.description,
            summary=request.summary,
            email=request.email,
            color=request.color,
            avatar=request.avatar,
            avatar_render=avatar_render,
            is_service=request.is_service,
        )

        if not profile:
            manager.get_or_create_profile(target_type, entity_id)
            profile = manager.update_profile(
                target_type,
                entity_id,
                display_name=request.display_name,
                description=request.description,
                summary=request.summary,
                email=request.email,
                color=request.color,
                avatar=request.avatar,
                avatar_render=avatar_render,
                is_service=request.is_service,
            )

        if not profile:
            raise HTTPException(status_code=500, detail="Failed to update profile")

        return _get_profile_detail(manager, target_type, normalized_entity_type, entity_id)
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, f"Failed to update {entity_type} profile '{entity_id}'") from e


@router.post("/v1/profiles/attribute", response_model=UserProfileAttribute, tags=["Profiles"])
def upsert_profile_attribute(request: ProfileAttributeUpsertRequest):
    """Create or update a profile attribute by entity identity and key."""
    service = get_service()

    try:
        from ...profiles import EntityType

        manager = get_profile_manager(service.config)
        entity_type = EntityType.USER if request.entity_type == "user" else EntityType.BOT
        attr = manager.set_attribute(
            entity_type=entity_type,
            entity_id=request.entity_id,
            category=request.category,
            key=request.key,
            value=request.value,
            confidence=request.confidence,
            source=request.source,
        )
        return attribute_to_response(attr)
    except Exception as e:
        raise internal_error(e, "Failed to upsert profile attribute") from e


@router.delete("/v1/profiles/attribute/{attribute_id}", tags=["Profiles"])
def delete_profile_attribute(attribute_id: int):
    """Delete a profile attribute by its ID."""
    service = get_service()

    try:
        
        manager = get_profile_manager(service.config)

        attr = manager.get_attribute_by_id(attribute_id)
        if not attr:
            raise HTTPException(status_code=404, detail=f"Attribute with ID {attribute_id} not found")

        success = manager.delete_attribute_by_id(attribute_id)
        if success:
            return {
                "success": True,
                "message": f"Deleted attribute {attr.category}.{attr.key} from {attr.entity_type}/{attr.entity_id}",
                "deleted": {
                    "id": attribute_id,
                    "entity_type": str(attr.entity_type),
                    "entity_id": attr.entity_id,
                    "category": attr.category,
                    "key": attr.key,
                    "value": attr.value,
                },
            }

        raise HTTPException(status_code=500, detail="Failed to delete attribute")
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, "Failed to delete profile attribute") from e
