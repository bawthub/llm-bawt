"""Model catalog + model-definition request/response schemas.

Split out of ``service/schemas.py`` (TASK-557). ``schemas.py`` re-imports every
name here so ``from ..schemas import X`` across the service is unchanged.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Per-1M-token USD rates for client-side cost estimation.

    All optional — a model with no pricing simply shows no computed cost.
    Stored in ``ModelDefinition.extra['pricing']`` (no schema migration) and
    surfaced here so the chat context badge can estimate turn cost as
    ``sum(tokens_of_kind * rate_of_kind) / 1_000_000``.
    """
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None

class RouteEligibility(BaseModel):
    """Whether a bot may be newly assigned to a route (``route_eligibility``)."""
    eligible: bool
    state: str
    reason: str | None = None

class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint.

    ``endpoint_id``/``access_path``/``eligibility`` label which route an alias
    rides and whether that route is usable now (TASK-1302 A3). They are absent
    for an alias with no catalog endpoint behind it.
    """
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-bawt"
    type: str | None = None
    model_id: str | None = None
    description: str | None = None
    pricing: ModelPricing | None = None
    endpoint_id: int | None = None
    access_path: str | None = None
    eligibility: RouteEligibility | None = None

class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: list[ModelInfo]

class ModelSwitchRequest(BaseModel):
    """Request to switch the active model."""
    model: str = Field(..., description="Model alias to switch to")

class ModelSwitchResponse(BaseModel):
    """Response from model switch."""
    success: bool
    message: str
    previous_model: str | None = None
    new_model: str | None = None

class ModelDetail(BaseModel):
    """Detailed model information."""
    id: str
    type: str | None = None
    model_id: str | None = None
    description: str | None = None
    current: bool = False
