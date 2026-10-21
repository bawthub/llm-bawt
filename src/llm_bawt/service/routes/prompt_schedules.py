"""User-owned prompt schedule API; ordinary LAN trust and UI identity boundary."""
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..dependencies import get_profile_manager, get_service
from ..prompt_capabilities import resolve_prompt_target
from ..prompt_schedule_management import PromptManagement
from ..prompt_timing import PromptTiming, resolve_local

async def publish_change(user, identifier):
    import logging
    try:
        service = get_service()
        subscriber = getattr(service, "_redis_subscriber", None)
        if subscriber is not None:
            await subscriber.publish_tool_event("system", user, {
                "_type": "prompt_schedule_changed", "user_id": user,
                "schedule_id": identifier, "bot_id": "system",
            })
    except Exception:
        logging.getLogger(__name__).warning("Schedule change event unavailable", exc_info=True)


async def notify_mutations(request: Request, background: BackgroundTasks):
    yield
    if request.method in {"POST", "PATCH", "DELETE"} and request.url.path.rsplit("/", 1)[-1] not in {"preview", "resolve-local"}:
        background.add_task(publish_change, owner(request.query_params.get("user")), request.path_params.get("identifier"))


router = APIRouter(prefix="/v1/prompt-schedules", tags=["Prompt schedules"],
                   dependencies=[Depends(notify_mutations)])


class ScheduleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    prompt: str = Field(min_length=1, max_length=100000)
    bot_id: str | None = Field(default=None, max_length=128)
    target_mode: Literal["specific", "auto"] = "specific"
    candidate_bot_ids: list[str] | None = None
    requested_model: str | None = Field(default=None, max_length=256)
    timing: PromptTiming
    clear_context: bool = True
    augment_memory: bool = True
    extract_memory: bool = True
    enabled: bool = True
    missed_policy: str = Field(default="run_latest", pattern="^(skip|run_latest)$")
    misfire_grace_seconds: int = Field(default=3600, ge=60, le=604800, strict=True)

    @field_validator("name", "prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def valid_target(self):
        if self.target_mode == "specific":
            if not self.bot_id or not self.bot_id.strip() or self.bot_id == "*":
                raise ValueError("Specific schedules require a concrete bot_id")
            self.bot_id = self.bot_id.strip().lower()
            if self.candidate_bot_ids is not None:
                raise ValueError("candidate_bot_ids requires automatic targeting")
        else:
            if self.bot_id not in {None, ""}:
                raise ValueError("Automatic schedules must not pin bot_id")
            self.bot_id = None
            if self.candidate_bot_ids is not None:
                normalized = [slug.strip().lower() for slug in self.candidate_bot_ids]
                if len(normalized) != len(set(normalized)) or any(not slug for slug in normalized):
                    raise ValueError("candidate_bot_ids must be unique, nonblank IDs")
                self.candidate_bot_ids = normalized
        return self


class CreateSchedule(ScheduleInput):
    idempotency_key: UUID


class Preview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timing: PromptTiming
    after: AwareDatetime | None = None
    count: int = Field(default=5, ge=1, le=10)


class LocalTime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local: datetime
    timezone: str
    offset_minutes: int | None = Field(default=None, ge=-1440, le=1440)


class RunNow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: UUID


def owner(user):
    value = (user or "").strip().lower()
    if not value:
        raise HTTPException(422, "An owning user is required")
    return value


def management():
    service = get_service()
    return PromptManagement(get_profile_manager(service.config).engine)


def validate_target(bot_id, model, user, target_mode="specific", candidate_bot_ids=None):
    try:
        if target_mode == "specific":
            resolve_prompt_target(get_service(), bot_id, model, user)
        else:
            from ..message_routing import MessageRouter, RouteRequest

            preview = MessageRouter(get_service()).preview(RouteRequest(
                user_id=user, candidate_bot_ids=candidate_bot_ids,
                requested_model=model, scope="schedule",
            ))
            if preview["selected_bot_id"] is None:
                raise ValueError("No opted-in compatible bots for this automatic schedule")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def validation_error(exc):
    # Pydantic errors may contain ValueError in ctx; keep the response JSON-safe.
    if isinstance(exc, ValidationError):
        return HTTPException(422, [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()])
    return HTTPException(422, str(exc))


@router.post("/preview")
def preview(body: Preview, user: str = Query(...)):
    owner(user)
    try:
        occurrences = body.timing.preview(body.after or datetime.now(timezone.utc), body.count)
    except ValueError as exc:
        raise validation_error(exc) from exc
    return {"occurrences": occurrences, "timezone": body.timing.timezone,
            "dst_policy": "Skip nonexistent local times; repeated minutes run once at the first offset."}


@router.post("/resolve-local")
def local_time(body: LocalTime, user: str = Query(...)):
    owner(user)
    try:
        instant = resolve_local(body.local, body.timezone, body.offset_minutes)
    except (ValueError, KeyError) as exc:
        offsets = []
        if "ambiguous" in str(exc):
            tz = ZoneInfo(body.timezone)
            offsets = sorted({int(body.local.replace(tzinfo=tz, fold=f).utcoffset().total_seconds() // 60) for f in (0, 1)})
        raise HTTPException(422, {"message": str(exc), "offset_choices": offsets}) from exc
    return {"utc": instant.isoformat()}


@router.get("")
def list_schedules(user: str = Query(...), bot_id: str | None = None, lifecycle: str | None = None,
                   search: str | None = Query(None, max_length=200), limit: int = Query(50, ge=1, le=100),
                   offset: int = Query(0, ge=0)):
    return management().list(owner(user), bot_id=bot_id, lifecycle=lifecycle, search=search, limit=limit, offset=offset)


@router.post("", status_code=201)
def create(body: CreateSchedule, user: str = Query(...)):
    user = owner(user)
    validate_target(body.bot_id, body.requested_model, user,
                    body.target_mode, body.candidate_bot_ids)
    return management().create(user, body)


@router.get("/{identifier}")
def get(identifier: str, user: str = Query(...)):
    return management().get(owner(user), identifier)


@router.patch("/{identifier}")
def patch(identifier: str, body: dict, user: str = Query(...)):
    user = owner(user)
    if type(body.get("expected_revision")) is not int:
        raise HTTPException(422, "expected_revision is required")
    current = management().get(user, identifier)
    if set(body) - set(ScheduleInput.model_fields) - {"expected_revision"}:
        raise HTTPException(422, "Unknown schedule fields")
    try:
        merged = {key: current[key] for key in ScheduleInput.model_fields}
        merged.update({key: value for key, value in body.items() if key != "expected_revision"})
        parsed = ScheduleInput.model_validate(merged)
        validate_target(parsed.bot_id, parsed.requested_model, user,
                        parsed.target_mode, parsed.candidate_bot_ids)
        return management().update(user, identifier, dict(body), ScheduleInput)
    except ValueError as exc:
        raise validation_error(exc) from exc


@router.post("/{identifier}/pause")
def pause(identifier: str, user: str = Query(...)):
    return management().lifecycle(owner(user), identifier, "pause")


@router.post("/{identifier}/resume")
def resume(identifier: str, user: str = Query(...)):
    return management().lifecycle(owner(user), identifier, "resume")


@router.delete("/{identifier}")
def cancel(identifier: str, user: str = Query(...)):
    return management().lifecycle(owner(user), identifier, "cancel")


@router.post("/{identifier}/run-now", status_code=202)
def run_now(identifier: str, body: RunNow, user: str = Query(...)):
    user = owner(user)
    current = management().get(user, identifier)
    validate_target(current["bot_id"], current["requested_model"], user,
                    current["target_mode"], current["candidate_bot_ids"])
    return management().run_now(user, identifier, str(body.idempotency_key))


@router.get("/{identifier}/runs")
def runs(identifier: str, user: str = Query(...), limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    return management().runs(owner(user), identifier, limit, offset)


@router.post("/{identifier}/runs/{run_id}/cancel", status_code=202)
def cancel_run(identifier: str, run_id: str, user: str = Query(...)):
    return management().cancel_run(owner(user), identifier, run_id)
