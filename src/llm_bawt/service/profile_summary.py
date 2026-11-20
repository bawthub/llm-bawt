"""On-demand profile summaries, separate from background candidate curation."""

import json
import re
import threading
import time

from fastapi import HTTPException

from ..memory.extraction.prompts import get_profile_consolidation_prompt
from ..models.message import Message
from ..profiles import EntityType
from ..runtime_settings import resolve_job_model
from .dependencies import get_profile_manager
from .logging import get_service_logger
from .schemas import ProfileSummaryRegenerated
from .state_events import PROFILE_SUMMARY, mark

log = get_service_logger(__name__)
_slot = threading.Lock()

_LABELS = {
    "identity": "Identity",
    "preferences": "Preferences",
    "interests": "Interests",
    "context": "Current context",
}


def _format_summary(raw: str) -> str:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), re.IGNORECASE | re.DOTALL)
    fields = json.loads(fenced[1] if fenced else raw)
    if not isinstance(fields, dict):
        raise ValueError("Expected a summary object")
    lines = []
    for key, label in {"name": "Name", **_LABELS}.items():
        value = fields.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("Expected text summary fields")
        value = value.strip()
        if value and value.lower() != "null":
            lines.append(f"{label}: {value}")
    summary = "\n".join(lines)
    if not summary or len(summary) > 4_000:
        raise ValueError("Empty or oversized profile summary")
    return summary


def regenerate_summary(service, entity_id: str) -> ProfileSummaryRegenerated:
    """Run in the route's worker thread; accept one model call at a time per app."""
    if not _slot.acquire(blocking=False):
        raise HTTPException(409, "A profile summary is already being regenerated. Try again shortly.")
    try:
        deadline = time.monotonic() + 45
        manager = get_profile_manager(service.config)
        snapshot = manager.get_profile(EntityType.USER, entity_id)
        if not snapshot or snapshot.is_service:
            raise HTTPException(404, "User profile not found.")

        facts = manager.get_profile_summary(EntityType.USER, snapshot.entity_id, use_summary=False)
        if not facts:
            raise HTTPException(409, "There are no saved profile details to summarize yet.")
        requested_model = resolve_job_model(service.config, "profile_maintenance_model") or resolve_job_model(
            service.config, "maintenance_model"
        )
        if not requested_model:
            raise HTTPException(503, "No profile summary model is configured.")
        client, _ = service._get_background_client(
            model_override=requested_model, job="profile_summary", allow_fallback=True
        )
        if client is None:
            raise HTTPException(503, "The configured profile summary model is unavailable.")

        raw = client.query_bounded(
            [Message(role="user", content=get_profile_consolidation_prompt(facts))],
            timeout_seconds=18, max_tokens=800, temperature=0.2,
        )
        summary = _format_summary(raw)
        if time.monotonic() >= deadline:
            raise HTTPException(504, "Profile summary generation timed out.")
        saved_at = manager.save_regenerated_summary(snapshot, summary)
        if saved_at is None:
            raise HTTPException(409, "Your profile summary changed during generation. Refresh before trying again.")
        mark(service, PROFILE_SUMMARY)
        return ProfileSummaryRegenerated(entity_id=snapshot.entity_id, summary=summary, summary_updated_at=saved_at)
    except HTTPException:
        raise
    except Exception as exc:
        # Provider and database exceptions can contain profile data. Don't log the payload.
        log.warning("Profile summary regeneration failed (%s)", type(exc).__name__)
        raise HTTPException(502, "Could not regenerate the profile summary. The last saved summary is unchanged.") from None
    finally:
        _slot.release()
