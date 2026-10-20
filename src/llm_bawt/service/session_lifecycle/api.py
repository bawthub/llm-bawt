"""Thin HTTP adapters for an app-owned, receipt-backed new session."""
from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException

from ...message_authorship import AuthorReference
from .intent import (
    Admission, Continuity, NewSessionIntent, Origin, Outcome,
)
from .lifecycle import NewSessionLifecycle


class NewSessionAPI:
    """Map public reset requests to a durable intent, never rotate in the route."""

    @staticmethod
    async def submit(
        service, *, bot_id: str, user_id: str, continuity: Continuity,
        origin: Origin, idempotency_key: str | None = None,
        reason: str | None = None,
    ) -> dict:
        key = (idempotency_key or "").strip() or f"api-new:{uuid4()}"
        intent = NewSessionIntent(
            bot_id=bot_id, user_id=user_id, author=AuthorReference.user(user_id),
            continuity=continuity, admission=Admission.SKIP,
            origin=origin, idempotency_key=key, reason=reason,
        )
        try:
            result = await NewSessionLifecycle(service).submit(intent)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        if result.outcome is not Outcome.COMPLETED:
            code = {Outcome.SKIPPED: 409, Outcome.FAILED: 503, Outcome.AMBIGUOUS: 504}[result.outcome]
            raise HTTPException(status_code=code, detail={
                "outcome": result.outcome.value, "stage": result.stage.value,
                "delivery_id": result.delivery_id, "turn_id": result.turn_id,
                "reason": result.detail or "reset did not complete",
            })
        return {
            "ok": True, "reset": True, "outcome": result.outcome.value,
            "bot_id": bot_id, "user_id": user_id,
            "old_session_id": result.old_session_id,
            "new_session_id": result.new_session_id,
            "receipt_message_id": result.receipt_message_id,
            "delivery_id": result.delivery_id, "turn_id": result.turn_id,
        }
