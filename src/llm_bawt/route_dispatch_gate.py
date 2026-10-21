"""Recheck route consent and bound model at the target-locked claim boundary."""
from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import text

from .bot_types import durable_agent_backend
from .bots import BotManager
from .service.request_model_execution import resolve_execution_endpoint


def check_route_claim(conn, candidate: dict, config) -> dict | None:
    """Return a claimable row or cancel an unaccepted/revoked route.

    The caller holds the target advisory lock; eligibility writes take it too.
    An accepted bridge attempt is never cancelled or re-targeted here.
    """
    meta = candidate.get("metadata_json") or {}
    route = meta.get("message_route")
    if not isinstance(route, dict):
        return candidate
    if candidate.get("transport_accepted_at") is not None:
        return candidate
    owner, target = route.get("owner_user_id"), candidate["target_bot_id"]
    receipt = conn.execute(text("""
        SELECT selected_bot_id, owner_user_id, model_ref, model_endpoint_id,
               delivery_id, session_policy, state FROM message_routes WHERE id=:id
    """), {"id": route.get("route_id")}).mappings().first()
    payload = candidate.get("payload_json") or {}
    from .service.message_routing import route_opted_in

    reason = None
    if not route_opted_in(conn, owner, target):
        reason = "Routing opt-in was revoked before dispatch"
    elif (not receipt or receipt["delivery_id"] != candidate["id"]
          or receipt["selected_bot_id"] != target or receipt["owner_user_id"] != owner
          or receipt["model_ref"] != route.get("model_ref")
          or receipt["model_endpoint_id"] != route.get("model_endpoint_id")
          or receipt["session_policy"] != candidate.get("session_policy")
          or receipt["state"] != "queued"
          or payload.get("model") != receipt["model_ref"]
          or payload.get("user") != owner or payload.get("bot_id") != target
          or payload.get("prefer_steer") is not False):
        reason = "Routed delivery no longer matches its immutable receipt"
    else:
        bot = BotManager(config).get_bot(target)
        if not bot or not getattr(bot, "enabled", True) or not durable_agent_backend(bot):
            reason = "Routed target is disabled or unsupported"
        else:
            try:
                bound = resolve_execution_endpoint(SimpleNamespace(config=config),
                                                   target, receipt["model_ref"], owner)
                if (bound != receipt["model_ref"] or
                        config.ensure_model_catalog().resolve_endpoint(
                            bound, harness=bot.harness,
                        ).id != receipt["model_endpoint_id"]):
                    reason = "Routed endpoint changed"
                if receipt["session_policy"] == "continue":
                    from .service.route_session_binding import check_session_fence

                    binding = route.get("session_binding")
                    if not isinstance(binding, dict) or payload.get("session_id") != binding.get("session_id"):
                        raise ValueError("Routed delivery lost its bound owner thread")
                    check_session_fence(conn, config, bot, owner, bound, binding)
                elif route.get("session_binding") is not None:
                    raise ValueError("Reset route unexpectedly carries a continuation binding")
            except ValueError as exc:
                reason = str(exc)
    if reason:
        from .inter_bot_delivery import InterBotDeliveryStore

        InterBotDeliveryStore.cancel_unaccepted_locked(conn, candidate, reason)
        return None
    return candidate
