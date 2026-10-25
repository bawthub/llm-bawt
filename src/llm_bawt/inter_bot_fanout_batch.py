"""One fan-out batch: resolve, rate limit, admit per target (TASK-1431 r3 §1.2).

Per-target, not all-or-nothing: a refusal for one target (unknown, private,
gated, …) is reported in ``refused`` and the rest still go. Expansion is read
once, at first submit, and recorded on every row (``metadata_json.batch``). A
retry of a keyed batch replays the recorded targets, so each returns its
original row with ``duplicate=true`` and never a second row or turn.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from .inter_bot_fanout_limit import (
    FanoutRateLimited, batch_id_for, batch_reader, sender_batch_lock,
)
from .inter_bot_send import DeliveryRefused, Principal
from .inter_bot_send_targets import (
    MAX_FANOUT_TARGETS, Expansion, SelectorError, TargetExpander, TargetSelector,
)
from .message_authorship import AuthorReference, normalize_author

if TYPE_CHECKING:
    from .inter_bot_send import BatchDeliveryCreate, DeliverySender


def _refusal_entry(target: str, status_code: int, detail: Any) -> dict[str, Any]:
    if isinstance(detail, dict):
        return {"target": target, "error": str(detail.get("error") or "refused"),
                "detail": detail.get("detail", detail), "status_code": status_code}
    return {"target": target, "error": "refused", "detail": str(detail), "status_code": status_code}


class FanoutBatch:
    def __init__(self, sender: DeliverySender, body: BatchDeliveryCreate, request: Any):
        self._sender = sender
        self._body = body
        self._request = request

    def _principal(self, sender: str) -> Principal:
        try:
            author = normalize_author(self._body.author_entity_type, self._body.author_entity_id)
        except ValueError as exc:
            raise DeliveryRefused(status_code=422, detail=str(exc)) from exc
        return Principal.of_author(author or AuthorReference.bot(sender))

    def _limits(self, principal: Principal, key: str) -> None:
        """Bot fan-out limits that need no database (design r3 §1.3)."""
        body = self._body
        if body.reply_mode == "waited":
            raise DeliveryRefused(status_code=422, detail={
                "error": "fanout_waited",
                "detail": "A fan-out never waits for replies; send to one bot to wait.",
            })
        if principal.kind != "bot":
            return
        if not key:
            raise DeliveryRefused(status_code=422, detail={
                "error": "fanout_key_required",
                "detail": "A bot's multi-target send requires an idempotency_key.",
            })
        if body.reset_requested:
            raise DeliveryRefused(status_code=422, detail={
                "error": "fanout_reset_forbidden",
                "detail": "A bot's multi-target send cannot reset sessions; use session_policy=continue.",
            })

    def _expand(self, selector: TargetSelector, sender: str, principal: Principal, engine) -> Expansion:
        from .bot_tags import BotTags
        from .inter_bot_send import BotManager

        snapshot = BotTags(engine).snapshot()
        if not snapshot.available:
            raise DeliveryRefused(status_code=503, detail={
                "error": "bot_tags_unavailable",
                "detail": "Bot tags could not be read, so no target can be resolved.",
            })
        expander = TargetExpander(BotManager(self._sender.service.config).list_bots(), snapshot)
        try:
            expander.validate_tags(selector)
            return expander.expand(
                selector, sender=sender, exclude=self._body.exclude,
                include_self=self._body.include_self, bot_caller=principal.kind == "bot",
            )
        except SelectorError as exc:
            raise DeliveryRefused(status_code=422, detail=str(exc)) from exc

    def _record(self, selector: TargetSelector, targets: tuple[str, ...]) -> dict[str, Any]:
        return {
            "selector": selector.to_json(),
            "exclude": sorted({slug.strip().lower() for slug in self._body.exclude}),
            "include_self": self._body.include_self,
            "targets": list(targets),
        }

    def _replay(self, recorded: dict[str, Any], selector: TargetSelector,
                expansion: Expansion) -> Expansion:
        """A retried key must name the same batch; it gets the recorded targets."""
        if {k: recorded.get(k) for k in ("selector", "exclude", "include_self")} != {
                k: v for k, v in self._record(selector, ()).items() if k != "targets"}:
            raise DeliveryRefused(status_code=409, detail={
                "error": "batch_conflict",
                "detail": "This idempotency_key already names a batch with different targets.",
            })
        targets = tuple(str(t) for t in recorded.get("targets") or ())
        return Expansion(targets, tuple(r for r in expansion.refused if r.target not in targets))

    async def run(self) -> tuple[int, dict[str, Any]]:
        body = self._body
        sender = body.sender_bot_id.strip().lower() or "unknown"
        key = (body.idempotency_key or "").strip()
        principal = self._principal(sender)
        self._limits(principal, key)
        try:
            selector = TargetSelector.parse(body.targets)
        except SelectorError as exc:
            raise DeliveryRefused(status_code=422, detail=str(exc)) from exc
        batch_id = batch_id_for(sender, key)
        engine = self._sender._dispatcher().store.engine
        if principal.kind == "bot":
            hold = sender_batch_lock(engine, sender)
        else:
            hold = batch_reader(engine) if key else nullcontext(None)
        deliveries: list[dict[str, Any]] = []
        refused: list[dict[str, Any]] = []
        with hold as ledger:
            recorded = ledger.recorded(sender, batch_id) if ledger is not None else None
            if recorded is None and principal.kind == "bot":
                try:
                    ledger.check(sender, batch_id)
                except FanoutRateLimited as exc:
                    raise DeliveryRefused(status_code=429, detail={
                        "error": "fanout_rate_limited", "detail": str(exc),
                    }) from exc
            expansion = self._expand(selector, sender, principal, engine)
            if recorded is not None:
                expansion = self._replay(dict(recorded), selector, expansion)
            if len(expansion.targets) > MAX_FANOUT_TARGETS:
                raise DeliveryRefused(status_code=422, detail={
                    "error": "too_many_targets",
                    "detail": f"{len(expansion.targets)} targets; a batch sends to at most "
                              f"{MAX_FANOUT_TARGETS}.",
                })
            refused.extend(_refusal_entry(r.target, r.status_code, {"error": r.error, "detail": r.detail})
                           for r in expansion.refused)
            batch = self._record(selector, expansion.targets)
            for target in expansion.targets:
                metadata = {**body.metadata, "batch_id": batch_id, "batch": batch}
                try:
                    _, receipt = await self._sender._admit_one(
                        body.for_target(target, metadata), self._request)
                except DeliveryRefused as exc:
                    refused.append(_refusal_entry(target, exc.status_code, exc.detail))
                    continue
                deliveries.append(receipt)
        return self._response(batch_id, deliveries, refused)

    def _response(self, batch_id: str, deliveries: list[dict[str, Any]],
                  refused: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        if not deliveries:
            outcomes = {(r["status_code"], repr(r["detail"]), r["error"]) for r in refused}
            if len(outcomes) == 1:
                only = refused[0]
                detail = only["detail"] if only["error"] == "refused" else {
                    "error": only["error"], "detail": only["detail"]}
                raise DeliveryRefused(status_code=only["status_code"], detail=detail)
            if not refused:
                raise DeliveryRefused(status_code=422, detail={
                    "error": "no_targets",
                    "detail": "The selector matched no bot that can receive this delivery.",
                })
        result = {
            "batch_id": batch_id,
            "mode": self._body.mode,
            "dispatched": any(bool(r.get("delivery_id")) and bool(r.get("success")) for r in deliveries),
            "deliveries": deliveries,
            "refused": [{k: v for k, v in r.items() if k != "status_code"} for r in refused],
        }
        return (202 if deliveries else 422), result
