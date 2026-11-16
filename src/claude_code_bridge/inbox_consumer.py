"""One bridge drain owner per bot; active clients, not session send locks.

The app still owns idle turns. This consumer is only attached for a run whose
app-pinned inbox capability was explicitly carried in its send command.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agent_bridge.bridge_token import headers

from .inbox_input import accept_inbox_input

log = logging.getLogger(__name__)


class InboxAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def __call__(self, path: str, body: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/v1/internal/inbox{path}", json=body, headers=headers(),
            )
            response.raise_for_status()
            return response.json()


@dataclass(eq=False)
class InboxRun:
    bot_id: str
    user_id: str
    backend: str
    session_key: str
    run: Any
    trigger_message_id: str
    session_id: str

    def claim_body(self, owner: str) -> dict:
        return dict(
            bot_id=self.bot_id, user_id=self.user_id, backend=self.backend,
            session_key=self.session_key, request_id=self.run.request_id, claim_owner=owner,
        )

    def matches(self, evidence: dict) -> bool:
        binding = evidence.get("binding") or {}
        return all(binding.get(key) == value for key, value in {
            "bot_id": self.bot_id, "user_id": self.user_id, "backend": self.backend,
            "session_key": self.session_key, "request_id": self.run.request_id,
            "trigger_message_id": self.trigger_message_id, "session_id": self.session_id,
        }.items())


@dataclass
class BotInbox:
    runs: list[InboxRun] = field(default_factory=list)
    operations: dict[str, dict] = field(default_factory=dict)
    receipts: deque[dict] = field(default_factory=deque)
    repairs: dict[str, dict] = field(default_factory=dict)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class InboxConsumer:
    def __init__(self, api, queue, *, owner: str, fallback_seconds: float = 30):
        self.api, self.queue, self.owner = api, queue, owner
        self.fallback_seconds = fallback_seconds
        self.bots: dict[str, BotInbox] = {}

    def owns(self, context: InboxRun) -> bool:
        run, key = context.run, context.session_key
        return bool(
            self.queue.get_active_client(key) is run
            and not self.queue.is_tombstoned(key, run.request_id)
            and not self.queue.is_stopping(key, run.request_id)
            and not run._completed.is_set() and not run._finishing and not run.maintenance
            and run.lifecycle_supported
        )

    def attach(self, context: InboxRun) -> None:
        inbox = self.bots.setdefault(context.bot_id, BotInbox())
        if context not in inbox.runs:
            inbox.runs.append(context)
        inbox.wake.set()
        if inbox.task is None or inbox.task.done():
            inbox.task = asyncio.create_task(self._run(context.bot_id), name=f"inbox:{context.bot_id}")

    def detach(self, context: InboxRun) -> None:
        inbox = self.bots.get(context.bot_id)
        if inbox and context in inbox.runs:
            inbox.runs.remove(context)
            inbox.wake.set()

    def observe(self, context: InboxRun, payload: dict) -> dict:
        """Decorate the existing lifecycle stream, never add another SDK reader.

        The app settles this private evidence before projecting its existing
        public lifecycle whitelist. Claim/write tokens must not enter that view.
        """
        inbox = self.bots.get(context.bot_id)
        operation = inbox.operations.get(payload.get("message_id")) if inbox else None
        if (operation is None or payload.get("origin") != "bot"
                or payload.get("run_request_id") != context.run.request_id
                or payload.get("steer_rpc_id") != operation["steer_rpc_id"]
                or not context.matches(operation["evidence"])):
            return payload
        receipt = {
            "delivery_id": operation["delivery_id"], "evidence": operation["evidence"],
            "state": payload["state"], "proof": payload.get("proof"),
        }
        inbox.receipts.append(receipt)
        inbox.wake.set()
        return {**payload, "inbox_write": {
            "delivery_id": operation["delivery_id"], "evidence": operation["evidence"],
        }}

    async def _flush(self, inbox: BotInbox) -> None:
        while inbox.receipts:
            receipt = inbox.receipts[0]
            result = await self.api("/lifecycle", receipt)
            if result.get("recorded") and receipt["state"] in {"queued", "started", "completed"}:
                command = receipt["evidence"]["command_uuid"]
                operation = inbox.operations.get(command)
                if operation:
                    inbox.repairs[operation["delivery_id"]] = operation
            elif result.get("recorded") and receipt["state"] in {"refused", "discarded", "cancelled", "uncertain"}:
                inbox.repairs.pop(receipt["delivery_id"], None)
            inbox.receipts.popleft()
        for delivery_id, operation in list(inbox.repairs.items()):
            result = await self.api("/repair", {
                "delivery_id": delivery_id, "claim_token": operation["claim_token"],
            })
            if result.get("acknowledged"):
                inbox.repairs.pop(delivery_id, None)

    async def drain(self, bot_id: str) -> bool:
        """One bounded pass, also used by deterministic provider-free tests."""
        inbox = self.bots[bot_id]
        await self._flush(inbox)
        recovered = (await self.api("/repair/claim", {
            "bot_id": bot_id, "claim_owner": self.owner,
        })).get("claim")
        if recovered:
            inbox.repairs[recovered["delivery_id"]] = recovered
            await self._flush(inbox)
        eligible = [context for context in inbox.runs if self.owns(context)]
        if len(eligible) != 1:
            return False
        context = eligible[0]
        operation = (await self.api("/claim", context.claim_body(self.owner))).get("claim")
        if operation is None:
            return False
        evidence = operation["evidence"]
        if not context.matches(evidence):
            return False  # Never prepare a response naming a different run/thread.
        command = evidence["command_uuid"]
        if command in inbox.operations:
            return False  # A logical operation is invoked at most once by this owner.
        inbox.operations[command] = operation

        async def prepare():
            result = await self.api("/prepare", {
                "delivery_id": operation["delivery_id"], "claim_token": operation["claim_token"],
                "binding": evidence["binding"],
            })
            return result.get("authorized") is True

        receipt = await accept_inbox_input(
            context.run, command_uuid=command, steer_rpc_id=operation["steer_rpc_id"],
            framed_message=operation["framed_message"], interrupt_item=operation["interrupt_item"],
            owns_run=lambda: self.owns(context), prepare_write=prepare,
        )
        # not_written is NOT a generic release: denied preparation may mean an
        # earlier WRITING operation exists. Until invocation-proof release is
        # implemented, leave it to the conservative durable phase fence.
        if receipt.outcome != "not_written":
            state = receipt.acceptance if receipt.outcome == "accepted" else (
                receipt.state if receipt.outcome == "rejected" else "uncertain"
            )
            inbox.receipts.append({
                "delivery_id": operation["delivery_id"], "evidence": evidence,
                "state": state, "proof": receipt.proof,
            })
        await self._flush(inbox)
        return True

    async def _run(self, bot_id: str) -> None:
        inbox = self.bots[bot_id]
        while True:
            inbox.wake.clear()
            try:
                if await self.drain(bot_id):
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transport failure never retries SDK input. Only durable
                # evidence/history repair is retried on the next wake.
                log.exception("Inbox pass failed: bot=%s", bot_id)
            try:
                await asyncio.wait_for(inbox.wake.wait(), timeout=self.fallback_seconds)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        tasks = [inbox.task for inbox in self.bots.values() if inbox.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
