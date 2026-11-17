"""Retry evidenced XACKs only; pending commands are never dispatched here.

ACK_READY records original transport disposition, not SDK acceptance, terminal
publication, exit proof or PostgreSQL delivery settlement. No TTL turns missing
evidence into execution rights. Source/group loss remains retained uncertainty.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass

from .command_ack_scripts import MINT_ACK, SETTLE_ACK

log = logging.getLogger(__name__)
_REASON = "claude_run_send_finally_v1"


def source_digest(fields: dict[str, str]) -> str:
    chunks = []
    for key, value in sorted(fields.items()):
        key_bytes, value_bytes = key.encode(), value.encode()
        chunks.append(
            str(len(key_bytes)).encode() + b":" + key_bytes
            + str(len(value_bytes)).encode() + b":" + value_bytes
        )
    return hashlib.sha1(b"".join(chunks)).hexdigest()


def receipt_key(stream: str, group: str, entry_id: str) -> str:
    identity = json.dumps([stream, group, entry_id], separators=(",", ":"))
    return "agent:command_ack:" + hashlib.sha256(identity.encode()).hexdigest()


SOURCE_MISSING = -3


@dataclass(frozen=True)
class CommandAckContext:
    stream: str
    group: str
    entry_id: str
    evidence: str

    @property
    def key(self) -> str:
        return receipt_key(self.stream, self.group, self.entry_id)


class CommandAckOwner:
    def __init__(
        self, redis, *, stream: str, group: str, backend: str, boot_id: str,
        page_size: int = 64, interval: float = 5, timeout: float = 5,
        started_ms: int | None = None, clock=time.time,
    ):
        self.redis, self.stream, self.group = redis, stream, group
        self.backend, self.boot_id = backend, boot_id
        self.page_size, self.interval, self.timeout = page_size, interval, timeout
        self.cursor = "-"
        self._cycle_end: str | None = None
        self.task: asyncio.Task | None = None
        # Read-only retained-entry inventory. Classification never grants an
        # ACK or a dispatch; it only says why an entry is still pending.
        self.clock = clock
        self.started_ms = int(clock() * 1000) if started_ms is None else started_ms
        self._cycle_inventory = self._empty_inventory()
        self.last_inventory: dict | None = None

    def capture(self, fields: dict[str, str], entry_id: str) -> CommandAckContext:
        """Synchronous snapshot, never an admission await or execution grant."""
        if (fields.get("action") != "chat.send" or fields.get("backend") != self.backend
                or not fields.get("request_id") or not self.boot_id):
            raise ValueError("Unsupported original command acknowledgment context")
        evidence = json.dumps({
            "version": 1, "state": "ACK_READY", "stream": self.stream,
            "group": self.group, "entry_id": entry_id, "action": "chat.send",
            "backend": self.backend, "request_id": fields["request_id"],
            "session_key": fields.get("session_key", ""),
            "house_epoch": fields.get("house_epoch", ""), "boot_id": self.boot_id,
            "source_digest": source_digest(fields), "reason": _REASON,
        }, sort_keys=True, separators=(",", ":"))
        return CommandAckContext(self.stream, self.group, entry_id, evidence)

    async def _eval(self, script: str, context: CommandAckContext) -> int:
        async with asyncio.timeout(self.timeout):
            return int(await self.redis.eval(
                script, 2, context.stream, context.key,
                context.group, context.entry_id, context.evidence,
            ))

    async def acknowledge(self, context: CommandAckContext) -> None:
        """Mint at the original disposal point, then attempt only transport ACK."""
        if self._load(context.entry_id, context.evidence) != context:
            raise ValueError("Unsupported original command acknowledgment context")
        minted = await self._eval(MINT_ACK, context)
        if minted == SOURCE_MISSING:
            # A trimmed entry can never be redelivered, so the original handler
            # keeps its pre-existing direct XACK authority. Recovery never does.
            async with asyncio.timeout(self.timeout):
                await self.redis.xack(self.stream, self.group, context.entry_id)
            return
        # Even an already-disposed entry may have an exact receipt to retire.
        if minted not in {1, 2} or not await self.settle(context):
            raise RuntimeError("Original command acknowledgment evidence conflicted")

    async def settle(self, context: CommandAckContext) -> bool:
        if self._load(context.entry_id, context.evidence) != context:
            return False
        return await self._eval(SETTLE_ACK, context) in {1, 2}

    def _load(self, entry_id: str, evidence: str) -> CommandAckContext | None:
        try:
            data = json.loads(evidence)
        except (TypeError, ValueError):
            return None
        names = {
            "version", "state", "stream", "group", "entry_id", "action", "backend",
            "request_id", "session_key", "house_epoch", "boot_id", "source_digest", "reason",
        }
        if (not isinstance(data, dict) or set(data) != names
                or type(data["version"]) is not int or data["version"] != 1
                or any(not isinstance(data[name], str) for name in names - {"version"})):
            return None
        if (data["state"] != "ACK_READY" or data["stream"] != self.stream
                or data["group"] != self.group or data["entry_id"] != entry_id
                or data["backend"] != self.backend or data["action"] != "chat.send"
                or data["reason"] != _REASON or not data["request_id"] or not data["boot_id"]
                or len(data["source_digest"]) != 40
                or any(char not in "0123456789abcdef" for char in data["source_digest"])):
            return None
        return CommandAckContext(self.stream, self.group, entry_id, evidence)

    async def scan(self) -> dict[str, int]:
        """One bounded page; unknown heads cannot hide later exact receipts."""
        if self._cycle_end is None:
            async with asyncio.timeout(self.timeout):
                summary = await self.redis.xpending(self.stream, self.group)
            self._cycle_end = summary["max"]
            if self._cycle_end is None:
                self.cursor = "-"
                self._publish_inventory()
                return {"examined": 0, "settled": 0, "retained": 0}
        async with asyncio.timeout(self.timeout):
            entries = await self.redis.xpending_range(
                self.stream, self.group, self.cursor, self._cycle_end, self.page_size,
            )
        counts = {"examined": len(entries), "settled": 0, "retained": 0}
        for entry in entries:
            entry_id = entry["message_id"]
            self.cursor = f"({entry_id}"
            async with asyncio.timeout(self.timeout):
                evidence = await self.redis.get(receipt_key(self.stream, self.group, entry_id))
            context = self._load(entry_id, evidence) if evidence else None
            if context is not None and await self.settle(context):
                counts["settled"] += 1
            else:
                counts["retained"] += 1
                self._classify(entry)
        if len(entries) < self.page_size:
            self.cursor = "-"
            self._cycle_end = None
            self._publish_inventory()
        return counts

    @staticmethod
    def _empty_inventory() -> dict:
        return {"legacy": 0, "uncertain": 0, "oldest_legacy_id": None, "max_deliveries": 0}

    def _classify(self, entry: dict) -> None:
        """Legacy = last delivered before this process booted, so no handler
        here can own it. Uncertain = delivered this boot with no receipt (a
        live run or a stranded disposal). Neither is evidence of nonexecution."""
        inv = self._cycle_inventory
        delivered_ms = int(self.clock() * 1000) - int(entry.get("time_since_delivered") or 0)
        if delivered_ms < self.started_ms:
            inv["legacy"] += 1
            if inv["oldest_legacy_id"] is None:
                inv["oldest_legacy_id"] = entry["message_id"]
        else:
            inv["uncertain"] += 1
        inv["max_deliveries"] = max(inv["max_deliveries"], int(entry.get("times_delivered") or 0))

    def _publish_inventory(self) -> None:
        inventory, self._cycle_inventory = self._cycle_inventory, self._empty_inventory()
        if inventory != self.last_inventory:
            level = logging.WARNING if inventory["legacy"] else logging.INFO
            log.log(level, "Command PEL retained without receipt (no ACK/replay): %s", inventory)
        self.last_inventory = inventory

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run(), name="commands:ack-settlement")

    async def _run(self) -> None:
        delay = self.interval
        while True:
            try:
                counts = await self.scan()
                if counts["examined"]:
                    log.info("Command ACK settlement: %s", counts)
                delay = self.interval
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("Command ACK settlement unavailable; retaining PEL", exc_info=True)
                delay = min(30, max(self.interval, delay * 2))
            # Pace empty/full successful pages too, not only transport failures.
            await asyncio.sleep(delay)

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
