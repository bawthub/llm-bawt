"""Redis consumer-group reads that survive the group disappearing (TASK-684).

Listeners used to create their group once at startup and then loop on
XREADGROUP. If the group (or the whole stream) vanished, every read raised
NOGROUP, the loop logged and slept, and it never recovered: tens of
thousands of errors until someone restarted the bridge.

``GroupReader`` recreates the group when Redis says it is gone and resumes
from the last entry this reader saw, so it neither replays the stream's
history nor skips entries added after that point. Entries that were
delivered but unacked when the group vanished are lost with the group; that
is Redis' state, not something a reader can rebuild.
"""

from __future__ import annotations

import logging
from typing import Any

from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)


def is_missing_group(exc: BaseException) -> bool:
    """Redis proved the stream (``no such key``) or group (``NOGROUP``) absent."""
    if not isinstance(exc, ResponseError):
        return False
    msg = str(exc)
    return "no such key" in msg.lower() or msg.startswith("NOGROUP")


# Equals ui_stream.LEGACY_UI_GROUP_PREFIX (ui_stream imports this module, so it
# can't be imported here); a test pins the two together.
RETIRED_UI_GROUP_PREFIX = "ui:"


async def ensure_group(redis: Any, stream: str, group: str, *, start_id: str) -> None:
    """Create ``group`` on ``stream`` (and the stream) unless it already exists.

    Refuses ``ui:*`` names: the per-window groups are retired (TASK-1582), and a
    recreating reader must not bring them back.
    """
    if group.startswith(RETIRED_UI_GROUP_PREFIX):
        raise ValueError(f"refusing to create retired UI consumer group {group!r}")
    try:
        await redis.xgroup_create(stream, group, id=start_id, mkstream=True)
    except ResponseError as exc:
        if not str(exc).startswith("BUSYGROUP"):
            raise


class GroupReader:
    """XREADGROUP ``>`` on fixed streams, recreating a vanished group.

    ``initial_id`` applies only when the group is first created at startup
    (``"0"`` consumes existing entries, ``"$"`` only new ones). The reader keeps
    a concrete boundary per stream: the group's last-delivered id, read right
    after the group is ensured, then advanced by every delivered entry. A
    recreation resumes from that boundary, so it neither replays entries the
    group already delivered nor skips ones published while it was missing, even
    for an idle reader. Only if no boundary was ever readable (startup creation
    failed) does recreation fall back to ``"$"``.
    """

    def __init__(
        self,
        redis: Any,
        group: str,
        consumer: str,
        streams: list[str],
        *,
        initial_id: str,
    ) -> None:
        self._redis = redis
        self.group = group
        self.consumer = consumer
        self.streams = list(streams)
        self._initial_id = initial_id
        self._last_ids: dict[str, str] = {}
        self.recreations = 0

    async def ensure(self) -> None:
        """Startup creation; an existing group keeps its own position.

        Best-effort like the code it replaces: a failure here is logged, and
        the first read's missing-group recovery creates the group instead.
        """
        for stream in self.streams:
            try:
                await ensure_group(self._redis, stream, self.group, start_id=self._initial_id)
            except Exception:
                logger.warning("Could not create consumer group %s on %s", self.group, stream,
                               exc_info=True)
                continue
            await self._record_boundary(stream)

    async def _record_boundary(self, stream: str) -> None:
        """Remember the group's last-delivered id unless a delivery already set it."""
        if stream in self._last_ids:
            return
        try:
            groups = await self._redis.xinfo_groups(stream)
        except Exception:
            logger.warning("Could not read consumer group %s position on %s", self.group, stream,
                           exc_info=True)
            return
        for info in groups:
            if info.get("name") == self.group and info.get("last-delivered-id"):
                self._last_ids[stream] = info["last-delivered-id"]
                return

    async def read(self, *, count: int, block: int) -> list:
        """One XREADGROUP; on a vanished group, recreate it and return ``[]``."""
        try:
            results = await self._redis.xreadgroup(
                self.group, self.consumer, {s: ">" for s in self.streams},
                count=count, block=block,
            )
        except ResponseError as exc:
            if not is_missing_group(exc):
                raise
            await self._recreate(exc)
            return []
        for stream, messages in results or []:
            if messages:
                self._last_ids[stream] = messages[-1][0]
        return results or []

    async def _recreate(self, exc: ResponseError) -> None:
        self.recreations += 1
        for stream in self.streams:
            start_id = self._last_ids.get(stream, "$")
            logger.warning(
                "Consumer group %s on %s is missing (%s); recreating from %s",
                self.group, stream, exc, start_id,
            )
            await ensure_group(self._redis, stream, self.group, start_id=start_id)
            await self._record_boundary(stream)
