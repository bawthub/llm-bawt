"""Typed memory-update conflicts and write outcomes for API and tool callers."""

from dataclasses import dataclass
from typing import Literal

MemoryWriteStatus = Literal["inserted", "updated", "duplicate", "skipped", "failed"]


@dataclass(frozen=True)
class MemoryWriteOutcome:
    """What add_memory actually did.

    ``memory_id`` is the row that now holds the content: the new or updated
    row, or the existing row a duplicate was folded into. It is None when
    nothing was persisted (``skipped`` or ``failed``).
    """

    status: MemoryWriteStatus
    memory_id: str | None = None
    error: str | None = None

    @property
    def persisted(self) -> bool:
        return self.status in ("inserted", "updated", "duplicate") and bool(self.memory_id)


class MemoryUpdateDuplicateError(ValueError):
    """The updated content duplicates a different memory."""


class MemoryUpdateConflictError(RuntimeError):
    """The target memory changed since the update began."""
