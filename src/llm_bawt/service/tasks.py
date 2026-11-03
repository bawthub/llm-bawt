"""
Task definitions for the background service.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class TaskType(Enum):
    """Types of tasks the background service can handle."""
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEANING_UPDATE = "meaning_update"
    MEMORY_MAINTENANCE = "memory_maintenance"
    PROFILE_MAINTENANCE = "profile_maintenance"
    HISTORY_SUMMARIZATION = "history_summarization"
    MEMORY_EXTRACTION = "memory_extraction"
    MEDIA_ASSETS_GC = "media_assets_gc"
    TOOL_RESULT_GC = "tool_result_gc"
    HARNESS_VERSION_CHECK = "harness_version_check"


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A task to be processed by the background service."""
    
    task_type: TaskType
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bot_id: str = "nova"
    user_id: str = ""  # Required - caller must set this
    priority: int = 0  # Higher = more urgent
    
    def __post_init__(self):
        if not self.user_id:
            raise ValueError("user_id is required for Task")
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        user_id = data.get("user_id")
        if not user_id:
            raise ValueError("user_id is required in Task data")
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            payload=data["payload"],
            created_at=datetime.fromisoformat(data["created_at"]),
            bot_id=data.get("bot_id", "nova"),
            user_id=user_id,
            priority=data.get("priority", 0),
        )


@dataclass
class TaskResult:
    """Result of a processed task."""
    
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    processing_time_ms: float = 0.0
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "completed_at": self.completed_at.isoformat(),
        }


# Task factory functions for common operations
def create_meaning_update_task(
    memory_id: str,
    intent: str | None = None,
    stakes: str | None = None,
    emotional_charge: float | None = None,
    recurrence_keywords: list[str] | None = None,
    updated_tags: list[str] | None = None,
    reason: str = "",
    bot_id: str = "nova",
) -> Task:
    """Create a task to update meaning metadata on an existing memory."""
    return Task(
        task_type=TaskType.MEANING_UPDATE,
        payload={
            "memory_id": memory_id,
            "intent": intent,
            "stakes": stakes,
            "emotional_charge": emotional_charge,
            "recurrence_keywords": recurrence_keywords,
            "updated_tags": updated_tags,
            "reason": reason,
        },
        bot_id=bot_id,
    )


def create_maintenance_task(
    bot_id: str = "nova",
    user_id: str = "system",
    run_consolidation: bool = True,
    run_recurrence_detection: bool = True,
    run_decay_pruning: bool = False,
    run_orphan_cleanup: bool = False,
    dry_run: bool = False,
) -> Task:
    """Create a unified memory maintenance task.
    
    This task orchestrates multiple maintenance operations:
    - consolidation: merge similar memories
    - recurrence_detection: identify and tag recurring themes
    - decay_pruning: archive low-importance old memories
    - orphan_cleanup: remove orphaned embeddings/metadata
    """
    return Task(
        task_type=TaskType.MEMORY_MAINTENANCE,
        payload={
            "run_consolidation": run_consolidation,
            "run_recurrence_detection": run_recurrence_detection,
            "run_decay_pruning": run_decay_pruning,
            "run_orphan_cleanup": run_orphan_cleanup,
            "dry_run": dry_run,
        },
        bot_id=bot_id,
        user_id=user_id,
        priority=-1,  # Lower priority - background job
    )


def create_profile_maintenance_task(
    entity_id: str,
    entity_type: str = "user",
    bot_id: str = "nova",
    dry_run: bool = False,
    priority: int = 5,
) -> Task:
    """Create a profile maintenance task."""
    # Task requires user_id; for profile maintenance, treat entity_id as user_id.
    # (If entity_type is "bot", this is still a required identifier for service routing.)
    return Task(
        task_type=TaskType.PROFILE_MAINTENANCE,
        bot_id=bot_id,
        user_id=entity_id,
        priority=priority,
        payload={
            "entity_id": entity_id,
            "entity_type": entity_type,
            "dry_run": dry_run,
        },
    )


def create_history_summarization_task(
    bot_id: str = "nova",
    user_id: str = "system",
    use_heuristic_fallback: bool = True,
    max_tokens_per_chunk: int = 4000,
    model: str | None = None,
    priority: int = -1,
) -> Task:
    """Create a history summarization task."""
    return Task(
        task_type=TaskType.HISTORY_SUMMARIZATION,
        bot_id=bot_id,
        user_id=user_id,
        priority=priority,
        payload={
            "use_heuristic_fallback": use_heuristic_fallback,
            "max_tokens_per_chunk": max_tokens_per_chunk,
            "model": model,
        },
    )


def create_media_assets_gc_task(
    grace_days: int = 7,
    dry_run: bool = False,
    bot_id: str = "system",
    user_id: str = "system",
    priority: int = -1,
) -> Task:
    """Create a media-assets garbage-collection task.

    Scans ``media_assets`` for rows that are either past their ``expires_at``
    or are older than ``grace_days`` and unreferenced by any
    ``{bot}_messages.attachments`` JSONB. Each orphan is removed via
    :meth:`llm_bawt.media.store.MediaStore.delete` so the on-disk blob
    variants are cleaned up alongside the row.
    """
    return Task(
        task_type=TaskType.MEDIA_ASSETS_GC,
        bot_id=bot_id,
        user_id=user_id,
        priority=priority,
        payload={
            "grace_days": grace_days,
            "dry_run": dry_run,
        },
    )


def create_tool_result_gc_task(
    retention_days: int = 14,
    dry_run: bool = False,
    bot_id: str = "system",
    user_id: str = "system",
    priority: int = -1,
) -> Task:
    """Create a tool-result garbage-collection task (TASK-594).

    Prunes overflow tool-result blobs (``toolblobs/<sha256>`` in the object
    store) and legacy ``tool_call_result_payloads`` rows older than
    ``retention_days``. Content-addressed blobs shared by a newer record are
    preserved via a keep-set. The ``tool_call_records`` rows themselves are
    never deleted — only the heavy result body is reclaimed.
    """
    return Task(
        task_type=TaskType.TOOL_RESULT_GC,
        bot_id=bot_id,
        user_id=user_id,
        priority=priority,
        payload={
            "retention_days": retention_days,
            "dry_run": dry_run,
        },
    )


def create_harness_version_check_task(
    config: dict | None = None,
    bot_id: str = "system",
    user_id: str = "system",
    priority: int = -1,
) -> Task:
    """Create a weekly harness version-check task (TASK-1975)."""
    return Task(
        task_type=TaskType.HARNESS_VERSION_CHECK,
        bot_id=bot_id,
        user_id=user_id,
        priority=priority,
        payload=dict(config or {}),
    )


def create_memory_extraction_task(
    bot_id: str = "nova",
    user_id: str = "system",
    batch_size: int = 50,
    model: str | None = None,
    priority: int = -1,
) -> Task:
    """Create a memory extraction task.

    Processes unprocessed messages into semantic memories via LLM extraction.
    """
    return Task(
        task_type=TaskType.MEMORY_EXTRACTION,
        bot_id=bot_id,
        user_id=user_id,
        priority=priority,
        payload={
            "batch_size": batch_size,
            "model": model,
        },
    )
