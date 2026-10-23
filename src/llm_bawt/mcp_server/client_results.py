"""Typed results returned by the memory client (TASK-1222).

``client.py`` re-exports both classes; import them from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryResult:
    """Result from a memory operation."""
    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    relevance: float = 0.0  # Only set for search results
    created_at: str | None = None
    last_accessed: str | None = None
    source_message_ids: list[str] = field(default_factory=list)
    intent: str | None = None
    stakes: str | None = None
    emotional_charge: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryResult":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            relevance=data.get("relevance", 0.0),
            created_at=data.get("created_at"),
            last_accessed=data.get("last_accessed"),
            source_message_ids=data.get("source_message_ids", []),
            intent=data.get("intent"),
            stakes=data.get("stakes"),
            emotional_charge=data.get("emotional_charge"),
        )


@dataclass
class MessageResult:
    """Result from a message operation."""
    id: str
    role: str
    content: str
    bot_id: str
    session_id: str | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MessageResult":
        return cls(
            id=data.get("id", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            bot_id=data.get("bot_id", ""),
            session_id=data.get("session_id"),
            created_at=data.get("created_at"),
        )
