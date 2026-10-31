"""Typed memory-update conflicts for API and tool callers."""


class MemoryUpdateDuplicateError(ValueError):
    """The updated content duplicates a different memory."""


class MemoryUpdateConflictError(RuntimeError):
    """The target memory changed since the update began."""
