"""Shared positive authorization contracts (TASK-1222)."""
from .policy import (
    AggregateScope, BotAccess, BotAccessError, BotPolicyResolver,
    PartitionScope, TrustedCaller,
)

__all__ = [
    "AggregateScope", "BotAccess", "BotAccessError", "BotPolicyResolver",
    "PartitionScope", "TrustedCaller",
]
