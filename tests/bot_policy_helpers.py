"""Explicit startup/refresh for isolated tests that don't run service lifespans."""
from llm_bawt.bot_access import BotPolicyResolver
from llm_bawt.bot_access.refresh import PolicyRefresher


def warm_policy_resolver(engine):
    # Production request handlers never invoke this helper. Tests without a
    # lifespan must explicitly establish or refresh their synthetic policy.
    PolicyRefresher(engine).refresh()
    return BotPolicyResolver(engine)
