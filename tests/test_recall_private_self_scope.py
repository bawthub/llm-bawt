"""TASK-2654: a private bot recalls searchable bots plus its own history.

Matches BotAccess.aggregate_scope; no other bot ever sees the private one.
"""
import pytest

from llm_bawt.recall.contracts import RecallError
from llm_bawt.recall.postgres import RecallDatabase, delivery_authors

ROWS = [{"slug": "fixture-public", "include_in_global_search": True},
        {"slug": "fixture-peer", "include_in_global_search": True},
        {"slug": "fixture-private", "include_in_global_search": False}]


class PolicyOnly(RecallDatabase):
    def __init__(self):
        super().__init__(None)

    def read(self, statement, params, timeout_ms=2000, retry_transient=True):
        return [dict(row) for row in ROWS]


@pytest.mark.asyncio
async def test_private_caller_scope_is_searchable_bots_plus_itself():
    scope = await PolicyOnly().owner_scope("fixture-private", "sample-user", "s1")
    assert scope.allowed_bot_ids - scope.excluded_bot_ids == {
        "fixture-private", "fixture-public", "fixture-peer"}
    assert set(delivery_authors(scope)) <= {"fixture-public", "fixture-peer"}


@pytest.mark.asyncio
async def test_private_caller_never_gains_another_private_bot():
    rows = ROWS + [{"slug": "fixture-other-private", "include_in_global_search": False}]

    class TwoPrivate(PolicyOnly):
        def read(self, statement, params, timeout_ms=2000, retry_transient=True):
            return [dict(row) for row in rows]

    scope = await TwoPrivate().owner_scope("fixture-private", "sample-user", "s1")
    assert "fixture-other-private" in scope.excluded_bot_ids
    assert "fixture-other-private" not in scope.allowed_bot_ids


@pytest.mark.asyncio
async def test_public_caller_never_gains_the_private_bot():
    scope = await PolicyOnly().owner_scope("fixture-public", "sample-user", "s1")
    assert "fixture-private" in scope.excluded_bot_ids
    assert "fixture-private" not in scope.allowed_bot_ids - scope.excluded_bot_ids


@pytest.mark.asyncio
async def test_unknown_caller_is_still_unauthorized():
    with pytest.raises(RecallError) as error:
        await PolicyOnly().owner_scope("fixture-unknown", "sample-user", "s1")
    assert error.value.code == "unauthorized"
