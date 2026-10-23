"""Fail-closed positive policy, using synthetic identities only."""
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller


ROWS = [
    {"slug": "public-fixture", "include_in_global_search": True},
    {"slug": "second_public", "include_in_global_search": True},
    {"slug": "private_fixture", "include_in_global_search": False},
    {"slug": "other_private", "include_in_global_search": False},
]


def resolver(rows=ROWS):
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value = rows
    return BotPolicyResolver(engine)


def caller(bot="private_fixture", user="user_a", **kwargs):
    return TrustedCaller(bot, user, "service", **kwargs)


def test_protected_aggregate_is_public_union_strict_self():
    access = resolver().resolve_sync(caller())
    scope = access.aggregate_scope(require_user=True)
    assert [(p.bot_id, p.partition, p.authority) for p in scope.partitions] == [
        ("private_fixture", "private_fixture", "caller"),
        ("public-fixture", "publicfixture", "global"),
        ("second_public", "second_public", "global"),
    ]
    assert scope.public_authors == {"public-fixture", "second_public"}
    assert scope.user_id == "user_a"
    assert access.require_target("PRIVATE_FIXTURE") == "private_fixture"
    with pytest.raises(BotAccessError, match="^access_denied$"):
        access.require_target("other_private")
    assert len(access.aggregate_scope("private_fixture").partitions) == 1


@pytest.mark.parametrize("identity", [None, caller("public-fixture")])
def test_public_and_anonymous_never_reach_private(identity):
    access = resolver().resolve_sync(identity)
    assert {p.bot_id for p in access.aggregate_scope().partitions} == {
        "public-fixture", "second_public",
    }
    for target in ["private_fixture", "missing", "", None]:
        with pytest.raises(BotAccessError, match="^access_denied$"):
            access.require_target(target)
    assert access.require_target(" PUBLIC-FIXTURE ") == "public-fixture"


@pytest.mark.parametrize("identity", [None, caller(user=None)])
def test_provenance_requires_a_trusted_user(identity):
    with pytest.raises(BotAccessError, match="^caller_unavailable$"):
        resolver().resolve_sync(identity).aggregate_scope(require_user=True)


@pytest.mark.parametrize("rows", [[], ROWS * 65, ROWS + [ROWS[0]],
    [{"slug": "public-fixture", "include_in_global_search": True},
     {"slug": "publicfixture", "include_in_global_search": False}],
] + [[{"slug": "public-fixture", "include_in_global_search": value}]
     for value in [None, 0, 1, "false", "true", {}]]
  + [[{"slug": slug, "include_in_global_search": True}]
     for slug in [None, "", " PUBLIC ", "PUBLIC", "a" * 64, "bad\x00slug"]])
def test_invalid_policy_never_becomes_a_public_default(rows):
    with pytest.raises(BotAccessError, match="^policy_unavailable$"):
        resolver(rows).resolve_sync()


@pytest.mark.parametrize("identity", [caller("missing"), caller(" PUBLIC "),
    caller(user=""), caller(expires_at=1), caller(expires_at=float("nan")),
    TrustedCaller("private_fixture", None, "turn"),
])
def test_unknown_malformed_or_expired_caller_fails(identity):
    with pytest.raises(BotAccessError, match="^caller_unavailable$"):
        resolver().resolve_sync(identity)


def test_policy_and_principal_are_immutable():
    access = resolver().resolve_sync(caller())
    with pytest.raises(TypeError):
        access.partitions["other_private"] = "publicfixture"
    with pytest.raises(FrozenInstanceError):
        access.caller.bot_id = "other_private"


@pytest.mark.asyncio
async def test_live_read_no_stale_fallback_and_database_error_is_sanitized():
    policy = resolver()
    first = await policy.resolve()
    assert first.require_target("public-fixture") == "public-fixture"
    conn = policy.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value = [
        dict(row, include_in_global_search=False) for row in ROWS
    ]
    second = await policy.resolve()
    with pytest.raises(BotAccessError, match="^access_denied$"):
        second.require_target("public-fixture")
    policy.engine.connect.side_effect = RuntimeError("sensitive backend detail")
    with pytest.raises(BotAccessError, match="^policy_unavailable$") as failure:
        await policy.resolve()
    assert failure.value.__cause__ is None
    assert policy.engine.connect.call_count == 3
