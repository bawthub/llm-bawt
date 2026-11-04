"""Retired bot slug aliases (TASK-1900): reservation, delete refusal, listing.

Synthetic slugs only, on the private app PostgreSQL rig from the create-route
tests. Alias rows are operator data, so tests insert them with SQL directly.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from llm_bawt import bot_slug_aliases
from llm_bawt.service.routes import settings
from llm_bawt.utils.bot_partition import sanitize_table_name
from tests.test_profile_route_gate_pg import _create, rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


_MINE: list[str] = []


@pytest.fixture(autouse=True)
def _drop_fixture_aliases(rig):  # noqa: F811
    """Torn down before the rig deletes its bots, which RESTRICT would refuse."""
    yield
    with rig.engine.begin() as conn:
        for old in _MINE:
            conn.execute(text("DELETE FROM bot_slug_aliases WHERE old_slug=:o"), {"o": old})
    _MINE.clear()


def _old(rig, label):  # noqa: F811
    """A unique, already-normalized retired slug for this test."""
    return sanitize_table_name(f"{rig.new_slug()}_{label}")


def _alias(rig, old, bot):  # noqa: F811
    _MINE.append(old)
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_slug_aliases (old_slug, bot_slug) VALUES (:o, :b)"),
                     {"o": old, "b": bot})


def _aliases(rig, *bots):  # noqa: F811
    with rig.engine.connect() as conn:
        grouped = bot_slug_aliases.aliases_by_bot(conn)
    return {bot: grouped[bot] for bot in bots if bot in grouped}


def test_create_refuses_a_reserved_slug_and_its_cleaned_form(rig):  # noqa: F811
    target = rig.new_slug()
    _create(rig, target)
    retired = _old(rig, "retired")
    _alias(rig, retired, target)
    # The slug a renamed bot used to carry, plus forms that clean to it.
    head, tail = retired[:4], retired[4:]
    for attempt in (retired, retired.upper(), f"{head}-{tail}"):
        with pytest.raises(HTTPException) as refused:
            _create(rig, attempt)
        assert refused.value.status_code == 409, attempt
        assert "reserved" in refused.value.detail
    assert rig.store.get(retired) is None


def test_an_alias_cannot_reserve_a_live_slug_or_point_nowhere(rig):  # noqa: F811
    live, other = rig.new_slug(), rig.new_slug()
    _create(rig, live)
    _create(rig, other)
    with pytest.raises(DBAPIError):
        _alias(rig, sanitize_table_name(live), other)
    with pytest.raises(DBAPIError):
        _alias(rig, _old(rig, "orphan"), "fixture-no-such-bot")
    with pytest.raises(DBAPIError):  # keys are stored in normalized form only
        _alias(rig, f"{_old(rig, 'mixed')}-X", live)
    assert _aliases(rig, live, other) == {}


def test_delete_with_aliases_refuses_before_any_purge(rig, monkeypatch):  # noqa: F811
    target = rig.new_slug()
    _create(rig, target)
    retired = _old(rig, "one")
    _alias(rig, retired, target)
    purged = []
    monkeypatch.setattr(settings, "_purge_bot_data_or_raise", lambda *_a: purged.append(True))

    with pytest.raises(HTTPException) as refused:
        asyncio.run(settings.delete_bot_profile(target, purge=True))
    assert refused.value.status_code == 409
    assert purged == []
    assert rig.store.get(target) is not None
    with pytest.raises(bot_slug_aliases.BotHasAliases):
        rig.store.delete(target)

    with rig.engine.begin() as conn:  # the recorded inverse frees the bot again
        conn.execute(text("DELETE FROM bot_slug_aliases WHERE old_slug=:o"), {"o": retired})
    assert rig.store.delete(target) is True


def test_aliases_group_by_target_and_follow_a_target_slug_change(rig):  # noqa: F811
    target = rig.new_slug()
    _create(rig, target)
    first, second = sorted([_old(rig, "a"), _old(rig, "b")])
    _alias(rig, second, target)
    _alias(rig, first, target)
    assert _aliases(rig, target) == {target: [first, second]}

    renamed = rig.new_slug()
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET slug=:n WHERE slug=:o"), {"n": renamed, "o": target})
    assert _aliases(rig, target, renamed) == {renamed: [first, second]}


def test_bot_list_reader_returns_aliases_by_bot(rig, monkeypatch):  # noqa: F811
    from llm_bawt.service import dependencies
    from llm_bawt.service.routes import models

    target = rig.new_slug()
    _create(rig, target)
    retired = _old(rig, "listed")
    _alias(rig, retired, target)
    monkeypatch.setattr(dependencies, "get_bot_profile_store", lambda _c: rig.store)

    assert models._read_slug_aliases(None)[target] == [retired]
