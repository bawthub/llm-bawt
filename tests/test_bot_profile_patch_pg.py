"""Profile PATCH compare-and-set and the change audit (TASK-1521).

Real PATCH route, real store, one private PostgreSQL DB per test (the ``api``
fixture from ``test_bot_tag_api_pg``). Covers the three ways a write refuses
instead of clobbering: stale ``expected_fields``, stale ``expected_tags``
without a tag change, and a merge onto a row another write changed (which the
route re-merges). Also covers the ``bot_profile_changes`` row each write leaves.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from llm_bawt.bot_profile_changes import ProfileRowChanged
from llm_bawt.bot_tags import catalog
from llm_bawt.bot_tags.vocabulary import API_ASSIGNER
from llm_bawt.service.routes import settings
from tests.pg_lock_wait import wait_blocked_by
from tests.test_bot_tag_api_pg import _backend_pid, api  # noqa: F401
from tests.test_bot_tags_pg import rig  # noqa: F401

pytestmark = pytest.mark.integration

BOT = "fixture-bot"


def _changes(api, slug=BOT):  # noqa: F811
    with api.rig.engine.connect() as conn:
        return [dict(row) for row in conn.execute(text(
            "SELECT changed_by, change_note, changes FROM bot_profile_changes"
            " WHERE bot_slug=:s ORDER BY id"), {"s": slug}).mappings()]


def test_matching_expected_fields_write_and_leave_an_audit_row(api):  # noqa: F811
    response = api.patch(BOT, description="new", updated_by="bot:fixture-caller",
                         change_note="fixture reason", expected_fields={"description": ""})

    assert response.status_code == 200, response.text
    assert api.store.get(BOT).description == "new"
    assert _changes(api)[-1] == {
        "changed_by": "bot:fixture-caller", "change_note": "fixture reason",
        "changes": {"description": {"before": "", "after": "new"}}}


def test_stale_expected_fields_refuse_and_write_nothing(api):  # noqa: F811
    api.patch(BOT, description="theirs")
    audited = len(_changes(api))

    response = api.patch(BOT, description="mine", color="#123456",
                         expected_fields={"description": ""})

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "fields_changed"
    assert response.json()["detail"]["current"] == {"description": "theirs"}
    profile = api.store.get(BOT)
    assert (profile.description, profile.color) == ("theirs", None)
    assert len(_changes(api)) == audited


@pytest.mark.parametrize("body", [{"expected_fields": {"tags": []}},
                                  {"expected_fields": {"not_a_field": 1}},
                                  {"expected_fields": None}])
def test_expected_fields_names_only_profile_fields(api, body):  # noqa: F811
    response = api.patch(BOT, description="must not land", **body)

    assert response.status_code == 422, response.text
    assert api.store.get(BOT).description == ""


def test_expected_tags_alone_guards_a_field_write(api):  # noqa: F811
    stale = api.patch(BOT, description="mine", expected_tags=["private"])
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"]["code"] == "tags_changed"
    assert api.store.get(BOT).description == ""

    fresh = api.patch(BOT, description="mine", expected_tags=[])
    assert fresh.status_code == 200, fresh.text
    assert api.store.get(BOT).description == "mine"


def test_a_tag_change_is_audited_with_the_writer(api):  # noqa: F811
    response = api.patch(BOT, tags=["inactive"], updated_by="bot:fixture-caller")

    assert response.status_code == 200, response.text
    assert _changes(api)[-1]["changes"] == {"tags": {"before": [], "after": ["inactive"]}}
    assert _changes(api)[-1]["changed_by"] == "bot:fixture-caller"


def test_a_write_without_a_writer_is_audited_as_the_api_and_a_no_op_is_not(api):  # noqa: F811
    api.patch(BOT, color="#123456")
    audited = len(_changes(api))
    assert _changes(api)[-1]["changed_by"] == API_ASSIGNER

    assert api.patch(BOT, color="#123456").status_code == 200
    assert len(_changes(api)) == audited


def test_disjoint_concurrent_edits_both_survive(api):  # noqa: F811
    """The racer merged color onto the row it read; the holder then changed description.

    The store refuses the racer's stale merge under the bot lock and the route
    re-merges color onto the fresh row, so neither write reverts the other.
    """
    with ThreadPoolExecutor(max_workers=1) as pool, api.rig.engine.connect() as holder:
        holder.begin()
        catalog.lock_bot(holder, BOT)
        racer = pool.submit(api.patch, BOT, color="#123456")
        wait_blocked_by(api.rig.engine, _backend_pid(holder), racer, timeout=10)
        holder.execute(text("UPDATE bot_profiles SET description='holder', updated_at=now()"
                            " WHERE slug=:s"), {"s": BOT})
        holder.commit()
        merged = racer.result(timeout=10)

    assert merged.status_code == 200, merged.text
    profile = api.store.get(BOT)
    assert (profile.description, profile.color) == ("holder", "#123456")


def test_same_field_race_with_expected_fields_is_a_409_not_a_retry(api):  # noqa: F811
    with ThreadPoolExecutor(max_workers=1) as pool, api.rig.engine.connect() as holder:
        holder.begin()
        catalog.lock_bot(holder, BOT)
        racer = pool.submit(api.patch, BOT, description="racer", expected_fields={"description": ""})
        wait_blocked_by(api.rig.engine, _backend_pid(holder), racer, timeout=10)
        holder.execute(text("UPDATE bot_profiles SET description='holder', updated_at=now()"
                            " WHERE slug=:s"), {"s": BOT})
        holder.commit()
        refused = racer.result(timeout=10)

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "fields_changed"
    assert api.store.get(BOT).description == "holder"


def test_a_row_that_keeps_changing_gives_up_after_three_merges(api, monkeypatch):  # noqa: F811
    attempts = []

    async def always_stale(payload, *, create_only):
        attempts.append(payload["_expected_row_version"])
        raise ProfileRowChanged(BOT, {}, "changed")

    monkeypatch.setattr(settings, "_persist_bot_profile", always_stale)
    response = api.patch(BOT, color="#123456")

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "row_changed"
    assert len(attempts) == settings._PATCH_ATTEMPTS == 3
