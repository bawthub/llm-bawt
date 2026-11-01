"""Actual mutation decisions on disposable PG; no live profiles or services."""
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotAccessError, TrustedCaller
from llm_bawt.bot_access.refresh import PolicyRefresher
from llm_bawt.bot_tags import catalog
from llm_bawt.bot_tags.grants import GrantActor, write_grant
from llm_bawt.bot_tags.store import BotTagStore
from tests.test_bot_policy_refresh_pg import listen, notified
from tests.test_bot_policy_refresh_pg import world as world

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_all_policy_input_mutations_update_two_consumer_decisions(world):  # noqa: F811 - pytest fixture
    second = create_engine(world.url, connect_args={"options": f"-csearch_path={world.schema},public"})
    owners = [PolicyRefresher(world.engine), PolicyRefresher(second)]
    tags = BotTagStore(world.engine)

    def public(policy):
        access = replace(policy, caller=TrustedCaller("fixture_owner", "fixture_user", "service"))
        try:
            access.require_target("fixture_peer")
            return True
        except BotAccessError:
            return False

    try:
        with listen(world.engine) as listener:
            for owner in owners:
                assert owner.refresh()

            def changed(check):
                assert {event.payload for event in notified(listener)} == {world.schema}
                for owner in owners:
                    assert owner.refresh()
                    assert check(owner.snapshot.get())

            world.store.upsert({"slug": "fixture_new", "tags": []})
            changed(lambda p: "fixture_new" in p.partitions)
            world.store.upsert({"slug": "fixture_peer", "include_in_global_search": False})
            changed(lambda p: not public(p))
            world.store.upsert({"slug": "fixture_peer", "include_in_global_search": True})
            changed(public)
            world.store.upsert({"slug": "fixture_peer", "tags": ["private"]})
            changed(lambda p: not public(p))
            world.store.upsert({"slug": "fixture_peer", "tags": []})
            changed(public)
            world.store.upsert({"slug": "fixture_peer", "tags": ["no_global_search"]})
            changed(lambda p: p.tags.has("fixture_peer", "no_global_search"))
            tags.unassign("no_global_search", "fixture_peer")
            changed(lambda p: not p.tags.has("fixture_peer", "no_global_search"))
            for granted in (True, False):
                with world.engine.begin() as conn:
                    assert write_grant(conn, "approve_ops", "fixture_peer", granted=granted,
                        reason="fixture", actor=GrantActor.for_person("fixture_operator"),
                        capability=None).changed
                changed(lambda p: p.tags.has("fixture_peer", "approve_ops") == granted)
            tags.create_role("fixture_role", "Fixture role")
            changed(lambda p: "fixture_role" in p.tags.catalog)
            tags.rename_role("fixture_role", "fixture_renamed")
            changed(lambda p: "fixture_renamed" in p.tags.catalog and "fixture_role" not in p.tags.catalog)
            tags.delete_role("fixture_renamed")
            changed(lambda p: "fixture_renamed" not in p.tags.catalog)
            assert world.store.delete("fixture_new")
            changed(lambda p: "fixture_new" not in p.partitions)
    finally:
        second.dispose()


def test_unrelated_profile_role_metadata_and_identical_tag_saves_emit_no_refresh(world):  # noqa: F811 - pytest fixture
    tags = BotTagStore(world.engine)
    tags.create_role("fixture_role", "Fixture role")
    world.store.upsert({"slug": "fixture_peer", "tags": ["fixture_role"]})
    with listen(world.engine) as listener:
        world.store.upsert({"slug": "fixture_peer", "description": "Unrelated description"})
        tags.update_role("fixture_role", {"description": "Unrelated metadata", "capabilities": ["fixture"]})
        world.store.upsert({"slug": "fixture_peer", "tags": ["fixture_role"]})
        with world.engine.begin() as conn:
            catalog.seed(conn)  # no-op startup seed is not an impacting change
        # A round trip consumes pending server messages without a quiet-period
        # sleep/negative timing assertion.
        listener.execute(text("SELECT 1"))
        raw = listener.connection.driver_connection
        raw.poll()
        assert not raw.notifies
