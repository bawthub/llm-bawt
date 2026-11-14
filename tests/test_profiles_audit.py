"""TASK-2280: conflict-safe creation and deletion of visible profile facts."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.profiles import (
    EntityProfile, EntityType, ProfileAttribute, ProfileManager,
    STATUS_CANDIDATE, STATUS_LIVE, STATUS_RETIRED,
)


@pytest.fixture
def manager(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
    SQLModel.metadata.create_all(engine, tables=[EntityProfile.__table__, ProfileAttribute.__table__])
    manager = ProfileManager.__new__(ProfileManager)
    manager.engine = engine
    yield manager
    engine.dispose()


def test_concurrent_normalized_creation_returns_one_new_profile(manager):
    barrier = Barrier(2)
    def create(name):
        barrier.wait()
        return manager.get_or_create_profile(EntityType.USER, name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, [" Fixture ", "fixture"]))
    assert len({profile.id for profile, _ in results}) == 1
    assert sorted(is_new for _, is_new in results) == [False, True]
    bot, is_new = manager.get_or_create_profile(EntityType.BOT, "fixture")
    assert is_new and bot.id != results[0][0].id


@pytest.mark.parametrize("method", ["key", "search", "id"])
def test_deletion_invalidates_summary_and_preserves_hidden_rows(manager, method):
    profile, _ = manager.get_or_create_profile(EntityType.USER, "fixture")
    with Session(manager.engine) as session:
        row = session.get(EntityProfile, profile.id)
        row.summary = "Lives in Boston"
        session.add(row)
        for status in (STATUS_RETIRED, STATUS_CANDIDATE, STATUS_LIVE):
            session.add(ProfileAttribute(entity_type=EntityType.USER, entity_id="fixture",
                                         category="fact", key="city", value="Boston", status=status))
        session.commit()
        live_id = session.exec(select(ProfileAttribute).where(ProfileAttribute.status == STATUS_LIVE)).one().id
    if method == "key":
        assert manager.delete_attribute(EntityType.USER, " Fixture ", "fact", "city")
        assert not manager.delete_attribute(EntityType.USER, "fixture", "fact", "city")
    elif method == "search":
        count, _ = manager.search_and_delete_attributes(EntityType.USER, "fixture", "Boston")
        assert count == 1
    else:
        assert manager.delete_attribute_by_id(live_id)
    with Session(manager.engine) as session:
        assert session.get(EntityProfile, profile.id).summary is None
        assert {row.status for row in session.exec(select(ProfileAttribute)).all()} == {
            STATUS_CANDIDATE, STATUS_RETIRED,
        }
