"""Service classification retains profile identity and survives ordinary writes."""
from llm_bawt.profiles import EntityProfile, EntityType, ProfileManager
from sqlalchemy import create_engine


def test_service_classification_persists_without_replacing_the_author():
    engine = create_engine("sqlite://")
    EntityProfile.__table__.create(engine)
    manager = object.__new__(ProfileManager)
    manager.engine = engine
    person, created = manager.get_or_create_profile(EntityType.USER, "automation")
    assert created and person.is_service is False
    flagged = manager.update_profile(EntityType.USER, "automation", is_service=True)
    assert flagged.id == person.id and flagged.is_service is True
    existing, created = manager.get_or_create_profile(EntityType.USER, "automation")
    assert not created and existing.id == person.id and existing.is_service is True
    edited = manager.update_profile(EntityType.USER, "automation", display_name="New name")
    assert edited.is_service is True
    assert manager.list_profiles(EntityType.USER)[0].is_service is True
    assert manager.update_profile(EntityType.USER, "automation", is_service=False).is_service is False
    engine.dispose()
