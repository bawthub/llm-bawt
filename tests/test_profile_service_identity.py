"""Service classification retains profile identity and survives ordinary writes."""
from llm_bawt.profiles import EntityProfile, EntityType, ProfileManager
from sqlalchemy import create_engine
from sqlmodel import Session


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


def test_new_system_user_is_service_without_reclassifying_the_typed_bot():
    engine = create_engine("sqlite://")
    EntityProfile.__table__.create(engine)
    manager = object.__new__(ProfileManager)
    manager.engine = engine
    try:
        user, created = manager.get_or_create_profile(EntityType.USER, " System ")
        assert created and user.entity_id == "system" and user.is_service is True
        bot, created = manager.get_or_create_profile(EntityType.BOT, "system")
        assert created and bot.is_service is False and bot.id != user.id
        same_user, created = manager.get_or_create_profile(EntityType.USER, "SYSTEM")
        assert not created and same_user.id == user.id and same_user.is_service is True
    finally:
        engine.dispose()


def test_system_creation_default_preserves_an_existing_classification():
    engine = create_engine("sqlite://")
    EntityProfile.__table__.create(engine)
    manager = object.__new__(ProfileManager)
    manager.engine = engine
    try:
        with Session(engine) as session:
            existing = EntityProfile(entity_type=EntityType.USER, entity_id="system", is_service=False)
            session.add(existing)
            session.commit()
            session.refresh(existing)
            existing_id = existing.id
        profile, created = manager.get_or_create_profile(EntityType.USER, "system")
        assert not created and profile.id == existing_id and profile.is_service is False
    finally:
        engine.dispose()
