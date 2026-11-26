"""A confirmed name reaches every profile render (TASK-2643)."""
from __future__ import annotations

import asyncio
import sys

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

# The server registers its tool modules; importing profile_tools first is circular.
from llm_bawt.mcp_server import server  # noqa: F401
from llm_bawt.profiles import (
    EntityProfile,
    EntityType,
    ProfileAttribute,
    ProfileManager,
)

U = EntityType.USER
profile_tools = sys.modules["llm_bawt.mcp_server.profile_tools"]


@pytest.fixture
def manager(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[EntityProfile.__table__, ProfileAttribute.__table__])
    manager = ProfileManager.__new__(ProfileManager)
    manager.engine = engine
    monkeypatch.setattr(profile_tools, "_get_profile_manager", lambda: manager)
    yield manager
    engine.dispose()


def set_name(entity_id: str, value: str, *, category: str | None = None, key: str = "name") -> str:
    return asyncio.run(profile_tools.profile(
        action="set", entity_type="user", entity_id=entity_id, category=category, key=key, value=value))


def test_name_fills_empty_display_name_behind_a_legacy_summary(manager):
    manager.get_or_create_profile(U, "sample")
    manager.update_profile(U, "sample", summary="Likes garden walks.")
    set_name("sample", "  Avery  ")
    assert manager.get_profile(U, "sample").display_name == "Avery"
    # The default render returns the stored summary early; the name still shows.
    assert "User's name: Avery" in manager.get_profile_summary(U, "sample")


def test_existing_display_name_is_never_replaced(manager):
    manager.get_or_create_profile(U, "sample")
    manager.update_profile(U, "sample", display_name="Sam")
    set_name("sample", "Avery")
    assert manager.get_profile(U, "sample").display_name == "Sam"


def test_settings_edit_committed_mid_save_is_kept(manager, monkeypatch):
    manager.get_or_create_profile(U, "sample")
    settings = ProfileManager.__new__(ProfileManager)
    settings.engine = manager.engine
    original = manager.set_attribute

    def set_then_settings_edit(*args, **kwargs):
        # The name was blank when this save began; Settings commits Sam before the fill runs.
        result = original(*args, **kwargs)
        settings.update_profile(U, "sample", display_name="Sam")
        return result

    monkeypatch.setattr(manager, "set_attribute", set_then_settings_edit)
    set_name("sample", "Avery")
    assert manager.get_profile(U, "sample").display_name == "Sam"


def test_blank_display_name_counts_as_empty(manager):
    manager.get_or_create_profile(U, "sample")
    manager.update_profile(U, "sample", display_name="   ")
    assert manager.fill_display_name_if_empty(U, "sample", "Avery") is True
    assert manager.fill_display_name_if_empty(U, "sample", "Other") is False
    assert manager.get_profile(U, "sample").display_name == "Avery"


def test_other_keys_and_categories_leave_display_name_alone(manager):
    manager.get_or_create_profile(U, "sample")
    set_name("sample", "Avery", key="nickname")
    set_name("sample", "Avery", category="preference")
    assert not manager.get_profile(U, "sample").display_name


def test_name_summary_is_name_only(manager):
    assert manager.get_user_name_summary("sample") == ""
    manager.get_or_create_profile(U, "sample")
    manager.update_profile(U, "sample", display_name="Sample Owner", summary="Likes garden walks.")
    manager.set_attribute(U, "sample", "preference", "food", "tacos")
    assert manager.get_user_name_summary("sample") == "User's name: Sample Owner"
    # A free-form name fact can carry other details; it never crosses over.
    manager.set_attribute(
        U, "sample", "fact", "name",
        "User's name is Avery. They work at Example Studio.",
    )
    out = manager.get_user_name_summary("sample")
    assert out == "User's name: Sample Owner"
    assert "garden" not in out and "tacos" not in out and "Example Studio" not in out


def test_name_summary_follows_display_name_filled_by_orientation(manager):
    manager.get_or_create_profile(U, "sample")
    set_name("sample", "Avery")
    assert manager.get_user_name_summary("sample") == "User's name: Avery"


def test_bot_without_memory_gets_the_name_in_its_prompt(manager):
    from types import SimpleNamespace as NS

    from llm_bawt.core import prompt_sections
    from llm_bawt.core.prompt_builder import PromptBuilder

    manager.get_or_create_profile(U, "sample")
    manager.update_profile(U, "sample", display_name="Sample Owner")
    manager.set_attribute(U, "sample", "preference", "food", "tacos")
    owner = NS(_db_available=True, profile_manager=None, profile_name_reader=manager,
               user_id="sample", verbose=False,
               _resolve_setting=lambda _key, default=None: default)
    builder = PromptBuilder()
    prompt_sections.PromptSectionsMixin._sec_user_context(owner, builder, "")
    prompt = builder.build()
    assert "User's name: Sample Owner" in prompt
    assert "tacos" not in prompt
