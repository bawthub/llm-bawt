"""Retired built-in prompts: replacement pointer, display and edit guard (TASK-1853)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
from llm_bawt.prompt_registry import PromptResolver, PromptTemplate, PromptTemplateStore, PromptTemplateVersion
from llm_bawt.service.routes import prompts as routes
from llm_bawt.service.schemas import PromptTemplateUpsertRequest

OLD_FACT = "memory.extraction.fact"
SOURCED_FACT = "memory.extraction.fact.sourced"
SRC = Path(__file__).resolve().parents[1] / "src" / "llm_bawt"
# Files allowed to name a retired key: where it is defined and loaded.
_DEFINITION_FILES = {"prompt_definitions.py", "prompt_default_loaders.py"}


def _seeded_store() -> PromptTemplateStore:
    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.config = None
    store.connection_url = "sqlite://test"
    store.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        store.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__]
    )
    store.seed_defaults()
    return store


@pytest.fixture
def store(monkeypatch) -> PromptTemplateStore:
    seeded = _seeded_store()

    class _Resolver(PromptResolver):
        def __init__(self, config=None):
            self.config = config
            self.store = seeded

    monkeypatch.setattr(routes, "PromptResolver", _Resolver)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    return seeded


def _snapshot(store: PromptTemplateStore, key: str) -> tuple[str, list[int]]:
    with Session(store.engine) as session:
        row = session.exec(select(PromptTemplate).where(PromptTemplate.key == key)).one()
        versions = session.exec(
            select(PromptTemplateVersion.version).where(PromptTemplateVersion.key == key)
        ).all()
        return row.body, sorted(versions)


def test_old_fact_key_is_retired_in_favour_of_the_sourced_key() -> None:
    old = DEFAULT_PROMPT_DEFINITIONS[OLD_FACT]
    assert old.deprecated and old.replaced_by == SOURCED_FACT
    assert not DEFAULT_PROMPT_DEFINITIONS[SOURCED_FACT].deprecated


def test_every_replacement_pointer_names_a_live_prompt() -> None:
    for key, definition in DEFAULT_PROMPT_DEFINITIONS.items():
        if not definition.deprecated:
            continue
        target = DEFAULT_PROMPT_DEFINITIONS.get(definition.replaced_by)
        assert target is not None, f"{key} points at unregistered {definition.replaced_by}"
        assert not target.deprecated, f"{key} points at retired {definition.replaced_by}"


def test_no_runtime_code_renders_a_retired_key() -> None:
    retired = [k for k, d in DEFAULT_PROMPT_DEFINITIONS.items() if d.deprecated]
    pattern = re.compile("|".join(rf"[\"']{re.escape(k)}[\"']" for k in retired))
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name not in _DEFINITION_FILES and pattern.search(path.read_text())
    ]
    assert offenders == []


def test_fact_extraction_resolves_the_sourced_key(monkeypatch) -> None:
    from llm_bawt.memory.extraction import prompts as extraction_prompts

    seen: list[str] = []

    class _Resolver:
        def render(self, *, key, variables):
            seen.append(key)
            return ""

    monkeypatch.setattr(extraction_prompts, "get_prompt_resolver", lambda: _Resolver())
    extraction_prompts.get_fact_extraction_prompt("User: hi")
    assert seen == [SOURCED_FACT]


def test_retired_key_is_still_seeded_and_readable_with_its_history(store) -> None:
    body, versions = _snapshot(store, OLD_FACT)
    assert body == DEFAULT_PROMPT_DEFINITIONS[OLD_FACT].default_body()
    assert versions == [1]
    shown = routes.get_prompt(OLD_FACT, scope_type="global", scope_id=None, exact=False)
    assert shown.body == body


def test_reads_report_the_retirement_and_replacement(store) -> None:
    resolved = routes.get_prompt(OLD_FACT, scope_type="global", scope_id=None, exact=False)
    exact = routes.get_prompt(OLD_FACT, scope_type="global", scope_id=None, exact=True)
    for item in (resolved, exact):
        assert item.deprecated is True and item.replaced_by == SOURCED_FACT

    live = routes.get_prompt(SOURCED_FACT, scope_type="global", scope_id=None, exact=False)
    assert live.deprecated is False and live.replaced_by is None

    listed = {p.key: p for p in routes.list_prompts(
        category=None, scope_type="global", scope_id=None, include_defaults=True
    ).prompts}
    index = {p.key: p for p in routes.list_prompts_index(
        category=None, scope_type="global", scope_id=None, include_defaults=True
    ).prompts}
    for items in (listed, index):
        assert items[OLD_FACT].deprecated is True
        assert items[OLD_FACT].replaced_by == SOURCED_FACT
        assert items[SOURCED_FACT].deprecated is False


@pytest.mark.parametrize("scope_type,scope_id", [("global", None), ("bot", "somebot")])
def test_edits_to_a_retired_key_are_refused_and_change_nothing(store, scope_type, scope_id) -> None:
    before = _snapshot(store, OLD_FACT)
    request = PromptTemplateUpsertRequest(
        body="Tuned {conversation}", scope_type=scope_type, scope_id=scope_id
    )
    with pytest.raises(HTTPException) as caught:
        routes.upsert_prompt(OLD_FACT, request)
    assert caught.value.status_code == 409
    assert caught.value.detail["error"] == "prompt_deprecated"
    assert caught.value.detail["replaced_by"] == SOURCED_FACT
    assert SOURCED_FACT in caught.value.detail["message"]
    assert _snapshot(store, OLD_FACT) == before


def test_reset_of_a_retired_key_is_refused_and_keeps_its_row(store) -> None:
    before = _snapshot(store, OLD_FACT)
    with pytest.raises(HTTPException) as caught:
        routes.reset_prompt(OLD_FACT, scope_type="global", scope_id=None)
    assert caught.value.status_code == 409
    assert caught.value.detail["replaced_by"] == SOURCED_FACT
    assert _snapshot(store, OLD_FACT) == before


def test_the_replacement_key_stays_editable(store) -> None:
    body = "Sourced, tuned.\n{conversation}"
    saved = routes.upsert_prompt(SOURCED_FACT, PromptTemplateUpsertRequest(body=body))
    assert saved.body == body and saved.deprecated is False
    assert _snapshot(store, SOURCED_FACT) == (body, [1, 2])
