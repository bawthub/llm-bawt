"""The ``injected_skills`` prompt section (TASK-1309).

It renders the resolver's admitted injected skills through the two registry
templates, and any failure skips the section instead of failing the build.
Fixture bots only; no database.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_bawt.skills.service as service_module
from agent_bridge.skill_sources import SkillBody, SkillEntry, SkillSource
from llm_bawt.core.prompt_builder import PromptBuilder
from llm_bawt.core.prompt_manifest import PROMPT_MANIFEST
from llm_bawt.core.prompt_sections import PromptSectionsMixin
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
from llm_bawt.skills.prompts import DEFAULT_ENTRY, DEFAULT_PREAMBLE, ENTRY_KEY, PREAMBLE_KEY
from llm_bawt.skills.resolver import Assignment, BotFacts, Caps, resolve


class _Source(SkillSource):
    kind = "fake"

    def list(self):
        return tuple(SkillEntry(r, r, "", self.kind, Path("/skills") / r) for r in ("a", "b", "big"))

    def read_body(self, ref):
        tokens = 500 if ref == "big" else 5
        return SkillBody(ref, f"Body {ref} {{literal}}", tokens, f"sha-{ref}")


def _resolution(rows=None, bot_type="agent"):
    facts = BotFacts("fixture-one", bot_type, "claude-code", None, frozenset(), False)
    rows = rows if rows is not None else [
        Assignment("bot", "fixture-one", "b", "inject", 1),
        Assignment("bot", "fixture-one", "a", "inject", 0),
        Assignment("bot", "fixture-one", "big", "inject", 2),
    ]
    return resolve(facts, rows, _Source(), Caps(per_skill=100, total=100))


class _Owner(PromptSectionsMixin):
    def __init__(self, templates):
        self.bot_id = "fixture-one"
        self.config = SimpleNamespace()
        self.resolves = []

        def resolve_body(key, scope_type="global", scope_id=None):
            self.resolves.append((key, scope_type, scope_id))
            body = templates.get(key)
            return None if body is None else SimpleNamespace(body=body, source="db_override")

        self.config_resolver = SimpleNamespace(resolve_body=resolve_body)


@pytest.fixture
def world(monkeypatch):
    state = {"resolution": _resolution(), "asked": []}

    class _Service:
        @classmethod
        def from_config(cls, engine, config):
            state["engine"] = engine
            return cls()

        def resolve_bot(self, slug):
            state["asked"].append(slug)
            if isinstance(state["resolution"], Exception):
                raise state["resolution"]
            return state["resolution"]

    monkeypatch.setattr(service_module, "SkillService", _Service)
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda _config: "engine")
    return state


def _section(owner):
    builder = PromptBuilder()
    owner._sec_injected_skills(builder, "")
    return {s.name: s for s in builder.enabled_sections}.get("injected_skills")


_TEMPLATES = {PREAMBLE_KEY: "## Pre", ENTRY_KEY: "### {ref} ({base_dir})\n{body}"}


def test_the_section_is_declared_agent_only_after_tag_context():
    names = [spec.name for spec in PROMPT_MANIFEST]
    spec = next(s for s in PROMPT_MANIFEST if s.name == "injected_skills")
    assert spec.applies_to == ("agent",)
    assert names.index("injected_skills") == names.index("tag_context") + 1


def test_admitted_skills_render_in_position_order_with_metadata(world):
    owner = _Owner(_TEMPLATES)
    section = _section(owner)

    assert section.content == ("## Pre\n\n### a (/skills/a)\nBody a {literal}"
                               "\n\n### b (/skills/b)\nBody b {literal}")
    assert section.metadata == {
        "source": "skills.resolve",
        "gate": "agent_backend & inject exceptions",
        "refs": ["a", "b"], "tokens": 10, "manifest_sha": None,
        "skipped": [{"ref": "big", "reason": "over_cap"}]}
    # The turn pins the manifest of the very resolution this section rendered.
    assert owner._skill_resolution is world["resolution"]
    assert owner._injected_skill_refs == ("a", "b")
    assert world["asked"] == ["fixture-one"] and world["engine"] == "engine"
    assert owner.resolves == [(PREAMBLE_KEY, "bot", "fixture-one"), (ENTRY_KEY, "bot", "fixture-one")]


@pytest.mark.parametrize("resolution", [None, "not_agent", "no_injected"])
def test_nothing_to_inject_reads_no_templates_and_adds_nothing(world, resolution):
    world["resolution"] = {
        None: None,
        "not_agent": _resolution(bot_type="chat"),
        "no_injected": _resolution(rows=[Assignment("bot", "fixture-one", "a", "enable")]),
    }[resolution]
    owner = _Owner(_TEMPLATES)
    owner._skill_resolution = "stale from an earlier turn"
    assert _section(owner) is None
    assert owner.resolves == []
    assert owner._skill_resolution is world["resolution"]


def test_an_unmounted_skill_source_injects_nothing(world, tmp_path):
    # A bot with inject rows while the app can't see the skill
    # checkout: every row is source_missing, nothing is injected or applied.
    from agent_bridge.skill_sources import AmbientSource
    from llm_bawt.skills.resolver import SOURCE_MISSING

    facts = BotFacts("fixture-one", "agent", "claude-code", None, frozenset(), False)
    rows = [Assignment("bot", "fixture-one", "a", "inject", 0),
            Assignment("bot", "fixture-one", "b", "enable", 1)]
    world["resolution"] = resolve(facts, rows, AmbientSource((tmp_path / "not-mounted",)),
                                  Caps(per_skill=100, total=100))
    owner = _Owner(_TEMPLATES)
    assert _section(owner) is None and owner.resolves == []
    assert {(e.ref, e.applied, e.reason) for e in world["resolution"].entries} == {
        ("a", False, SOURCE_MISSING), ("b", False, SOURCE_MISSING)}
    assert world["resolution"].manifest() is None


def test_a_resolution_failure_skips_the_section(world):
    world["resolution"] = RuntimeError("db down")
    owner = _Owner(_TEMPLATES)
    owner._skill_resolution = "stale from an earlier turn"
    assert _section(owner) is None
    assert owner._skill_resolution is None


def test_an_entry_template_without_a_body_slot_skips_the_section(world):
    owner = _Owner({PREAMBLE_KEY: "## Pre", ENTRY_KEY: "### {ref}"})
    owner._injected_skill_refs = ("stale",)
    assert _section(owner) is None
    assert owner._injected_skill_refs == ()  # TASK-1691: no body, no injected claim


def test_an_unreadable_template_skips_the_section(world):
    owner = _Owner({})

    def down(*_args, **_kwargs):
        raise RuntimeError("registry down")

    owner.config_resolver = SimpleNamespace(resolve_body=down)
    assert _section(owner) is None
    # TASK-1691: the manifest may still mark them inject; the hook signal must not.
    assert owner._skill_resolution is world["resolution"] and owner._injected_skill_refs == ()


def test_registry_defaults_are_the_seed_constants():
    assert DEFAULT_PROMPT_DEFINITIONS[PREAMBLE_KEY].default_body() == DEFAULT_PREAMBLE
    entry = DEFAULT_PROMPT_DEFINITIONS[ENTRY_KEY]
    assert entry.default_body() == DEFAULT_ENTRY
    assert "body" in entry.required_vars
