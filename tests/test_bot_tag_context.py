"""The ``agents.tag_context`` section and its facts (TASK-1249).

The section stays inert until an operator writes the template: an empty body
reads nothing and adds nothing. With a template it renders this bot's role-tag
instructions and the family roster as sorted facts, and a broken template or an
unreadable fact source skips the section instead of failing the build. Fixture
bots and tags only; no database.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import llm_bawt.bot_tags.context as context_module
from llm_bawt.bot_tags.context import RoleTag, RosterBot, TagContext, render_tag_context
from llm_bawt.core.prompt_builder import PromptBuilder
from llm_bawt.core.prompt_manifest import PROMPT_MANIFEST
from llm_bawt.core.prompt_sections import PromptSectionsMixin

_ROLES = [
    RoleTag("fixture_b", "Fixture B", "Checks things.", ("review",), "role.fixture_b"),
    RoleTag("fixture_a", "Fixture A", "Builds things.", ("implement",), None),
    RoleTag("fixture_c", "Fixture C", "", (), "role.fixture_gone"),
]
_BOTS = [
    RosterBot("fixture-two", ("fixture_b",), None, None),
    RosterBot("fixture-one", ("fixture_a", "fixture_b"), "model-x@path-y", "vendor-z"),
    RosterBot("fixture-plain", (), "model-x@path-y", "vendor-z"),
]
_BODIES = {"role.fixture_b": "  Review the commit, not the summary.  "}


def _render(own=("fixture_a", "fixture_b", "fixture_c"), roles=_ROLES, bots=_BOTS):
    return render_tag_context(bots, roles, _BODIES.get, own)


def test_facts_render_sorted_with_instructions_or_descriptions():
    rendered = _render()

    assert rendered.own_tag_instructions == (
        "### Fixture A (`fixture_a`)\nBuilds things.\n\n"
        "### Fixture B (`fixture_b`)\nReview the commit, not the summary.\n\n"
        "### Fixture C (`fixture_c`)")
    assert rendered.bot_roster == (
        "- fixture-one: tags fixture_a, fixture_b; endpoint model-x@path-y; provider vendor-z\n"
        "- fixture-plain: tags none; endpoint model-x@path-y; provider vendor-z\n"
        "- fixture-two: tags fixture_b; endpoint unknown; provider unknown")
    assert rendered.tag_catalog == (
        "- `fixture_a` (Fixture A): capabilities implement. Builds things.\n"
        "- `fixture_b` (Fixture B): capabilities review. Checks things.\n"
        "- `fixture_c` (Fixture C): capabilities none")


def test_a_key_with_no_prompt_flags_missing_instructions_and_still_renders():
    rendered = _render()

    assert rendered.flags == ("missing_instructions:fixture_c",)
    assert "### Fixture C" in rendered.own_tag_instructions


def test_a_tag_without_a_key_is_not_flagged():
    assert _render(own=("fixture_a",)).flags == ()


def test_input_order_never_changes_the_bytes():
    assert _render(roles=list(reversed(_ROLES)), bots=list(reversed(_BOTS))) == _render()


def test_a_bot_with_no_role_tags_gets_none_for_its_own_instructions():
    assert _render(own=("private",)).own_tag_instructions == "none"


def test_a_rename_changes_only_the_slug():
    # fixture_bb sorts where fixture_b did; a rename that moves the sort
    # position reorders the lists, which is the only other change.
    renamed = [replace(role, slug="fixture_bb") if role.slug == "fixture_b" else role
               for role in _ROLES]
    bots = [RosterBot(b.slug, tuple("fixture_bb" if t == "fixture_b" else t for t in b.tags),
                      b.endpoint_ref, b.provider) for b in _BOTS]

    before = _render()
    after = _render(own=("fixture_a", "fixture_bb", "fixture_c"), roles=renamed, bots=bots)

    for name in ("own_tag_instructions", "bot_roster", "tag_catalog"):
        assert getattr(after, name) != getattr(before, name)
        assert getattr(after, name).replace("fixture_bb", "fixture_b") == getattr(before, name)
    assert after.flags == before.flags


# --- the section --------------------------------------------------------------


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
def facts(monkeypatch):
    state = {"loads": 0, "context": TagContext("OWN", "ROSTER", "CATALOG")}

    def load(_engine, _config, bot_id, instructions):
        state["loads"] += 1
        state["bot_id"] = bot_id
        state["instructions"] = instructions
        if isinstance(state["context"], Exception):
            raise state["context"]
        return state["context"]

    monkeypatch.setattr(context_module, "load_tag_context", load)
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda _config: "engine")
    return state


def _section(owner):
    builder = PromptBuilder()
    owner._sec_tag_context(builder, "")
    return {s.name: s for s in builder.enabled_sections}.get("tag_context")


def test_the_section_is_declared_agent_only_after_the_global_prompt():
    names = [spec.name for spec in PROMPT_MANIFEST]
    spec = next(s for s in PROMPT_MANIFEST if s.name == "tag_context")

    assert spec.applies_to == ("agent",)
    assert names.index("tag_context") == names.index("agent_global_prompt") + 1


@pytest.mark.parametrize("template", [None, "", "  \n"])
def test_an_empty_template_reads_no_facts_and_adds_nothing(facts, template):
    owner = _Owner({"agents.tag_context": template})

    assert _section(owner) is None
    assert facts["loads"] == 0
    assert owner.resolves == [("agents.tag_context", "bot", "fixture-one")]


def test_the_template_renders_the_facts(facts):
    owner = _Owner({"agents.tag_context": "Mine:\n{own_tag_instructions}\n{bot_roster}\n"
                                          "{tag_catalog} {{literal}}",
                    "role.fixture_b": "BODY"})

    section = _section(owner)

    assert section.content == "Mine:\nOWN\nROSTER\nCATALOG {literal}"
    assert section.metadata == {"source": "registry:db_override",
                                "gate": "agent_backend & agents.tag_context template"}
    assert facts["bot_id"] == "fixture-one"
    assert facts["instructions"]("role.fixture_b") == "BODY"
    assert facts["instructions"]("role.missing") is None


def test_flags_ride_the_section_provenance(facts):
    facts["context"] = TagContext("OWN", "R", "C", ("missing_instructions:fixture_c",))

    section = _section(_Owner({"agents.tag_context": "{own_tag_instructions}"}))

    assert section.metadata["provenance"] == ["missing_instructions:fixture_c"]


@pytest.mark.parametrize("template", ["{unknown_var}", "{0}", "{own_tag_instructions"])
def test_a_template_that_does_not_render_skips_the_section(facts, template):
    assert _section(_Owner({"agents.tag_context": template})) is None


def test_unreadable_facts_skip_the_section(facts):
    facts["context"] = RuntimeError("db down")

    assert _section(_Owner({"agents.tag_context": "{bot_roster}"})) is None
