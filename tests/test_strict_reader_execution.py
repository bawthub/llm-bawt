"""Strict named-reader routing never reads the shared bot registry (TASK-2590).

Synthetic identities only. Every registry entry point is rigged to fail the
test, so cold, invalidated, missing and mismatched cases prove the strict
path never reaches it and never falls back.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bawt import bots
from llm_bawt.bots import Bot, load_named_bot
from llm_bawt.core.initialization import OrchestratorInitializationMixin as BaseLLMBawt
from llm_bawt.service import stream_execution_selection as selection
from llm_bawt.service import strict_reader_execution as strict

SLUG = "sample-agent"


@pytest.fixture(autouse=True)
def registry_forbidden(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("shared bot registry was read")

    monkeypatch.setattr(bots, "_load_bots_config", boom)
    monkeypatch.setattr(bots, "_load_db_bot_profiles", boom)
    monkeypatch.setattr(bots, "_load_bot_tags", boom)
    monkeypatch.setattr(bots, "_LOADED", False)  # cold
    monkeypatch.setattr(bots, "_FORCE_RELOAD", True)  # invalidated
    monkeypatch.setattr(bots, "_RETRY_AFTER", 0.0)
    monkeypatch.setattr(bots.BotManager, "__init__", boom)


def _row(slug=SLUG, **over):
    base = dict(
        slug=slug, name="Sample", description="", system_prompt="hi",
        prompt_override_id=None, requires_memory=False, voice_optimized=False,
        tts_mode=False, include_summaries=True, include_in_global_search=True,
        uses_tools=False, uses_search=False, uses_home_assistant=False,
        default_model=None, harness="claude-code", endpoint_id=None, house=None,
        house_pinned=False, color=None, cli_color=None, avatar=None,
        avatar_render=None, default_voice=None, nextcloud_config=None,
        bot_type="agent", agent_backend="claude-code", agent_backend_config=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class ProfileStore:
    def __init__(self, rows):
        self.rows = {r.slug: r for r in rows}
        self.engine = object()
        self.calls = []

    def get(self, slug):
        self.calls.append(slug)
        return self.rows.get(slug)

    def list_all(self):
        raise AssertionError("list_all read every profile")


class TagStore:
    def __init__(self):
        self.calls = []

    def tags_for(self, bot):
        self.calls.append(bot)
        return ["coder"]

    def tags_by_bot(self):
        raise AssertionError("tags_by_bot read every assignment")


def _reader(rows):
    profiles, tags = ProfileStore(rows), TagStore()

    def read(config, slug):
        return load_named_bot(config, slug, profile_store=profiles, tag_store=tags)

    return read, profiles, tags


class Config(SimpleNamespace):
    def model_copy(self, deep=False):
        return self


def _service(slug=SLUG):
    return SimpleNamespace(config=Config(), strict_reader_bot=slug)


def _request(bot=SLUG, model="sample-endpoint", user="sample-user"):
    return SimpleNamespace(bot_id=bot, model=model, user=user, augment_memory=False, extract_memory=False)


class FakeInstance:
    def __init__(self, *, admitted_bot, **kwargs):
        self.admitted_bot = admitted_bot
        self.kwargs = kwargs
        self.history_manager = SimpleNamespace(extract_memory=None)


@pytest.fixture
def endpoint_ok(monkeypatch):
    seen = []

    def validate(service, bot, model):
        seen.append((bot.slug, model))
        return "sample-endpoint-ref"

    monkeypatch.setattr(strict, "validate_bound_endpoint", validate)
    return seen


def test_named_read_touches_only_one_profile_and_its_tags():
    read, profiles, tags = _reader([_row(), _row("other-agent")])
    bot = read(None, " Sample-Agent ")
    assert bot.slug == SLUG and bot.tags == ["coder"]
    assert profiles.calls == [SLUG] and tags.calls == [SLUG]


def test_cold_invalidated_registry_strict_request_builds_from_named_profile(endpoint_ok):
    read, profiles, _ = _reader([_row()])
    model, instance = strict.create_strict_instance(
        _service(), _request(), read_bot=read, instance_type=FakeInstance,
    )
    assert model == "sample-endpoint-ref"
    assert instance.admitted_bot.slug == SLUG
    assert instance.kwargs["bot_id"] == SLUG
    assert endpoint_ok == [(SLUG, "sample-endpoint")]
    assert profiles.calls == [SLUG]


def test_missing_profile_refuses_without_fallback(endpoint_ok):
    read, *_ = _reader([])
    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(_service(), _request(), read_bot=read, instance_type=FakeInstance)
    assert endpoint_ok == []


def test_mismatched_profile_refuses(endpoint_ok):
    def read(config, slug):
        return Bot(slug="other-agent", name="Other", description="", system_prompt="")

    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(_service(), _request(), read_bot=read, instance_type=FakeInstance)


@pytest.mark.parametrize("req", [_request(model=None), _request(user=None), _request(bot="other-agent")])
def test_missing_inputs_or_other_bot_refuse(endpoint_ok, req):
    read, profiles, _ = _reader([_row()])
    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(_service(), req, read_bot=read, instance_type=FakeInstance)
    assert profiles.calls == []


def test_invalid_endpoint_refuses(monkeypatch):
    def validate(service, bot, model):
        raise ValueError("Bound execution endpoint no longer matches backend")

    monkeypatch.setattr(strict, "validate_bound_endpoint", validate)
    read, *_ = _reader([_row()])
    with pytest.raises(ValueError):
        strict.create_strict_instance(_service(), _request(), read_bot=read, instance_type=FakeInstance)


def test_selection_routes_strict_before_any_ordinary_resolution(monkeypatch):
    calls = []
    monkeypatch.setattr(selection, "create_strict_instance", lambda s, r: calls.append("strict") or ("m", "inst"))
    for name in ("prompt_delivery_record", "routed_delivery_record"):
        monkeypatch.setattr(selection, name, lambda *a: (_ for _ in ()).throw(AssertionError(name)))
    service = _service()
    service._resolve_request_model = lambda *a: (_ for _ in ()).throw(AssertionError("ordinary"))
    assert selection.select_stream_execution(service, _request(), SLUG, True) == ("m", [], "inst", None)
    assert calls == ["strict"]


@pytest.mark.parametrize("configured", [None, "", "other-agent"])
def test_unconfigured_or_other_bot_keeps_ordinary_path(monkeypatch, configured):
    monkeypatch.setattr(selection, "create_strict_instance", lambda *a: (_ for _ in ()).throw(AssertionError("strict")))
    monkeypatch.setattr(selection, "prompt_delivery_record", lambda *a: None)
    monkeypatch.setattr(selection, "routed_delivery_record", lambda *a: None)
    service = SimpleNamespace(config=None, strict_reader_bot=configured,
                              _resolve_request_model=lambda *a: ("ordinary-model", ["w"]))
    assert selection.select_stream_execution(service, _request(), SLUG, True) == ("ordinary-model", ["w"], None, None)


def _bare_strict_instance(bot):
    inst = object.__new__(strict.StrictReaderLLMBawt)
    inst._admitted_bot = bot
    inst.client = SimpleNamespace()
    inst.resolved_model_alias = "sample-endpoint-ref"
    inst.verbose = False
    return inst


def test_strict_instance_init_bot_uses_admitted_profile_only():
    bot = Bot(slug=SLUG, name="Sample", description="", system_prompt="")
    inst = _bare_strict_instance(bot)
    inst.bot_id = SLUG
    BaseLLMBawt._init_bot(inst, None)
    assert inst.bot is bot and inst.client.bot_name == "Sample"
    assert inst._lookup_bot(None, SLUG) is bot


def test_strict_instance_has_no_other_bot_or_default_fallback():
    inst = _bare_strict_instance(Bot(slug=SLUG, name="Sample", description="", system_prompt=""))
    inst.bot_id = "other-agent"
    with pytest.raises(strict.StrictReaderRefused):
        BaseLLMBawt._init_bot(inst, None)
    with pytest.raises(strict.StrictReaderRefused):
        inst._fallback_bot(None)


# -- turn assembly: skills and roster stay scoped to the admitted bot --------

from contextlib import contextmanager  # noqa: E402

from llm_bawt.bot_tags import facade as tag_facade  # noqa: E402
from llm_bawt.core import prompt_sections  # noqa: E402
from llm_bawt.skills import service as skill_service  # noqa: E402
from llm_bawt.skills.resolver import Assignment  # noqa: E402

ROWS = [
    Assignment("global", "*", "shared-skill", "inject", 0, None),
    Assignment("bot", SLUG, "own-skill", "inject", 1, None),
    Assignment("tag", "coder", "role-skill", "inject", 2, None),
    Assignment("bot", "other-agent", "foreign-skill", "inject", 0, None),
    Assignment("tag", "reviewer", "foreign-role-skill", "inject", 0, None),
]


class ScopedStore:
    """Emulates rows_for's WHERE clause; any unscoped read fails."""

    def __init__(self):
        self.asked = []

    @contextmanager
    def read(self):
        yield "conn"

    def rows_for(self, conn, slug, tags):
        self.asked.append((slug, frozenset(tags)))
        return [r for r in ROWS if r.scope_type == "global"
                or (r.scope_type == "bot" and r.scope_id == slug)
                or (r.scope_type == "tag" and r.scope_id in tags)]

    def rows(self, conn):
        raise AssertionError("every skill assignment was read")


class Source:
    kind = "synthetic"

    def get(self, ref):
        return SimpleNamespace(root=f"/skills/{ref}")

    def read_body(self, ref):
        return SimpleNamespace(tokens=10, content_sha=ref, text=f"BODY:{ref}")

    def list(self):
        return []

    def complete(self):
        return True


class Builder:
    def __init__(self):
        self.sections = {}

    def add_section(self, name, content, **_k):
        self.sections[name] = content


@pytest.fixture
def skills_world(monkeypatch):
    store = ScopedStore()
    real_init = skill_service.SkillService.__init__

    def init(self, engine, **kwargs):
        real_init(self, engine, **kwargs)
        self.store, self.source = store, Source()

    monkeypatch.setattr(skill_service.SkillService, "__init__", init)
    monkeypatch.setattr(skill_service.SkillService, "_is_migrated", lambda self, conn: True)
    monkeypatch.setattr(skill_service.SkillService, "_world",
                        lambda *a: (_ for _ in ()).throw(AssertionError("skill world read")))
    monkeypatch.setattr(skill_service, "caps_from_settings",
                        lambda config: skill_service.Caps(per_skill=1000, total=1000))
    monkeypatch.setattr(tag_facade.BotTags, "snapshot",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tag snapshot read")))
    import llm_bawt.utils.db as db
    monkeypatch.setattr(db, "get_shared_engine", lambda config: "engine")
    return store


def _assembling_instance():
    bot = Bot(slug=SLUG, name="Sample", description="", system_prompt="", tags=["coder"],
              bot_type="agent", agent_backend="claude-code")
    inst = _bare_strict_instance(bot)
    inst.bot_id, inst.bot, inst.config = SLUG, bot, SimpleNamespace()
    inst.config_resolver = SimpleNamespace(resolve_body=lambda key, **k: SimpleNamespace(
        body="{body}" if key.endswith("entry") or "entry" in key else "", source="t"))
    return inst


def test_strict_turn_assembly_resolves_only_admitted_skills_and_omits_roster(skills_world, monkeypatch):
    monkeypatch.setattr("llm_bawt.bot_tags.context.load_tag_context",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("family roster read")))
    inst = _assembling_instance()
    builder = Builder()
    inst._sec_tag_context(builder, "")
    inst._sec_injected_skills(builder, "")
    assert "tag_context" not in builder.sections
    refs = {e.ref for e in inst._skill_resolution.injected}
    assert refs == {"shared-skill", "own-skill", "role-skill"}
    body = builder.sections["injected_skills"]
    assert "BODY:own-skill" in body and "BODY:role-skill" in body
    assert "foreign" not in body
    assert skills_world.asked == [(SLUG, frozenset({"coder"}))]


def test_strict_skills_refuse_before_migration(skills_world, monkeypatch):
    monkeypatch.setattr(skill_service.SkillService, "_is_migrated", lambda self, conn: False)
    inst = _assembling_instance()
    builder = Builder()
    inst._sec_injected_skills(builder, "")
    assert inst._skill_resolution is None and "injected_skills" not in builder.sections


def test_private_admitted_bot_gets_no_skills(skills_world):
    inst = _assembling_instance()
    inst._admitted_bot.include_in_global_search = False
    builder = Builder()
    inst._sec_injected_skills(builder, "")
    assert "injected_skills" not in builder.sections


def test_ordinary_turn_assembly_keeps_registry_resolution():
    seen = []

    class OrdinarySvc:
        @classmethod
        def from_config(cls, engine, config):
            seen.append(engine)
            return SimpleNamespace(resolve_bot=lambda slug: seen.append(slug) or "resolution")

    ordinary = object.__new__(prompt_sections.PromptSectionsMixin)
    ordinary.config, ordinary.bot_id = None, SLUG
    assert ordinary._tag_context_enabled() is True
    assert ordinary._resolve_skills(OrdinarySvc, "engine") == "resolution"
    assert seen == ["engine", SLUG]
