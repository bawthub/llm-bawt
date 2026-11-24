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
