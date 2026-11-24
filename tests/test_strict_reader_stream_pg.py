"""TASK-2590: one streamed strict-reader turn on private PostgreSQL.

Synthetic identities only. The shared bot registry is cold and invalidated and
every shared read is fatal. The real named reader reads the admitted profile
from PostgreSQL; the real streaming path admits, streams and persists the turn
with the strict instance built by its real constructor. Context assembly is
production: the real prompt manifest walk and Skills section, resolving
synthetic ``skill_assignments`` rows through real SQL against a synthetic skill
source. Only the remote completion (a recording stream client), the model
catalog entry and the history transport (the harness's PG-backed history) are
faked. A foreign profile and foreign rows exist in the same database and never
reach the turn.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt import bots
from llm_bawt.bot_tags import catalog as tag_catalog
from llm_bawt.bot_tags.errors import TagConflict
from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service import strict_reader_execution as strict
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.skills import service as skill_service
from llm_bawt.skills.resolver import Assignment
from llm_bawt.skills.store import SkillAssignmentStore
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_message_native_upsert import capture_sql
from tests.test_message_native_upsert_pg import http_runtime, real_storage  # noqa: F401
from tests.test_native_selected_owner_pg import native_runtime  # noqa: F401

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("private_app_pg", "persisted_master_key"),
]

TAG = "fx_strict_role"
FOREIGN_TAG = "fx_foreign_role"
#: ref -> (scope_type, scope, mode); "admitted"/"foreign" resolve to slugs.
ROWS = {
    "fx-global-skill": ("global", "*", "inject"),
    "fx-hidden-skill": ("global", "*", "disable"),
    "fx-admitted-skill": ("bot", "admitted", "inject"),
    "fx-tagged-skill": ("tag", TAG, "inject"),
    "fx-foreign-skill": ("bot", "foreign", "inject"),
    "fx-foreign-tag-skill": ("tag", FOREIGN_TAG, "inject"),
}
APPROVED = ("fx-global-skill", "fx-admitted-skill", "fx-tagged-skill")
FOREIGN = ("fx-foreign-skill", "fx-foreign-tag-skill")


def _body(ref: str) -> str:
    return f"SYNTHETIC-BODY-{ref.upper()}"


def _forbid_shared_reads(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("shared bot registry was read")

    monkeypatch.setattr(bots, "_LOADED", False)
    monkeypatch.setattr(bots, "_FORCE_RELOAD", True)
    monkeypatch.setattr(bots, "_RETRY_AFTER", 0.0)
    for name in ("_load_bots_config", "_load_db_bot_profiles", "_load_bot_tags"):
        monkeypatch.setattr(bots, name, boom)
    monkeypatch.setattr(bots.BotManager, "__init__", boom)
    monkeypatch.setattr(BotProfileStore, "__init__", boom)
    monkeypatch.setattr(BotTagStore, "tags_by_bot", boom)
    monkeypatch.setattr(BotTagStore, "snapshot", boom, raising=False)


def _request(rig, **over):
    body = {"model": "fixture-native", "bot_id": rig.bot, "user": rig.user, "stream": True,
            "augment_memory": False, "user_message_id": str(uuid.uuid4()),
            "assistant_message_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": "hello strict"}], **over}
    return ChatCompletionRequest.model_validate(body)


def _stream(service, request):
    async def run():
        return [chunk async for chunk in service.chat_completion_stream(request)]
    return "".join(asyncio.run(run()))


HOUSE = "fx-strict-house"
GENERATION = "gen-fxcandidate"
LAYERS = {"fx-source": "a" * 40}


def write_skill_tree(root, body=None):
    body = body or _body
    root.mkdir(parents=True, exist_ok=True)
    for ref in ROWS:
        skill = root / ref
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            f"---\nname: {ref}\ndescription: synthetic {ref}\n---\n{body(ref)}\n")


def _seed_skills(rig, root, foreign):
    tags = BotTagStore(rig.engine)
    for slug in (TAG, FOREIGN_TAG):
        try:
            tags.create_role(slug, f"Fixture {slug}")
        except TagConflict:  # the private database outlives one test
            pass
    store = SkillAssignmentStore(rig.engine)
    with store.write() as conn:
        tag_catalog.assign(conn, rig.bot, [TAG], "fixture")
        tag_catalog.assign(conn, foreign, [FOREIGN_TAG], "fixture")
        for position, (ref, (scope_type, scope, mode)) in enumerate(ROWS.items()):
            scope = {"admitted": rig.bot, "foreign": foreign}.get(scope, scope)
            store.upsert(conn, Assignment(scope_type=scope_type, scope_id=scope, skill_ref=ref,
                                          mode=mode, position=position, updated_by="fixture"))


@pytest.fixture
def strict_runtime(native_runtime, rig, monkeypatch, tmp_path):  # noqa: F811
    from llm_bawt import model_catalog

    profiles = BotProfileStore(rig.config)
    foreign = rig.bot + "x"  # created by http_runtime; must never be read

    def admit(**over):
        profiles.upsert({"slug": rig.bot, "name": "Sample Strict", "bot_type": "agent",
                         "agent_backend": "claude-code", "include_in_global_search": True,
                         "house": HOUSE, **over})

    admit()
    # A synthetic skill-sources store with one immutable candidate generation.
    store = tmp_path / "skill-store"
    root = store / "generations" / GENERATION
    write_skill_tree(root)
    import json as _json
    (root / ".layers.json").write_text(_json.dumps(LAYERS))
    monkeypatch.setenv("SKILL_SOURCES_ROOT", str(store))
    _seed_skills(rig, root, foreign)
    # Synthetic catalog only: the operator's real skill checkout is never read.
    synthetic = skill_service.AmbientSource((root,))
    monkeypatch.setattr(skill_service, "AmbientSource", lambda *a, **k: synthetic)

    rt = native_runtime
    built = []
    # Fake remote completion: the harness's recording stream client, on the
    # native stream path. The instance itself is an agent build (claude-code
    # catalog type), so the real agent prompt manifest runs.
    real_resolve = model_catalog.resolve_model_config

    def resolve_model(config, alias, *a, **k):
        if alias == "fixture-native":
            return {"type": "claude-code", "model_id": "fixture-native"}
        return real_resolve(config, alias, *a, **k)

    def fake_client(self):
        built.append(self)
        return SimpleNamespace(**vars(rt.llm.client))

    def harness_history(self):
        # History transport: the harness's real PG-backed HistoryManager.
        self.history_manager = rt.llm.history_manager

    monkeypatch.setattr(model_catalog, "resolve_model_config", resolve_model)
    monkeypatch.setattr(strict.StrictReaderLLMBawt, "_initialize_client", fake_client)
    monkeypatch.setattr(strict.StrictReaderLLMBawt, "_init_history", harness_history)
    monkeypatch.setattr(strict, "validate_bound_endpoint",
                        lambda service, bot, model: model if bot.slug == rig.bot else None)
    rt.service.strict_reader_bot = rig.bot
    rt.service.strict_reader_house = HOUSE
    rt.service.strict_reader_skill_generation = GENERATION
    rt.service.strict_reader_skill_layers = dict(LAYERS)
    rt.service.config = rig.config  # the real PG config for the named reader
    rt.service._get_llm_bawt.side_effect = AssertionError("ordinary instance used")
    _forbid_shared_reads(monkeypatch)
    skill_service._MIGRATED.clear()
    return SimpleNamespace(rt=rt, built=built, foreign=foreign, admit=admit, store=store)


def _system(call) -> str:
    return "\n".join(str(getattr(m, "content", m.get("content") if isinstance(m, dict) else ""))
                     for m in call
                     if (getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None))
                     == "system")


def test_streamed_strict_turn_reads_only_admitted_profile_and_seals_tools(strict_runtime, rig):
    rt = strict_runtime.rt
    with capture_sql(rig.engine) as sql:
        text = _stream(rt.service, _request(rig))
    assert "native " in text and "answer" in text and "[DONE]" in text
    assert len(strict_runtime.built) == 1
    instance = strict_runtime.built[0]
    assert type(instance) is strict.StrictReaderLLMBawt
    assert instance.bot.slug == rig.bot and instance.client.tools_disabled is True
    assert len(rt.model_calls) == 1
    assert [role for role, _, _ in rt.appends] == ["user", "assistant"]

    # Real context assembly: approved bodies reach the model, foreign ones never.
    system = _system(rt.model_calls[0])
    for ref in APPROVED:
        assert _body(ref) in system, ref
    for ref in (*FOREIGN, "fx-hidden-skill"):
        assert _body(ref) not in system, ref
    manifest = instance._skill_resolution.manifest()
    assert {e["ref"] for e in manifest["injected"]} == set(APPROVED)
    assert manifest["disabled"] == ["fx-hidden-skill"]
    # Only the synthetic catalog; no ambient operator skill.
    assert {e["ref"] for e in manifest["entries"]} == set(ROWS) - {"fx-hidden-skill"}
    assert set(instance._injected_skill_refs) == set(APPROVED)

    profile_reads = [s for s in sql.all_statements if "FROM bot_profiles" in s]
    assert profile_reads, sql.all_statements
    assert all("bot_profiles.slug =" in s or "slug =" in s for s in profile_reads), profile_reads
    skill_reads = [s for s in sql.all_statements if "FROM skill_assignments" in s]
    # Only the scoped query (global, own, own tags); never the all-rows read.
    assert skill_reads and all("scope_type = 'global'" in s and "ANY(" in s
                               for s in skill_reads), skill_reads
    assert not any(strict_runtime.foreign in s for s in sql.all_statements)
    assert not any("FROM bot_profile_tags GROUP BY" in s for s in sql.all_statements)


def test_streamed_strict_turn_refuses_when_skills_cannot_resolve(strict_runtime, rig):
    rt = strict_runtime.rt
    strict_runtime.admit(include_in_global_search=False)  # private: scoped resolution refuses
    text = _stream(rt.service, _request(rig))
    assert "private bot; scoped resolution unsupported" in text or "stream_interrupted" in text
    assert "native " not in text and "answer" not in text
    assert len(strict_runtime.built) == 1  # the strict instance refused while assembling
    assert rt.model_calls == []
    assert [role for role, _, _ in rt.appends if role == "assistant"] == []


@pytest.mark.parametrize("over", [{"session_id": "explicit"}, {"inter_bot_delivery_id": "d-1"}],
                         ids=["explicit-thread", "inter-bot"])
def test_unsupported_strict_turns_fail_closed_before_model(strict_runtime, rig, over):
    rt = strict_runtime.rt
    if "session_id" in over:
        over = {"session_id": rig.session_id}
    with pytest.raises(Exception):
        _stream(rt.service, _request(rig, **over))
    assert rt.model_calls == [] and strict_runtime.built == []


def test_ordinary_turn_unchanged_when_strict_unset(native_runtime, rig):  # noqa: F811
    native_runtime.service.strict_reader_bot = None
    text = _stream(native_runtime.service, _request(rig, session_id=rig.session_id))
    assert "native " in text and "answer" in text
    assert native_runtime.service._get_llm_bawt.called
    assert getattr(native_runtime.llm.client, "tools_disabled", False) is False
