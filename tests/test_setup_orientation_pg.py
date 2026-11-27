"""First-bot orientation admission against the production create transaction."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from llm_bawt.bot_create_transaction import BotCreateTransaction
from llm_bawt.core.prompt_builder import PromptBuilder, SectionPosition
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_ledger import delivery_audience
from llm_bawt.memory.model_catalog_migration import migrate_model_catalog
from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.message_authorship import AuthorReference, SETUP_ORIENTATION_AUTHOR_ID
from llm_bawt.prompt_registry import get_prompt_resolver
from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.inter_bot_claims import validate_inter_bot_claim
from llm_bawt.service.prompt_stale import STALE_KEY
from llm_bawt.service.routes.setup import _write_first_bot_settings
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.service.setup_orientation import (
    ORIENTATION_PROMPT_KEY, SETTING_KEY, profile_name_line, record_choice,
)
from llm_bawt.service.system_prompt_pin import COMPACTION, SystemPromptPinStore, apply_session_pin
from llm_bawt.service.turn_admission import lock_target
from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.task_turn_context import TaskTurnContext
from llm_bawt.working_memory.service import WorkingMemoryService
from llm_bawt.working_memory.turn import WorkingMemoryTurn

pytestmark = pytest.mark.integration


def test_two_setup_submissions_commit_only_one_first_chat(private_app_pg):
    profiles = BotProfileStore(private_app_pg)
    settings = RuntimeSettingsStore(private_app_pg)
    deliveries = InterBotDeliveryStore(private_app_pg)
    assert profiles.engine is settings.engine is deliveries.engine
    migrate_model_catalog(profiles.engine)
    with profiles.engine.begin() as conn:
        model = conn.execute(text("""
            INSERT INTO models (key, vendor, display_name, default_context_window)
            VALUES ('orientation-model', 'anthropic', 'Orientation model', 10000) RETURNING id
        """)).scalar_one()
        endpoint = conn.execute(text("""
            INSERT INTO model_endpoints (model_id, access_path_id)
            SELECT :m, id FROM access_paths WHERE key='anthropic-api' RETURNING id
        """), {"m": model}).scalar_one()

    guard = SimpleNamespace(lock=lock_target, check=lambda *_: None)
    start = Barrier(2, timeout=10)

    def create(slug, *, opted_in=True, fail=False, race=False):
        def on_create(conn, profile):
            _write_first_bot_settings(conn, private_app_pg, profile.slug, {"tenant.voice_enabled": False})
            record_choice(conn, profile, opted_in=opted_in, user_id=f"user-{slug}",
                          prompt="Welcome", store=deliveries)
            if fail:
                raise RuntimeError("rollback after enqueue")

        payload = {"slug": slug, "name": slug, "system_prompt": "Be helpful",
                   "bot_type": "agent", "agent_backend": "claude-code", "harness": "claude-code",
                   "default_model": "orientation-model", "endpoint_id": endpoint}
        if race:
            start.wait()
        return BotCreateTransaction(profiles).create(payload, {}, guard, lambda *_: None, on_create=on_create)

    with pytest.raises(RuntimeError, match="rollback after enqueue"):
        create("rollback", fail=True)
    assert profiles.get("rollback") is None
    assert not deliveries.list()
    globals_after_rollback = settings.get_scope_settings("global", "*")
    assert not {SETTING_KEY, "default_bot", "tenant.voice_enabled"} & globals_after_rollback.keys()

    with profiles.engine.begin() as conn:
        assert profile_name_line(conn, "user-introduction-a") == ""
        for user in ("user-introduction-a", "user-introduction-b"):
            conn.execute(text("""
                INSERT INTO entity_profiles (entity_type, entity_id, display_name, created_at, updated_at)
                VALUES ('USER', :user, '  Avery   Example ', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """), {"user": user})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(create, "introduction-a", race=True)
        second = pool.submit(create, "introduction-b", race=True)
        assert {first.result(timeout=20).slug, second.result(timeout=20).slug} == {"introduction-a", "introduction-b"}
    rows = deliveries.list()
    assert len(rows) == 1
    row = rows[0]
    target = row.target_bot_id
    with pytest.raises(HTTPException, match="409"):
        create(target)
    assert [d.id for d in deliveries.list()] == [row.id]
    global_settings = settings.get_scope_settings("global", "*")
    assert global_settings["default_bot"] == target
    assert global_settings["tenant.voice_enabled"] is False
    state = global_settings[SETTING_KEY]
    assert state == {"bot_id": target, "opted_in": True, "milestone": "first_chat", "delivery_id": row.id}
    assert delivery_audience(row, default_user="unrelated").user_id == f"user-{target}"
    assert "orientation-model, connection anthropic-api" in row.message
    # The model's welcome cue states the profile name next to its instruction,
    # but the ledger message and its compact preview stay name-free even with a
    # short cue like this one (TASK-2643).
    cue = deliveries.payload(row.id)["messages"][0]["content"]
    assert cue.startswith("Welcome\n\nTheir profile name: Avery Example.\n\nChecked in setup:")
    assert row.message.startswith("Welcome\n\nChecked in setup:")
    assert "Avery" not in row.message
    assert "Avery" not in str(row.to_compact())
    assert deliveries.payload(row.id)["prefer_steer"] is False
    # Not local mode: the welcome turn must read the profile name (TASK-2643).
    assert deliveries.payload(row.id)["augment_memory"] is True
    assert deliveries.payload(row.id)["extract_memory"] is False

    claimed = deliveries.claim_next(target, claim_owner="fixture", steer_capable=False)
    assert claimed is not None and claimed.id == row.id
    request = ChatCompletionRequest.model_validate({
        **deliveries.payload(row.id), "inter_bot_claim_token": claimed.claim_token,
    })
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=deliveries))
    author = validate_inter_bot_claim(service, request)
    assert author.entity_type == "system" and author.entity_id == SETUP_ORIENTATION_AUTHOR_ID

    _check_onboarding_prompt(private_app_pg, settings, row, author)

    # A second choice cannot replace the committed milestone, even opt-out.
    create("later-bot", opted_in=False)
    assert settings.get_scope_settings("global", "*")[SETTING_KEY] == state
    assert len(deliveries.list()) == 1


def _check_onboarding_prompt(config, settings, delivery, setup_author):
    bot = delivery.target_bot_id
    user = f"user-{bot}"
    backend = PostgreSQLMemoryBackend(config, bot_id=bot)
    session, separate, other_user = (str(uuid4()) for _ in range(3))
    with settings.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status)
            VALUES (:session, :bot, :user, 'active'), (:separate, :bot, :user, 'archived'),
                   (:other, :bot, 'another-owner', 'active')
        """), {"bot": bot, "user": user, "session": session, "separate": separate, "other": other_user})

    bawt = ServiceLLMBawt.__new__(ServiceLLMBawt)
    bawt.config, bawt.bot_id, bawt.user_id = config, bot, user
    bawt._refresh_history_if_stale = lambda: None
    bawt.debug = False
    bawt.model_definition = {"type": "agent_backend"}
    bawt._inject_user_prefix = bawt._tts_mode = False
    base = PromptBuilder().add_section("base", "Base persona", position=SectionPosition.BASE_PROMPT)
    bawt._assemble_system_builder = lambda _prompt: base.copy()

    def render(sid, prompt="Help me choose", *, mid=None, author=None):
        def persist(role, body, **kwargs):
            backend.add_message(kwargs["message_id"] or str(uuid4()), role, body, 1,
                                session_id=sid, author=kwargs["author"], extract_memory=False)

        bawt.history_manager = SimpleNamespace(add_message=persist)
        return bawt.prepare_messages_for_query(
            prompt, message_id=mid, author=author or AuthorReference.user(bawt.user_id),
            orientation_session_id=sid,
        )

    registry = get_prompt_resolver(config)
    assert registry.definition_for(ORIENTATION_PROMPT_KEY) is not None
    registry.store.upsert(key=ORIENTATION_PROMPT_KEY, body="Tenant orientation")
    assert render(session)[0].content == "Base persona"  # choice alone is not a thread binding
    registry.store.upsert(key=ORIENTATION_PROMPT_KEY, scope_type="bot", scope_id=bot,
                          body="First orientation edit")
    welcome = render(session, delivery.message, mid=delivery.user_message_id, author=setup_author)
    assert welcome[0].content == "Base persona\n\nFirst orientation edit"
    assert welcome[-1].content == delivery.message  # welcome template remains separate
    assert render(session)[0].content == welcome[0].content

    registry.store.upsert(key=ORIENTATION_PROMPT_KEY, scope_type="bot", scope_id=bot,
                          body="Second orientation edit")
    later = render(session)
    assert later[0].content == "Base persona\n\nSecond orientation edit"
    assert later[-1].content == "Help me choose"
    assert render(separate)[0].content == "Base persona"
    bawt.user_id = "another-owner"
    assert render(other_user)[0].content == "Base persona"
    assert render(session)[0].content == "Base persona"
    bawt.user_id, bawt.bot_id = user, "another-bot"
    assert render(session)[0].content == "Base persona"
    bawt.bot_id = bot
    assert render(session)[0].content == later[0].content  # no bleed on the cached builder

    with settings.engine.begin() as conn:
        old_value = conn.execute(text("""
            SELECT value_json FROM runtime_settings WHERE scope_type='global' AND scope_id='*' AND key=:key
        """), {"key": SETTING_KEY}).scalar_one()
        conn.execute(text("""
            UPDATE runtime_settings SET value_json=:value
            WHERE scope_type='global' AND scope_id='*' AND key=:key
        """), {"value": json.dumps({**json.loads(old_value), "opted_in": False}), "key": SETTING_KEY})
    assert render(session)[0].content == "Base persona"
    with settings.engine.begin() as conn:
        conn.execute(text("""
            UPDATE runtime_settings SET value_json=:value
            WHERE scope_type='global' AND scope_id='*' AND key=:key
        """), {"value": old_value, "key": SETTING_KEY})

    _check_orientation_pin(bawt, settings, registry, session, render)


def _check_orientation_pin(bawt, settings, registry, session, render):
    bot, user = bawt.bot_id, bawt.user_id
    settings.set_value("bot", bot, "working_memory_enabled", True)
    memory_service = WorkingMemoryService(settings.engine)
    memory_service.store.ensure_schema()
    task_context = TaskTurnContext(
        session_id=session, turn_id=f"turn-{uuid4()}", trigger_message_id=str(uuid4()),
        bot_id=bot, user_id=user, issued_at=0,
    )
    item = memory_service.bot_create(task_context, {"objective": "Initial memory"}, str(uuid4()))["item"]

    # Use the real assembler and managed-memory section; omit unrelated manifest sections.
    del bawt._assemble_system_builder
    bawt._prompt_builder = PromptBuilder()
    bawt._prompt_builder.add_section("base", "Base persona", position=SectionPosition.BASE_PROMPT)
    bawt._prompt_builder.add_section("global", "Global rules", position=SectionPosition.GLOBAL_INSTRUCTIONS)
    bawt._walk_manifest = lambda builder, prompt, _stage: bawt._sec_working_memory(builder, prompt)
    bawt.client = SimpleNamespace(
        _backend=SimpleNamespace(pins_system_prompt=True), compose_system_prompt=lambda base: base,
    )
    turns = TurnLogStore(bawt.config)
    store = SystemPromptPinStore(turns.engine)
    binding = {"thread_session_id": session, "explicit_thread": True}
    ctx = SimpleNamespace(
        svc=SimpleNamespace(_turn_log_store=turns), request=SimpleNamespace(), llm_bawt=bawt,
        bot_id=bot, user_id=user, is_agent_backend=True, bridge_reset_action=None, thread_binding=binding,
    )

    def turn(prompt="Help me choose", *, compact_succeeded=False):
        ctx.user_prompt = prompt
        wm = WorkingMemoryTurn.begin(ctx)
        assert wm.render and wm.render.state == "ok"
        trigger_id = str(uuid4())
        messages = render(session, prompt, mid=trigger_id)
        turn_id = f"turn-{uuid4()}"
        with turns.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                       user_prompt, trigger_message_id)
                VALUES (:id, clock_timestamp(), '/test', true, :bot, :user, 'ok', :prompt, :trigger)
            """), {"id": turn_id, "bot": bot, "user": user, "prompt": prompt, "trigger": trigger_id})
        outbound, decision = apply_session_pin(
            ctx, bawt.client, messages, turn_id=turn_id, sections=wm.sections, working_memory_text=wm.render.text,
        )
        assert decision is not None
        if decision.is_boundary:
            assert store.confirm(session_id=session, turn_id=turn_id)
        if compact_succeeded:  # usage is recorded only after the compaction call runs
            with turns.engine.begin() as conn:
                conn.execute(text("UPDATE turn_logs SET token_usage_json=:usage WHERE id=:id"), {
                    "usage": json.dumps({"context_action": "compact", "context_action_outcome": "success"}),
                    "id": turn_id,
                })
        return decision, outbound[0].content, messages[0].content, wm.render.text

    def stale():
        with turns.engine.connect() as conn:
            metadata = conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:sid"),
                                    {"sid": session}).scalar_one()
        return (metadata or {}).get(STALE_KEY)

    initial, pinned, _, _ = turn()
    assert initial.is_boundary and "Second orientation edit" in pinned
    binding["thread_resume_id"] = "fixture-resume"
    memory_service.owner_update(bot, user, item["id"], item["revision"], {"objective": "Changed memory"})
    kept, outbound, rendered, memory_text = turn()
    assert "Changed memory" in memory_text and rendered != pinned
    assert (kept.action, kept.pin_turn_id, outbound) == ("kept", initial.pin_turn_id, pinned)
    assert stale() is None
    assert rendered.startswith("Base persona\n\nGlobal rules\n\nSecond orientation edit\n\n")
    assert rendered.endswith("\n\n" + memory_text)

    registry.store.upsert(key=ORIENTATION_PROMPT_KEY, scope_type="bot", scope_id=bot,
                          body="## Setup: Orientation\nUpdated guidance")
    held, outbound, _, _ = turn()
    assert held.action == "kept" and outbound == pinned
    marker = stale()
    assert marker and marker["comparison"] == "instructions" and "Setup: Orientation" in marker["sections"]
    compact, outbound, _, _ = turn("/compact", compact_succeeded=True)
    assert compact.action == "kept" and outbound == pinned
    refreshed, outbound, rendered, _ = turn()
    assert refreshed.reason == COMPACTION and not refreshed.snapshot_unverified
    assert outbound == rendered and "Updated guidance" in outbound and "Changed memory" in outbound
    assert stale() is None
