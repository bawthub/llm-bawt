"""Hermetic producer/outbox/dispatcher integration; no external services.

SQLite exercises transaction rollback and shared gates, not PostgreSQL locking.
The transport store shim persists receipts but never calls a bot or Redis.
"""
import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlmodel import Session

from llm_bawt.inter_bot_delivery import DeliveryRecord, InterBotDeliveryStore
from llm_bawt.inter_bot_preclaim import load_claim_head
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.prompt_capabilities import resolve_prompt_target
from llm_bawt.service.prompt_delivery import PromptDeliveryWorker
from llm_bawt.service.prompt_delivery_state import receipt
from llm_bawt.service.prompt_schedule_models import PromptOccurrence, PromptSchedule
from llm_bawt.service.prompt_timing import PromptTiming
from llm_bawt.service.scheduler import JobRun, create_scheduler_tables
from tests.sqlite_support import fast_sqlite_engine

NOW = datetime.now(timezone.utc)
BOT = "disposable-schedule-bot"
OWNER = "test-owner"


class LocalDeliveryStore:
    """Only transport I/O is substituted; outbox and claim gates are real."""
    def __init__(self, engine):
        self.engine = engine
        self.calls = 0
        self.failure = None

    def get(self, delivery_id):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                               {"id": delivery_id}).mappings().first()
            return DeliveryRecord.from_mapping(row) if row else None

    def enqueue(self, **kwargs):
        self.calls += 1
        if self.failure == "before":
            raise RuntimeError("injected before enqueue")
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE idempotency_key=:key"),
                               {"key": kwargs["idempotency_key"]}).mappings().first()
            if row:
                return DeliveryRecord.from_mapping(row), True
            delivery_id, message_id, turn_id = InterBotDeliveryStore.stable_ids()
            payload = dict(kwargs["payload"], user_message_id=message_id,
                           inter_bot_delivery_id=delivery_id, inter_bot_turn_id=turn_id,
                           inter_bot_bridge_request_id="req_delivery_" + delivery_id[9:])
            values = dict(id=delivery_id, sender=kwargs["sender_bot_id"],
                          owner=kwargs["author"].entity_id, target=kwargs["target_bot_id"],
                          message=kwargs["message"], key=kwargs["idempotency_key"],
                          message_id=message_id, turn_id=turn_id,
                          payload=json.dumps(payload), meta=json.dumps(kwargs["metadata"]))
            conn.execute(text("""INSERT INTO inter_bot_deliveries
                (id,sender_bot_id,author_entity_type,author_entity_id,target_bot_id,message,
                 idempotency_key,user_message_id,turn_id,payload_json,metadata_json)
                VALUES (:id,:sender,'user',:owner,:target,:message,:key,:message_id,
                        :turn_id,:payload,:meta)"""), values)
        if self.failure == "after":
            raise RuntimeError("injected after committed enqueue")
        return self.get(delivery_id), False

    def payload(self, delivery_id):
        with self.engine.connect() as conn:
            return json.loads(conn.execute(text("SELECT payload_json FROM inter_bot_deliveries WHERE id=:id"),
                                           {"id": delivery_id}).scalar_one())

    def update(self, delivery_id, **fields):
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE inter_bot_deliveries SET " + ",".join(f"{k}=:{k}" for k in fields)
                              + " WHERE id=:id"), dict(fields, id=delivery_id))
        return self.get(delivery_id)

    def mark_transport_accepted(self, delivery_id, claim_token):
        self.update(delivery_id, transport_accepted_at=NOW.isoformat())
        return True

    def mark_delivered(self, delivery_id, claim_token, **kwargs):
        return self.update(delivery_id, status="DELIVERED", **kwargs)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import llm_bawt.bots as bots
    bot = bots.Bot(slug=BOT, name="Disposable", description="Test only", system_prompt="Test",
                   agent_backend="claude-code", harness="claude-code", default_model="first", endpoint_id=1)
    monkeypatch.setattr(bots.BotManager, "get_bot", lambda self, slug: bot if slug == BOT else None)
    catalog = ModelCatalog([
        ModelEndpoint(i, ModelIdentity(i, name, "anthropic", name),
                      AccessPath(1, "test-native", "anthropic", "anthropic-messages", None, "test"), name)
        for i, name in [(1, "first"), (2, "second")]
    ])
    config = SimpleNamespace(ensure_model_catalog=lambda: catalog)
    engine = fast_sqlite_engine(tmp_path / "delivery.db")
    create_scheduler_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT,
            status TEXT, started_at TEXT, ended_at TEXT, archived_at TEXT, session_metadata TEXT)"""))
        conn.execute(text("""CREATE TABLE turn_logs (id TEXT PRIMARY KEY, status TEXT, ended_at TEXT, error_text TEXT)"""))
        conn.execute(text("""CREATE TABLE inter_bot_deliveries (
            ordinal INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, sender_bot_id TEXT,
            author_entity_type TEXT, author_entity_id TEXT, target_bot_id TEXT, message TEXT,
            payload_json TEXT, metadata_json TEXT, user_message_id TEXT, turn_id TEXT,
            idempotency_key TEXT UNIQUE, status TEXT DEFAULT 'QUEUED', attempt_count INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 5, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP, available_at TEXT DEFAULT CURRENT_TIMESTAMP,
            transport_accepted_at TEXT, last_error TEXT, response_model TEXT, response_chars INTEGER,
            next_retry_at TEXT, claim_token TEXT, claim_owner TEXT, lease_expires_at TEXT,
            session_policy TEXT DEFAULT 'continue')"""))
        conn.execute(text("INSERT INTO sessions(id,bot_id,user_id,status) VALUES ('personal',:bot,:owner,'active')"),
                     {"bot": BOT, "owner": OWNER})
    delivery = LocalDeliveryStore(engine)
    turns = {}
    service = SimpleNamespace(config=config, _turn_log_store=SimpleNamespace(get_turn=turns.get))
    dispatcher = SimpleNamespace(store=delivery, _emit=AsyncMock(), wake=lambda target: None)
    service._inter_bot_dispatcher = dispatcher
    worker = PromptDeliveryWorker(engine, service)
    yield SimpleNamespace(engine=engine, bot=bot, catalog=catalog, service=service,
                          worker=worker, dispatcher=dispatcher, delivery=delivery, turns=turns)
    engine.dispose()


def reserve(rig, *, clear=True, once=False, now=NOW, grace=3600):
    timing = (PromptTiming(kind="once", once_at=now + timedelta(seconds=1)) if once else
              PromptTiming(kind="interval", interval_seconds=60, anchor_at=now + timedelta(seconds=1)))
    job = rig.worker.store.create(owner=OWNER, bot_id=BOT, name="Automation", prompt="Literal saved prompt",
                                  now=now, timing=timing, clear_context=clear,
                                  requested_model="second", misfire_grace_seconds=grace)
    run = rig.worker.store.reserve_due(now + timedelta(seconds=1))[0]
    return job, run


def enqueue(rig, run, now=NOW):
    prepared = rig.worker._prepare(run, now)
    return rig.worker._enqueue(run, prepared, rig.dispatcher, now)


def occurrence(rig, run):
    with Session(rig.engine) as session:
        return session.get(PromptOccurrence, run)


def mutate(rig, job, run, *, lifecycle=None, cancel=False):
    with Session(rig.engine) as session:
        if lifecycle:
            row = session.get(PromptSchedule, job)
            row.lifecycle = lifecycle
            session.add(row)
        if cancel:
            row = session.get(JobRun, run)
            row.result_json = json.dumps({**receipt(row), "cancel_requested": True})
            session.add(row)
        session.commit()


def test_validator_actual_catalog_default_explicit_and_harness(rig):
    assert resolve_prompt_target(rig.service, BOT, None, OWNER) == "first@test-native"
    assert resolve_prompt_target(rig.service, BOT, "second", OWNER) == "second@test-native"
    assert rig.bot.default_model == "first" and rig.bot.endpoint_id == 1
    with pytest.raises(ValueError, match="missing"):
        resolve_prompt_target(rig.service, "not-a-bot", None, OWNER)
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_prompt_target(rig.service, BOT, "removed", OWNER)
    rig.bot.harness = "codex"
    with pytest.raises(ValueError, match="isolation"):
        resolve_prompt_target(rig.service, BOT, None, OWNER)


@pytest.mark.parametrize("clear", [True, False])
def test_end_to_end_two_occurrences_sessions_model_authorship_and_completion(rig, clear):
    job, first = reserve(rig, clear=clear)
    first_record = enqueue(rig, first)
    first_session = occurrence(rig, first).session_id
    assert first_record.author == AuthorReference.user(OWNER)
    payload = rig.delivery.payload(first_record.id)
    assert payload["prefer_steer"] is False
    assert payload["model"] == "second@test-native"
    assert payload["session_id"] == first_session != "personal"
    assert payload["messages"] == [{"role": "user", "content": "Literal saved prompt"}]
    # Real dispatcher drains the ordinary service generator; only inference is mocked.
    calls = []
    async def stream(request):
        calls.append(request)
        rig.turns[first_record.turn_id] = SimpleNamespace(ended_at=NOW, status="ok", error_text=None,
                                                        response_text="done", model=request.model)
        yield 'data: [DONE]\n\n'
    rig.service.chat_completion_stream = stream
    dispatcher = object.__new__(InterBotDeliveryDispatcher)
    dispatcher.service, dispatcher.store, dispatcher._emit = rig.service, rig.delivery, AsyncMock()
    claimed = rig.delivery.update(first_record.id, status="DISPATCHING", claim_token="claim-test")
    asyncio.run(dispatcher._dispatch(claimed))
    rig.worker._reconcile(first, NOW + timedelta(seconds=10))
    assert len(calls) == 1 and occurrence(rig, first).state == "succeeded"
    second = rig.worker.store.reserve_due(NOW + timedelta(seconds=61))[0]
    second_record = enqueue(rig, second, NOW + timedelta(seconds=61))
    second_session = occurrence(rig, second).session_id
    assert (first_session != second_session) is clear
    assert first_record.id != second_record.id
    with rig.engine.connect() as conn:
        sessions = conn.execute(text("SELECT id,status FROM sessions")).all()
        assert ("personal", "active") in sessions
        assert all(status == "archived" for sid, status in sessions if sid != "personal")
    assert rig.bot.default_model == "first" and rig.bot.endpoint_id == 1


@pytest.mark.parametrize("fault", ["before", "after"])
def test_restart_after_enqueue_fault_is_bounded_and_never_duplicates(rig, fault):
    job, run = reserve(rig, once=True, grace=60)
    prepared = rig.worker._prepare(run, NOW)
    rig.delivery.failure = fault
    with pytest.raises(RuntimeError, match="injected"):
        rig.worker._enqueue(run, prepared, rig.dispatcher, NOW)
    assert occurrence(rig, run).state == "pending"
    # Restart after the deadline. A committed delivery must be recovered FIRST;
    # an attempt that never committed is safely skipped without sending.
    rig.worker = PromptDeliveryWorker(rig.engine, rig.service)
    rig.delivery.failure = None
    prepared = rig.worker._prepare(run, NOW + timedelta(minutes=2))
    record = rig.worker._enqueue(run, prepared, rig.dispatcher, NOW + timedelta(minutes=2))
    if fault == "after":
        assert record and rig.delivery.calls == 1
        rig.worker._reconcile(run, NOW + timedelta(minutes=2))
    assert occurrence(rig, run).state == "skipped"
    with Session(rig.engine) as session:
        assert session.get(PromptSchedule, job).lifecycle == "completed"


@pytest.mark.parametrize("action, expected", [("pause", "cancelled"), ("cancel", "cancelled"), ("expire", "skipped")])
def test_claim_gate_linearizes_lifecycle_deadline_and_removes_reservation(rig, action, expected):
    job, run = reserve(rig, now=NOW - timedelta(minutes=2), grace=3600)
    record = enqueue(rig, run)
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs(id,status) VALUES (:id,'reserved')"), {"id": record.turn_id})
    if action == "pause":
        mutate(rig, job, run, lifecycle="paused")
    elif action == "cancel":
        mutate(rig, job, run, cancel=True)
    else:
        with Session(rig.engine) as session:
            row = session.get(PromptOccurrence, run)
            row.snapshot_json = {**row.snapshot_json, "start_deadline": (NOW - timedelta(seconds=1)).isoformat()}
            session.add(row)
            session.commit()
    with rig.engine.begin() as conn:
        assert load_claim_head(conn, BOT, rig.service.config) is None
    assert occurrence(rig, run).state == expected
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM turn_logs")).scalar_one() == 0


def test_accepted_recovery_ignores_pause_and_deadline_never_blind_replays(rig):
    job, run = reserve(rig)
    record = enqueue(rig, run)
    rig.delivery.update(record.id, transport_accepted_at=NOW.isoformat())
    mutate(rig, job, run, lifecycle="paused", cancel=True)
    with rig.engine.begin() as conn:
        assert load_claim_head(conn, BOT, rig.service.config)["id"] == record.id
    rig.worker._reconcile(run, NOW + timedelta(days=8))
    assert occurrence(rig, run).state == "running"
    rig.delivery.update(record.id, status="FAILED", last_error="outcome is ambiguous and was not replayed")
    rig.worker._reconcile(run, NOW + timedelta(days=8))
    assert occurrence(rig, run).state == "unknown"
    assert rig.worker._prepare(run, NOW) is None
    assert rig.delivery.calls == 1


def test_preclaim_disabled_model_and_thread_fail_closed(rig):
    job, run = reserve(rig)
    record = enqueue(rig, run)
    rig.service.config.ensure_model_catalog = lambda: ModelCatalog()
    with rig.engine.begin() as conn:
        assert load_claim_head(conn, BOT, rig.service.config) is None
    assert rig.delivery.get(record.id).status == "FAILED"
    assert occurrence(rig, run).state == "failed"


def test_valid_claim_reads_bound_model_and_nonactive_thread(rig):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    rig.bot.endpoint_id = 1
    with rig.engine.begin() as conn:
        head = load_claim_head(conn, BOT, rig.service.config)
        assert head["id"] == record.id
        assert json.loads(head["payload_json"])["model"] == "second@test-native"
    assert occurrence(rig, run).state == "queued"


@pytest.mark.parametrize("status", ["active", "deleted"])
def test_changed_thread_cannot_be_claimed(rig, status):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status=:status WHERE id=:id"),
                     {"status": status, "id": occurrence(rig, run).session_id})
        assert load_claim_head(conn, BOT, rig.service.config) is None
    assert rig.delivery.get(record.id).status == "FAILED"


def test_pending_removed_bot_is_terminal_without_enqueue(rig, monkeypatch):
    import llm_bawt.bots as bots
    _, run = reserve(rig)
    monkeypatch.setattr(bots.BotManager, "get_bot", lambda *a: None)
    assert rig.worker._prepare(run, NOW) is None
    assert occurrence(rig, run).state == "failed"
    assert rig.delivery.calls == 0


def test_claim_gate_rolls_back_cancellation_with_outer_transaction(rig):
    job, run = reserve(rig)
    record = enqueue(rig, run)
    mutate(rig, job, run, cancel=True)
    with pytest.raises(RuntimeError):
        with rig.engine.begin() as conn:
            assert load_claim_head(conn, BOT, rig.service.config) is None
            raise RuntimeError("claim transaction interrupted")
    assert rig.delivery.get(record.id).status == "QUEUED"
    assert occurrence(rig, run).state == "queued"


def test_pending_cancel_before_prepare_does_not_create_thread(rig):
    job, run = reserve(rig)
    mutate(rig, job, run, cancel=True)
    assert rig.worker._prepare(run, NOW) is None
    assert occurrence(rig, run).state == "cancelled"
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sessions")).scalar_one() == 1


def test_session_transaction_failure_rolls_back_preparation(rig):
    _, run = reserve(rig)
    def fail(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("UPDATE prompt_occurrences"):
            raise RuntimeError("injected preparation commit")
    event.listen(rig.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            rig.worker._prepare(run, NOW)
    finally:
        event.remove(rig.engine, "before_cursor_execute", fail)
    assert occurrence(rig, run).session_id is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sessions")).scalar_one() == 1


def test_ordinary_callback_does_not_consult_prompt_tables(rig):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    rig.delivery.update(record.id, metadata_json="{}")
    with rig.engine.begin() as conn:
        assert load_claim_head(conn, BOT, None)["id"] == record.id


def test_missing_persisted_receipt_is_unknown_not_recreated(rig):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM inter_bot_deliveries WHERE id=:id"), {"id": record.id})
    rig.worker._reconcile(run, NOW)
    assert occurrence(rig, run).state == "unknown"
    assert rig.worker._prepare(run, NOW) is None


def test_prompt_seed_only_reads_automation_owner_instance(rig):
    from unittest.mock import Mock
    from llm_bawt.service.prompt_execution import build_prompt_seed
    from llm_bawt.models.message import Message

    load = Mock()
    history = SimpleNamespace(load_history=load, build_context_payload=Mock(
        return_value=SimpleNamespace(seed_messages=[])))
    instance = SimpleNamespace(bot_id=BOT, resolved_model_alias="second@test-native",
                               history_manager=history,
                               config_resolver=SimpleNamespace(resolve_config_setting=lambda key:
                                   SimpleNamespace(value="inline+summaries" if key == "history_scope" else 3000)))
    rig.service.config.resolve_context_budget = lambda ref: (1000, 100, 900)
    rig.service._resolve_request_model = Mock(side_effect=AssertionError("Personal resolver"))
    rig.service._get_llm_bawt = Mock(side_effect=AssertionError("Personal owner cache"))
    binding = {"thread_session_id": "automation", "explicit_thread": True}
    assert build_prompt_seed(rig.service, instance, binding) == []
    load.assert_called_once_with(session_id="automation")
    history.build_context_payload.return_value.seed_messages = [Message(role="user", content="Only automation")]
    assert build_prompt_seed(rig.service, instance, binding)[0]["content"] == "Only automation"
    load.reset_mock()
    assert build_prompt_seed(rig.service, instance, {**binding, "thread_resume_id": "sdk-automation"}) is None
    load.assert_not_called()


def test_late_terminal_accepted_turn_cannot_be_reserved_again(rig):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    rig.delivery.update(record.id, transport_accepted_at=NOW.isoformat())
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs(id,status,ended_at) VALUES (:id,'ok',:ended)"),
                     {"id": record.turn_id, "ended": NOW.isoformat()})
        assert load_claim_head(conn, BOT, rig.service.config) is None
    assert rig.delivery.get(record.id).status == "DELIVERED"


def test_real_stream_coordinator_selects_isolated_prompt_instance(rig, monkeypatch):
    from unittest.mock import Mock
    from llm_bawt.service.background_service import BackgroundService
    from llm_bawt.service.schemas import ChatCompletionRequest
    import llm_bawt.service.chat_streaming as streaming
    from llm_bawt.service import prompt_execution, stream_execution_selection

    _, run = reserve(rig)
    record = enqueue(rig, run)
    payload = rig.delivery.payload(record.id)
    # The real dispatcher injects the claim token after claiming the row;
    # model that trusted request instead of presenting an unfenced payload.
    rig.delivery.update(record.id, status="DISPATCHING", claim_token="claim-test")
    payload["inter_bot_claim_token"] = "claim-test"
    service = BackgroundService.__new__(BackgroundService)
    service.config = rig.service.config
    service._inter_bot_dispatcher = rig.dispatcher
    service._turn_log_store = rig.service._turn_log_store
    service._turn_log_store.engine = None
    service._turn_log_store.accept_session_binding = Mock()
    service._resolve_request_model = Mock(side_effect=AssertionError("Personal model resolver used"))
    service._get_llm_bawt = Mock(side_effect=AssertionError("Personal cached instance used"))
    service._persist_turn_log = Mock()
    service._maybe_summarize_on_new = Mock()
    service._maybe_rotate_agent_session = Mock()
    service._redis_subscriber = None
    rig.delivery.validate_claim = lambda **kwargs: True
    isolated = SimpleNamespace(
        bot=rig.bot, bot_id=BOT, user_id=OWNER,
        client=SimpleNamespace(model_definition={"type": "claude-code"}, _bot_config={}),
        history_manager=SimpleNamespace(_db_backend=SimpleNamespace(get_session=lambda sid: {"session_metadata": {}})),
    )
    monkeypatch.setattr(stream_execution_selection, "create_prompt_instance", lambda svc, req: (req.model, isolated))
    monkeypatch.setattr(prompt_execution, "build_prompt_seed", lambda *a: [])
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", lambda *a, **kw: None)
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_task_turn_context", lambda **kw: "test-context")
    monkeypatch.setattr(streaming, "ToolEventCoordinator", lambda engine: None)
    monkeypatch.setattr(streaming, "prepare_stream_request_attachments", AsyncMock(
        return_value=("Literal saved prompt", None, None, None)))
    contexts, events = [], []
    class Worker:
        def __init__(self, context):
            self.context = context
            contexts.append(context)
        async def _publish_unified(self, event):
            events.append(event)
        def _stream_to_queue(self):
            self.context.loop.call_soon_threadsafe(self.context.chunk_queue.put_nowait, "done.")
            self.context.loop.call_soon_threadsafe(self.context.chunk_queue.put_nowait, None)
    monkeypatch.setattr(streaming, "TurnStreamWorker", Worker)
    class Presentations:
        def __init__(self, engine):
            pass
        def resolve_many_safe(self, authors):
            return {(a.entity_type, a.entity_id): a.to_dict() for a in authors}
    monkeypatch.setattr("llm_bawt.entity_presentation.EntityPresentationResolver", Presentations)
    async def collect():
        return [chunk async for chunk in service.chat_completion_stream(ChatCompletionRequest(**payload))]
    chunks = asyncio.run(collect())
    assert any("done." in chunk for chunk in chunks)
    assert contexts[0].llm_bawt is isolated
    assert contexts[0].model_alias == "second@test-native"
    assert contexts[0].thread_binding["thread_session_id"] == payload["session_id"]
    assert contexts[0].thread_binding["explicit_thread"] is True
    service._turn_log_store.accept_session_binding.assert_called_once()
    binding = service._turn_log_store.accept_session_binding.call_args.kwargs
    assert binding["bot_id"] == BOT and binding["user_id"] == OWNER
    assert binding["session_id"] == payload["session_id"]
    assert binding["admission_claim"] is not None
    assert events[0]["session_id"] == payload["session_id"]
    assert events[0]["author"]["entity_type"] == "user"
    assert events[0]["author"]["entity_id"] == OWNER


def test_bare_delivery_success_is_not_inference_success(rig):
    _, run = reserve(rig)
    record = enqueue(rig, run)
    record = replace(record, status="DELIVERED", transport_accepted_at=NOW)
    assert rig.worker._delivery_state(record)[0] == "unknown"


def test_prompt_instance_uses_override_without_personal_model_or_client_mutation(rig, monkeypatch):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.service.prompt_execution import PromptLLMBawt, create_prompt_instance
    from llm_bawt.service.schemas import ChatCompletionRequest
    from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin
    import llm_bawt.core.initialization as initialization
    import copy

    config = rig.service.config
    config.resolve_model = lambda ref, harness=None, default=None: rig.catalog.resolve(ref, harness=harness)
    config.get_tool_format = lambda **kw: "none"
    config.model_copy = lambda **kw: copy.copy(config)
    monkeypatch.setattr(initialization, "get_model_lifecycle", lambda config: pytest.fail("Touched personal lifecycle"))
    monkeypatch.setattr(initialization, "RuntimeSettingsResolver", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(initialization, "ConfigResolver", lambda **kw: SimpleNamespace())
    monkeypatch.setattr("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", lambda *a, **k: 300)
    def client(self):
        value = AgentBackendClient.__new__(AgentBackendClient)
        value.model_definition = self.model_definition
        value._bot_config = {}
        return value
    monkeypatch.setattr(PromptLLMBawt, "_initialize_client", client)
    for name in ("_init_memory", "_init_search", "_init_home_assistant", "_init_newsapi", "_init_web_fetch", "_init_system_prompt"):
        monkeypatch.setattr(PromptLLMBawt, name, lambda *a: None)
    def history(self):
        self.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(get_session=lambda sid: {
            "session_metadata": {"agent_session_keys": {"claude_code": "sdk-automation"}}
        }))
    monkeypatch.setattr(PromptLLMBawt, "_init_history", history)
    rig.service._session_model_overrides = {(BOT, OWNER): "personal-model"}
    request = ChatCompletionRequest(bot_id=BOT, user=OWNER, model="second@test-native",
                                    session_id="automation", messages=[{"role": "user", "content": "Test"}])
    model, instance = create_prompt_instance(rig.service, request)
    assert model == "second@test-native"
    assert instance.client._bot_config["model"] == "second"
    assert instance.client._bot_config["endpoint_id"] == 2
    assert instance.model_lifecycle is None
    assert instance.config is not config
    assert rig.service._session_model_overrides == {(BOT, OWNER): "personal-model"}
    assert rig.bot.default_model == "first" and rig.bot.endpoint_id == 1
    binding = ChatStreamingBridgeMixin()._bind_agent_thread(instance, request)
    assert binding["explicit_thread"] and binding["thread_session_id"] == "automation"
    assert binding["thread_resume_id"] == "sdk-automation"
    assert "thread_session_id" not in instance.client._bot_config


def test_automation_history_excludes_global_summaries_and_fails_closed():
    from unittest.mock import Mock
    from llm_bawt.models.message import Message
    from llm_bawt.service.prompt_execution import PromptHistoryManager
    history = PromptHistoryManager.__new__(PromptHistoryManager)
    loader = Mock(return_value=[
        Message(role="summary", content="Personal secret", session_id="personal"),
        Message(role="user", content="Automation only", session_id="automation"),
        Message(role="summary", content="Own summary", session_id="automation"),
    ])
    history._db_backend = SimpleNamespace(load_session_scoped=loader)
    history.load_history()
    loader.assert_not_called()
    assert history.messages == []
    history.load_history(session_id="automation")
    assert [message.content for message in history.messages] == ["Automation only", "Own summary"]
    loader.side_effect = RuntimeError("scoped read failed")
    with pytest.raises(RuntimeError, match="scoped read"):
        history.load_history(session_id="automation")
    assert history.messages == []


def test_inactive_session_identity_cannot_take_over_another_owner(rig):
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
    with rig.engine.begin() as conn:
        with pytest.raises(ValueError, match="owned thread"):
            PostgreSQLShortTermManager.create_inactive_session_row(
                conn, bot_id=BOT, user_id="another-owner", session_id="personal")
