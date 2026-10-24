"""TASK-1202: offline private-PostgreSQL routing and consent fences."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import event, text

from llm_bawt.bots import Bot, BotManager
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.message_routing import MessageRouter, RouteConflict, RouteRequest
from llm_bawt.bot_tags import TagSnapshot
from llm_bawt.service import message_routing
from tests.bot_tag_fixtures import PRIVATE_MODES, clear_tags, ensure_tag_tables, make_private, tag
from tests.pg_isolation import isolated_pg_store

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(monkeypatch):
    store = isolated_pg_store(InterBotDeliveryStore)
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bot_profiles (
                slug VARCHAR(128) PRIMARY KEY, last_session_id VARCHAR(36)
            )
        """))
        ensure_tag_tables(conn)
    owner = "route-owner-" + uuid.uuid4().hex
    slugs = ["test_route_a_" + uuid.uuid4().hex, "test_route_b_" + uuid.uuid4().hex]
    bots = [Bot(slug=slug, name=slug, description="test", system_prompt="test",
                bot_type="agent", agent_backend="claude-code", harness="claude-code",
                endpoint_id=1, default_model="route-test") for slug in slugs]
    with store.engine.begin() as conn:
        for slug in slugs:
            conn.execute(text("INSERT INTO bot_profiles(slug) VALUES (:slug) ON CONFLICT DO NOTHING"),
                         {"slug": slug})
    catalog = ModelCatalog([ModelEndpoint(
        1, ModelIdentity(1, "route-test", "anthropic", "Test"),
        AccessPath(1, "test-native", "anthropic", "anthropic-messages", None, "test"),
        "route-test",
    )])
    monkeypatch.setattr(BotManager, "list_bots", lambda self: bots)
    monkeypatch.setattr(BotManager, "get_bot", lambda self, slug: next(
        (bot for bot in bots if bot.slug == slug), None,
    ))
    monkeypatch.setattr(store.config.__class__, "ensure_model_catalog", lambda self: catalog)
    dispatcher = SimpleNamespace(store=store, announce_admitted=Mock())
    service = SimpleNamespace(config=store.config, _inter_bot_dispatcher=dispatcher)
    router = MessageRouter(service)
    ctx = SimpleNamespace(router=router, bots=bots, slugs=slugs, owner=owner,
                          store=store, dispatcher=dispatcher, tagged=list(slugs))
    yield ctx
    with store.engine.begin() as conn:
        clear_tags(conn, *ctx.tagged)
    store.engine.dispose()


def test_opt_in_preview_oldest_and_stable_retry(rig):
    request = RouteRequest(user_id=rig.owner)
    assert rig.router.preview(request)["state"] == "not_routed"
    assert all(not row["enabled"] for row in rig.router.candidates(request)["candidates"])
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    preview = rig.router.preview(request)
    assert preview["selected_bot_id"] == min(rig.slugs)
    assert preview["candidates"][0]["last_message_at"] is None
    receipt = rig.router.route(request, message="test one", idempotency_key="same-key")
    assert receipt["selected_bot_id"] == min(rig.slugs)
    assert receipt["model_endpoint_id"] == 1
    assert receipt["ranking"]["version"] == 1
    assert receipt["ranking"]["selected_last_message_at"] is None
    assert receipt["ranking"]["compatible_candidate_ids"] == sorted(rig.slugs)
    assert rig.store.payload(receipt["delivery_id"])["prefer_steer"] is False
    assert rig.router.preview(request)["selected_bot_id"] == max(rig.slugs)
    retry = rig.router.route(request, message="test one", idempotency_key="same-key")
    assert retry["delivery_id"] == receipt["delivery_id"] and retry["duplicate"]
    with pytest.raises(RouteConflict):
        rig.router.route(request, message="changed", idempotency_key="same-key")
    with rig.store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM message_routes WHERE owner_user_id=:owner"),
                            {"owner": rig.owner}).scalar_one() == 1


def test_revocation_prevents_unaccepted_claim_and_accepted_work_is_not_retargeted(rig):
    selected = min(rig.slugs)
    rig.router.set_enabled(rig.owner, selected, True)
    receipt = rig.router.route(RouteRequest(user_id=rig.owner), message="consent", idempotency_key="a")
    rig.router.set_enabled(rig.owner, selected, False)
    assert rig.store.claim_next(selected, claim_owner="test", steer_capable=True) is None
    assert rig.store.get(receipt["delivery_id"]).status == "CANCELLED"
    assert "revoked" in rig.store.get(receipt["delivery_id"]).last_error
    assert rig.router.route(RouteRequest(user_id=rig.owner), message="consent", idempotency_key="a")["selected_bot_id"] == selected


def _excluded_bot(rig, kind: str, mode: str | None = None) -> str:
    """A fixture bot listed by BotManager and excluded by its tags."""
    slug = f"test_route_{kind}_{uuid.uuid4().hex}"
    rig.tagged.append(slug)
    bot = Bot(slug=slug, name=slug, description="test", system_prompt="test",
              bot_type="agent", agent_backend="claude-code", harness="claude-code",
              endpoint_id=1, default_model="route-test")
    rig.bots.insert(0, bot)
    with rig.store.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_profiles(slug) VALUES (:slug)"), {"slug": slug})
        if kind == "private":
            make_private(conn, slug, mode)
        else:
            tag(conn, slug, kind)
    return slug


def _statements(rig):
    seen: list[tuple[str, object]] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        seen.append((statement, parameters))

    event.listen(rig.store.engine, "before_cursor_execute", spy)
    return seen, lambda: event.remove(rig.store.engine, "before_cursor_execute", spy)


@pytest.mark.parametrize("kind,mode", [*(("private", m) for m in PRIVATE_MODES), ("inactive", None)])
def test_private_and_inactive_bots_are_listed_never_configured_and_never_queried(
        rig, kind, mode):
    excluded = _excluded_bot(rig, kind, mode)
    seen, stop = _statements(rig)
    try:
        all_rows = rig.router.candidates(RouteRequest(user_id=rig.owner))["candidates"]
    finally:
        stop()
    row = next(row for row in all_rows if row["bot_id"] == excluded)
    assert not row["configurable"] and not row["enabled"]
    assert row["exclusion_reasons"] == [f"{kind} bot", "not opted in"]
    # The snapshot is read before the eligibility read, and no statement after
    # it names the excluded bot.
    texts = [statement for statement, _ in seen]
    snapshot_at = next(i for i, t in enumerate(texts) if "bot_tag_catalog" in t)
    eligibility_at = next(i for i, t in enumerate(texts) if "message_route_eligibility" in t)
    assert snapshot_at < eligibility_at
    assert not any(excluded in repr(parameters) for _, parameters in seen[snapshot_at + 1:])
    with pytest.raises(ValueError, match="cannot be configured"):
        rig.router.set_enabled(rig.owner, excluded, True)
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    assert rig.router.preview(RouteRequest(user_id=rig.owner))["selected_bot_id"] in rig.slugs
    assert rig.router.preview(RouteRequest(user_id=rig.owner, candidate_bot_ids=[]))["state"] == "not_routed"


def test_an_unreadable_tag_snapshot_yields_no_candidates(rig, monkeypatch):
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)

    class _Down:
        def __init__(self, engine=None):
            pass

        def snapshot(self, conn=None):
            return TagSnapshot.unavailable()

    monkeypatch.setattr(message_routing, "BotTags", _Down)
    rows = rig.router.candidates(RouteRequest(user_id=rig.owner))["candidates"]
    assert rows and all(not row["configurable"] and not row["enabled"] for row in rows)
    assert {row["exclusion_reasons"][0] for row in rows} == {"private bot"}
    assert rig.router.preview(RouteRequest(user_id=rig.owner))["state"] == "not_routed"
    with pytest.raises(ValueError, match="cannot be configured"):
        rig.router.set_enabled(rig.owner, rig.slugs[0], True)


def test_rank_oldest_original_message_and_ignore_other_owner_and_summaries(rig):
    from llm_bawt.memory.postgresql import ensure_bot_partitions

    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    older, newer = sorted(rig.slugs)
    with rig.store.engine.begin() as conn:
        for slug in rig.slugs:
            ensure_bot_partitions(conn, slug)
            session_id = str(uuid.uuid4())
            conn.execute(text("""
                INSERT INTO sessions(id,bot_id,user_id,status,started_at)
                VALUES (:id,:bot,:owner,'archived',now())
            """), {"id": session_id, "bot": slug, "owner": rig.owner})
            conn.execute(text("""
                INSERT INTO messages(bot_id,id,role,content,timestamp,session_id)
                VALUES (:bot,:id,'user','test only',extract(epoch from now() - (:days * interval '1 day')),:session)
            """), {"bot": slug, "id": str(uuid.uuid4()), "days": 20 if slug == older else 2,
                    "session": session_id})
            if slug == older:
                conn.execute(text("""
                    INSERT INTO messages(bot_id,id,role,content,timestamp,session_id)
                    VALUES (:bot,:id,'summary','ignored',extract(epoch from now()),:session)
                """), {"bot": slug, "id": str(uuid.uuid4()), "session": session_id})
                other = str(uuid.uuid4())
                conn.execute(text("""
                    INSERT INTO sessions(id,bot_id,user_id,status,started_at)
                    VALUES (:id,:bot,'other-owner','archived',now())
                """), {"id": other, "bot": slug})
                conn.execute(text("""
                    INSERT INTO messages(bot_id,id,role,content,timestamp,session_id)
                    VALUES (:bot,:id,'assistant','different owner',extract(epoch from now()),:session)
                """), {"bot": slug, "id": str(uuid.uuid4()), "session": other})
    preview = rig.router.preview(RouteRequest(user_id=rig.owner))
    assert preview["selected_bot_id"] == older
    assert preview["candidates"][0]["last_message_at"] is not None
    rig.router.route(RouteRequest(user_id=rig.owner), message="occupy oldest", idempotency_key="busy")
    assert rig.router.preview(RouteRequest(user_id=rig.owner))["selected_bot_id"] == newer


def test_all_busy_fallback_keeps_oldest_and_never_steers(rig):
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    for slug in rig.slugs:
        rig.router.route(RouteRequest(user_id=rig.owner, candidate_bot_ids=[slug]),
                         message="occupy", idempotency_key=slug)
    request = RouteRequest(user_id=rig.owner)
    preview = rig.router.preview(request)
    assert preview["selected_bot_id"] == min(rig.slugs)
    assert "all compatible bots busy" in preview["reason"]
    route = rig.router.route(request, message="queued", idempotency_key="fallback")
    assert route["selected_bot_id"] == min(rig.slugs)
    assert rig.store.payload(route["delivery_id"])["prefer_steer"] is False


def test_concurrent_same_key_binds_one_target_and_one_delivery(rig):
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    start = threading.Barrier(2)
    results = []

    def route():
        start.wait()
        results.append(rig.router.route(RouteRequest(user_id=rig.owner),
                                        message="one action", idempotency_key="race"))

    threads = [threading.Thread(target=route) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert len(results) == 2 and results[0]["delivery_id"] == results[1]["delivery_id"]
    assert results[0]["selected_bot_id"] == results[1]["selected_bot_id"]


def test_concurrent_independent_intents_choose_distinct_idle_targets(rig):
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    start = threading.Barrier(2)
    routes = []
    failures = []

    def send(index):
        try:
            start.wait(timeout=5)
            routes.append(rig.router.route(RouteRequest(user_id=rig.owner),
                                           message=f"intent {index}", idempotency_key=f"distinct-{index}"))
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=send, args=(i,)) for i in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not failures
    assert {route["selected_bot_id"] for route in routes} == set(rig.slugs)


def test_scheduled_preparation_and_direct_route_share_owner_pool_lock(rig):
    from sqlmodel import Session
    from llm_bawt.service.prompt_delivery import PromptDeliveryWorker
    from llm_bawt.service.prompt_schedule_models import PromptSchedule
    from llm_bawt.service.prompt_schedule_store import PromptScheduleStore
    from llm_bawt.service.prompt_timing import PromptTiming
    from llm_bawt.service.scheduler import ScheduledJob, create_scheduler_tables

    third = "test_route_c_" + uuid.uuid4().hex
    rig.bots.append(Bot(slug=third, name=third, description="test", system_prompt="test",
                        bot_type="agent", agent_backend="claude-code", harness="claude-code",
                        endpoint_id=1, default_model="route-test"))
    rig.slugs.append(third)
    with rig.store.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_profiles(slug) VALUES (:slug)"), {"slug": third})
    create_scheduler_tables(rig.store.engine)
    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    now = datetime.now(timezone.utc)
    timing = PromptTiming(kind="interval", interval_seconds=60,
                          anchor_at=now + timedelta(seconds=1))
    schedules = PromptScheduleStore(rig.store.engine)
    run_ids = []
    for index in (1, 2):
        job = schedules.create(owner=rig.owner, bot_id=rig.slugs[0], name=f"Test {index}",
                               prompt=f"automated {index}", timing=timing, now=now)
        with Session(rig.store.engine) as session, session.begin():
            session.get(ScheduledJob, job).bot_id = "*"
            session.get(PromptSchedule, job).target_mode = "auto"
    run_ids = schedules.reserve_due(now + timedelta(seconds=2))
    assert len(run_ids) == 2
    worker = PromptDeliveryWorker(rig.store.engine, rig.router.service)
    barrier = threading.Barrier(3)
    targets, errors = [], []

    def prepare(run_id):
        try:
            barrier.wait(timeout=5)
            result = worker._prepare(run_id, now + timedelta(seconds=2))
            targets.append(result[0]["bot_id"])
        except Exception as exc:
            errors.append(exc)

    def route_direct():
        try:
            barrier.wait(timeout=5)
            targets.append(rig.router.route(RouteRequest(user_id=rig.owner),
                                            message="direct", idempotency_key="alongside-auto")["selected_bot_id"])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=prepare, args=(run_id,)) for run_id in run_ids]
    threads.append(threading.Thread(target=route_direct))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert not errors
    assert set(targets) == set(rig.slugs)
    with Session(rig.store.engine) as session:
        from llm_bawt.service.prompt_schedule_models import PromptOccurrence
        prepared_targets = [session.get(PromptOccurrence, run_id).snapshot_json["bot_id"] for run_id in run_ids]
    assert len(set(prepared_targets)) == 2
    assert [worker._prepare(run_id, now + timedelta(seconds=2))[0]["bot_id"]
            for run_id in run_ids] == prepared_targets


def test_actor_scoped_receipts_and_truthful_authorship(rig):
    from llm_bawt.message_authorship import AuthorReference

    selected = rig.slugs[0]
    rig.router.set_enabled(rig.owner, selected, True)
    request = RouteRequest(user_id=rig.owner, candidate_bot_ids=[selected])
    first = rig.router.route(request, message="same", idempotency_key="shared", actor=AuthorReference.bot("bot-one"))
    second = rig.router.route(request, message="same", idempotency_key="shared", actor=AuthorReference.bot("bot-two"))
    third = rig.router.route(request, message="same", idempotency_key="shared")
    assert len({first["delivery_id"], second["delivery_id"], third["delivery_id"]}) == 3
    assert rig.store.get(first["delivery_id"]).author == AuthorReference.bot("bot-one")
    assert rig.store.get(second["delivery_id"]).author == AuthorReference.bot("bot-two")
    assert rig.store.get(third["delivery_id"]).author == AuthorReference.user(rig.owner)
    assert rig.router.route(request, message="same", idempotency_key="shared",
                            actor=AuthorReference.bot("bot-one"))["delivery_id"] == first["delivery_id"]


def test_continue_snapshots_owner_thread_and_rejects_pointer_changes(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    route = rig.router.route(RouteRequest(user_id=rig.owner), message="bound", idempotency_key="thread")
    payload = rig.store.payload(route["delivery_id"])
    with rig.store.engine.begin() as conn:
        prior = conn.execute(text("SELECT user_id, status FROM sessions WHERE id=:id"),
                             {"id": payload["session_id"]}).mappings().one()
        assert prior["user_id"] == rig.owner and prior["status"] == "active"
        conn.execute(text("UPDATE sessions SET status='archived', ended_at=now() WHERE id=:id"),
                     {"id": payload["session_id"]})
        conn.execute(text("""
            INSERT INTO sessions(id, bot_id, user_id, status, started_at)
            VALUES (:id,:bot,:owner,'active',now())
        """), {"id": str(uuid.uuid4()), "bot": target, "owner": rig.owner})
    assert rig.store.claim_next(target, claim_owner="test", steer_capable=True) is None
    assert rig.store.get(route["delivery_id"]).status == "CANCELLED"
    assert "thread changed" in rig.store.get(route["delivery_id"]).last_error


@pytest.mark.parametrize("policy", ["reset_retain_history", "reset_without_history"])
def test_routed_reset_uses_shared_rotation_and_one_successor(rig, policy):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    prior = rig.router.route(RouteRequest(user_id=rig.owner), message="prior", idempotency_key="prior")
    with rig.store.engine.connect() as conn:
        previous_session = conn.execute(text("""
            SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:owner AND status='active'
        """), {"bot": target, "owner": rig.owner}).scalar_one()
    assert rig.store.cancel(prior["delivery_id"]).status == "CANCELLED"
    request = RouteRequest(user_id=rig.owner, session_policy=policy)
    receipt = rig.router.route(request, message="after reset", idempotency_key="reset")
    claim = rig.store.claim_next(target, claim_owner="test", steer_capable=True)
    assert claim is not None and claim.id == receipt["delivery_id"]
    assert claim.session_policy == policy
    assert claim.old_session_id == previous_session
    assert claim.new_session_id != previous_session
    assert rig.store.payload(claim.id)["session_id"] == claim.new_session_id
    with rig.store.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM sessions WHERE bot_id=:bot AND user_id=:owner AND status='active'
        """), {"bot": target, "owner": rig.owner}).scalar_one() == 1
    retry = rig.router.route(request, message="after reset", idempotency_key="reset")
    assert retry["delivery_id"] == claim.id and retry["selected_bot_id"] == target


def test_continue_rejects_persisted_provider_mismatch_and_reset_does_not_continue(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    with rig.store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions(id, bot_id, user_id, status, started_at, session_metadata)
            VALUES (:id,:bot,:owner,'active',now(),CAST(:meta AS jsonb))
        """), {"id": str(uuid.uuid4()), "bot": target, "owner": rig.owner,
               "meta": '{"agent_session_keys":{"claude_code":"sdk-1"},"agent_session_key_models":{"claude_code":"unknown-provider/model"}}'})
    preview = rig.router.preview(RouteRequest(user_id=rig.owner))
    assert preview["selected_bot_id"] is None
    assert "provider binding" in preview["candidates"][0]["exclusion_reasons"][0]
    reset = rig.router.route(RouteRequest(user_id=rig.owner, session_policy="reset_without_history"),
                             message="fresh", idempotency_key="reset")
    assert reset["selected_bot_id"] == target
    assert "session_id" not in rig.store.payload(reset["delivery_id"])


def test_continue_adopts_the_external_first_session_writer(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    winner = str(uuid.uuid4())
    inserted = False

    def race(_conn, _cursor, statement, _params, _context, _many):
        nonlocal inserted
        if inserted or not statement.lstrip().startswith("INSERT INTO sessions"):
            return
        inserted = True
        with rig.store.engine.begin() as rival:
            rival.execute(text("""
                INSERT INTO sessions(id, bot_id, user_id, started_at, status)
                VALUES (:id,:bot,:owner,now(),'active')
            """), {"id": winner, "bot": target, "owner": rig.owner})

    event.listen(rig.store.engine, "before_cursor_execute", race)
    try:
        routed = rig.router.route(RouteRequest(user_id=rig.owner), message="first", idempotency_key="race-first")
    finally:
        event.remove(rig.store.engine, "before_cursor_execute", race)
    assert inserted
    assert rig.store.payload(routed["delivery_id"])["session_id"] == winner


def test_continue_rejects_changed_provider_binding_even_when_sdk_key_stays_same(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    with rig.store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions(id, bot_id, user_id, status, started_at, session_metadata)
            VALUES (:id,:bot,:owner,'active',now(),CAST(:meta AS jsonb))
        """), {"id": str(uuid.uuid4()), "bot": target, "owner": rig.owner,
               "meta": '{"agent_session_keys":{"claude_code":"sdk-1"},"agent_session_key_models":{"claude_code":"route-test"}}'})
    route = rig.router.route(RouteRequest(user_id=rig.owner), message="bound", idempotency_key="provider-change")
    with rig.store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(session_metadata,
                '{agent_session_key_models,claude_code}', '"route-test@test-native"'::jsonb)
            WHERE bot_id=:bot AND user_id=:owner
        """), {"bot": target, "owner": rig.owner})
    assert rig.store.claim_next(target, claim_owner="test", steer_capable=True) is None
    assert rig.store.get(route["delivery_id"]).status == "CANCELLED"
    assert "provider session changed" in rig.store.get(route["delivery_id"]).last_error


def test_revocation_after_claim_prevents_transport_acceptance(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    receipt = rig.router.route(RouteRequest(user_id=rig.owner), message="late revocation", idempotency_key="late")
    claim = rig.store.claim_next(target, claim_owner="test", steer_capable=True)
    assert claim and claim.status == "DISPATCHING"
    rig.router.set_enabled(rig.owner, target, False)
    assert rig.store.mark_transport_accepted(claim.id, claim.claim_token) is False
    assert rig.store.get(claim.id).status == "CANCELLED"
    with rig.store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM turn_logs WHERE id=:id"),
                            {"id": claim.turn_id}).scalar_one() == 0


def test_recalled_and_injected_messages_do_not_affect_activity(rig):
    from llm_bawt.memory.postgresql import ensure_bot_partitions

    for slug in rig.slugs:
        rig.router.set_enabled(rig.owner, slug, True)
    older, newer = sorted(rig.slugs)
    with rig.store.engine.begin() as conn:
        for slug in rig.slugs:
            ensure_bot_partitions(conn, slug)
            sid = str(uuid.uuid4())
            conn.execute(text("""
                INSERT INTO sessions(id,bot_id,user_id,status,started_at)
                VALUES (:id,:bot,:owner,'archived',now())
            """), {"id": sid, "bot": slug, "owner": rig.owner})
            conn.execute(text("""
                INSERT INTO messages(bot_id,id,role,content,timestamp,session_id,author_entity_type)
                VALUES (:bot,:id,'user','real',:stamp,:sid,'user')
            """), {"bot": slug, "id": str(uuid.uuid4()), "sid": sid,
                   "stamp": datetime(2026, 1, 1 if slug == older else 2, tzinfo=timezone.utc).timestamp()})
            if slug == older:
                conn.execute(text("""
                    INSERT INTO messages(bot_id,id,role,content,timestamp,session_id,author_entity_type,recalled_history)
                    VALUES (:bot,:id,'user','recalled',extract(epoch from now()),:sid,'user',true),
                           (:bot,:id2,'user','injected',extract(epoch from now()),:sid,'system',false)
                """), {"bot": slug, "id": str(uuid.uuid4()), "id2": str(uuid.uuid4()), "sid": sid})
    assert rig.router.preview(RouteRequest(user_id=rig.owner))["selected_bot_id"] == older


def test_user_cancel_preserves_accepted_ordinary_retry_and_error_history(rig):
    from llm_bawt.message_authorship import AuthorReference

    target = rig.slugs[0]
    ordinary, _ = rig.store.enqueue(
        sender_bot_id="ordinary-test", target_bot_id=target,
        author=AuthorReference.user(rig.owner), message="ordinary",
        idempotency_key="ordinary-accepted", session_policy="continue",
        payload={"bot_id": target, "user": rig.owner,
                 "messages": [{"role": "user", "content": "ordinary"}]},
    )
    with rig.store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET transport_accepted_at=now(),
                last_error='prior retry error' WHERE id=:id
        """), {"id": ordinary.id})
    cancelled = rig.store.cancel(ordinary.id)
    assert cancelled.status == "CANCELLED"
    assert cancelled.last_error == "prior retry error"

    rig.router.set_enabled(rig.owner, target, True)
    routed = rig.router.route(RouteRequest(user_id=rig.owner), message="routed", idempotency_key="accepted")
    with rig.store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET transport_accepted_at=now() WHERE id=:id
        """), {"id": routed["delivery_id"]})
    assert rig.store.cancel(routed["delivery_id"]) is None
    assert rig.store.get(routed["delivery_id"]).status == "QUEUED"


def test_revoked_requeued_head_does_not_block_next_delivery(rig):
    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    from llm_bawt.message_authorship import AuthorReference

    request = RouteRequest(user_id=rig.owner, candidate_bot_ids=[target])
    first = rig.router.route(request, message="first", idempotency_key="first")
    second, _ = rig.store.enqueue(
        sender_bot_id="ordinary-test", target_bot_id=target,
        author=AuthorReference.user(rig.owner), message="second",
        idempotency_key="ordinary-second", session_policy="continue",
        payload={"bot_id": target, "user": rig.owner,
                 "messages": [{"role": "user", "content": "second"}],
                 "prefer_steer": False, "stream": True},
    )
    claim = rig.store.claim_next(target, claim_owner="test", steer_capable=True)
    assert claim.id == first["delivery_id"]
    requeued = rig.store.requeue(claim.id, claim.claim_token, "transient", delay_seconds=0)
    assert requeued.status == "QUEUED"
    with rig.store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM turn_logs WHERE id=:id"),
                            {"id": requeued.turn_id}).scalar_one() == 0
    rig.router.set_enabled(rig.owner, target, False)
    assert rig.store.claim_next(target, claim_owner="test", steer_capable=True) is None
    assert rig.store.get(first["delivery_id"]).status == "CANCELLED"
    claim = rig.store.claim_next(target, claim_owner="test", steer_capable=True)
    assert claim is not None and claim.id == second.id


def test_invalid_activity_timestamp_excludes_candidate_instead_of_ranking_as_never_messaged(rig):
    from llm_bawt.memory.postgresql import ensure_bot_partitions

    target = rig.slugs[0]
    rig.router.set_enabled(rig.owner, target, True)
    with rig.store.engine.begin() as conn:
        ensure_bot_partitions(conn, target)
        sid = str(uuid.uuid4())
        conn.execute(text("""
            INSERT INTO sessions(id,bot_id,user_id,status,started_at)
            VALUES (:id,:bot,:owner,'archived',now())
        """), {"id": sid, "bot": target, "owner": rig.owner})
        conn.execute(text("""
            INSERT INTO messages(bot_id,id,role,content,timestamp,session_id)
            VALUES (:bot,:id,'user','invalid timestamp','Infinity'::float8,:sid)
        """), {"id": str(uuid.uuid4()), "bot": target, "sid": sid})
    preview = rig.router.preview(RouteRequest(user_id=rig.owner))
    assert preview["selected_bot_id"] is None
    assert "activity or admission state unavailable" in preview["candidates"][0]["exclusion_reasons"]


def test_provider_resume_fence_uses_persisted_bridge_model_not_pinned_bot_default(rig):
    from llm_bawt.service.route_session_binding import check_provider_binding

    catalog = ModelCatalog([
        ModelEndpoint(1, ModelIdentity(1, "route-test", "anthropic", "Test"),
                      AccessPath(1, "test-native", "anthropic", "anthropic-messages", None, "test"),
                      "route-test"),
        ModelEndpoint(2, ModelIdentity(2, "other-test", "anthropic", "Other"),
                      AccessPath(2, "other-native", "anthropic", "anthropic-messages", None, "test"),
                      "other-test"),
    ])
    config = SimpleNamespace(ensure_model_catalog=lambda: catalog)
    meta = {"agent_session_keys": {"claude_code": "sdk-1"},
            "agent_session_key_models": {"claude_code": "route-test"}}
    assert check_provider_binding(config, rig.bots[0], catalog.resolve_endpoint(1), meta) == "sdk-1"
    with pytest.raises(ValueError, match="Continuing across model providers"):
        check_provider_binding(config, rig.bots[0], catalog.resolve_endpoint(2), meta)
    meta["agent_session_key_models"] = {}
    with pytest.raises(ValueError, match="model is unknown"):
        check_provider_binding(config, rig.bots[0], catalog.resolve_endpoint(1), meta)


def test_codex_continuation_rejects_changed_stored_model():
    from llm_bawt.service.route_session_binding import check_provider_binding

    path = AccessPath(3, "openai-oauth", "openai", "responses", None, "oauth")
    catalog = ModelCatalog([
        ModelEndpoint(3, ModelIdentity(3, "gpt-one", "openai", "One"), path, "gpt-one"),
        ModelEndpoint(4, ModelIdentity(4, "gpt-two", "openai", "Two"), path, "gpt-two"),
    ])
    config = SimpleNamespace(ensure_model_catalog=lambda: catalog)
    bot = SimpleNamespace(agent_backend="codex", harness="codex")
    meta = {"agent_session_keys": {"codex": "sdk-1"},
            "agent_session_key_models": {"codex": "gpt-one"}}
    assert check_provider_binding(config, bot, catalog.resolve_endpoint(3), meta) == "sdk-1"
    with pytest.raises(ValueError, match="Codex cannot resume"):
        check_provider_binding(config, bot, catalog.resolve_endpoint(4), meta)


def test_signed_route_cannot_impersonate_another_owner(monkeypatch):
    from fastapi import HTTPException
    from llm_bawt.service.routes.message_routing import RouteMessage, route
    from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER
    import llm_bawt.task_turn_context as task_context

    monkeypatch.setattr(task_context, "open_task_turn_context", lambda _: SimpleNamespace(
        user_id="owner-a", bot_id="test-bot", turn_id="turn-test",
    ))
    body = RouteMessage(user_id="owner-b", message="test", idempotency_key="key")
    request = SimpleNamespace(headers={TASK_TURN_CONTEXT_HEADER: "signed"})
    with pytest.raises(HTTPException) as exc:
        route(body, request)
    assert exc.value.status_code == 403
