"""Ownership rollout rehearsals on disposable PostgreSQL; no live identities."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from uuid import uuid4

import pytest
from sqlalchemy import text

from llm_bawt.memory import ownership_migration as migration
from llm_bawt.profiles import EntityType, ProfileManager
from llm_bawt.runtime_settings import BotProfileStore
from tests.pg_lock_wait import wait_blocked_by
from tests.private_postgres import private_app_pg  # noqa: F401
from tests.test_memory_update_pg import memory_rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def rig(memory_rig):  # noqa: F811
    storage, backend, bot = memory_rig
    profiles = ProfileManager(storage.config)
    BotProfileStore(storage.config).upsert({"slug": bot})
    users = [f"synthetic-{uuid4().hex}" for _ in range(2)]
    for user in users:
        profiles.get_or_create_profile(EntityType.USER, user)
    with backend.engine.begin() as conn:
        migration.ensure_schema(conn)
        # Rehearse a legacy profile that predates the new-profile hook.
        conn.execute(text(f"DELETE FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                     {"bot": backend.bot_id_sanitized})
    return backend, bot, users


def add_session(conn, bot, user, *, legacy=False):
    session = str(uuid4())
    conn.execute(text("""
        INSERT INTO sessions (id, bot_id, user_id, started_at, status, session_metadata)
        VALUES (:id, :bot, :user, now(), 'archived', CAST(:meta AS jsonb))
    """), {"id": session, "bot": bot, "user": user,
           "meta": '{"legacy_backfill":true}' if legacy else '{}'})
    return session


def add_message(conn, backend, session, user, *, forgotten=False, author_type="user"):
    table = backend._forgotten_table_name if forgotten else backend._messages_table_name
    source = str(uuid4())
    conn.execute(text(f"""
        INSERT INTO {table} (id, role, content, timestamp, session_id,
                             author_entity_type, author_entity_id)
        VALUES (:id, 'user', 'synthetic evidence', 1, :session, :type, :user)
    """), {"id": source, "session": session, "type": author_type, "user": user})
    return source


def seed(rig, *, legacy=False, forgotten=False):
    backend, bot, users = rig
    memory = str(uuid4())
    backend.add_memory(memory, "Synthetic retained personal fact")
    with backend.engine.begin() as conn:
        session = add_session(conn, bot, users[0], legacy=legacy)
        add_message(conn, backend, session, users[0], forgotten=forgotten)
    return memory


def run(rig, **kwargs):
    backend, bot, _users = rig
    with backend.engine.begin() as conn:
        return migration.activate(conn, bot, actor_id="synthetic-operator", **kwargs)


def owner(rig, memory):
    backend, _bot, _users = rig
    with backend.engine.connect() as conn:
        return conn.execute(text(f"SELECT owner_user_id FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": memory}).scalar_one()


def test_operator_cli_counts_only_dry_run_apply_and_repeat(rig, capsys):
    import json
    from scripts.memory_ownership import main
    memory = seed(rig)
    backend, bot, users = rig
    args = ["activate", "--bot", bot, "--actor", "synthetic-operator"]
    assert main(args, engine=backend.engine) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "would_activate"
    assert owner(rig, memory) is None
    assert main([*args, "--apply"], engine=backend.engine) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "activated"
    assert bot not in output and all(user not in output for user in users)
    assert owner(rig, memory) == users[0]
    assert main([*args, "--apply"], engine=backend.engine) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_active"


def test_operator_cli_refusal_reports_remedy_without_source_data(rig, capsys):
    import json
    from scripts.memory_ownership import main
    seed(rig, legacy=True)
    backend, bot, _users = rig
    assert main(["activate", "--bot", bot, "--actor", "synthetic-operator"],
                engine=backend.engine) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "legacy_evidence_requires_attestation"
    assert result["remedy"]
    assert "Synthetic retained" not in str(result)


def test_expansion_and_dry_run_neither_stamp_nor_activate(rig):
    memory = seed(rig)
    backend, bot, _users = rig
    with backend.engine.begin() as conn:
        migration.ensure_schema(conn)
        assert migration.status(conn)["unmarked_bots"] >= 1
    result = run(rig)
    assert result["status"] == "would_activate"
    assert result["counts"]["stamp_memories"] == 1
    assert owner(rig, memory) is None
    assert bot not in str(result) and memory not in str(result)
    with backend.engine.connect() as conn:
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()


def test_single_user_activation_is_atomic_idempotent_after_second_user(rig):
    memory = seed(rig)
    backend, bot, users = rig
    with backend.engine.connect() as conn:
        unmarked_before = migration.status(conn)["unmarked_bots"]
    assert run(rig, dry_run=False)["status"] == "activated"
    with backend.engine.connect() as conn:
        assert migration.status(conn)["unmarked_bots"] == unmarked_before - 1
    assert owner(rig, memory) == users[0]
    with backend.engine.begin() as conn:
        add_session(conn, bot, users[1])
    assert run(rig, dry_run=False)["status"] == "already_active"
    assert owner(rig, memory) == users[0]
    with backend.engine.connect() as conn:
        receipt = conn.execute(text(f"SELECT * FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                               {"bot": backend.bot_id_sanitized}).mappings().one()
        assert receipt["actor_id"] == "synthetic-operator"
        assert receipt["activated_at"] is not None
        assert receipt["counts"]["stamp_memories"] == 1


def test_multiuser_marks_without_backfill_and_attestation_cannot_override(rig):
    memory = seed(rig)
    backend, bot, users = rig
    with backend.engine.begin() as conn:
        add_session(conn, bot, users[1])
    assert run(rig, attest_user=users[0], dry_run=False)["reason"] == "other_human_owner"
    result = run(rig, dry_run=False)
    assert result["status"] == "activated" and result["counts"]["stamp_memories"] == 0
    assert owner(rig, memory) is None


@pytest.mark.parametrize("forgotten", [False, True])
def test_legacy_default_user_is_not_proof_but_explicit_attestation_can_resolve(rig, forgotten):
    memory = seed(rig, legacy=True, forgotten=forgotten)
    assert run(rig, dry_run=False)["reason"] == "legacy_evidence_requires_attestation"
    assert owner(rig, memory) is None
    assert run(rig, attest_user=rig[2][0])["status"] == "would_activate"
    assert owner(rig, memory) is None
    assert run(rig, attest_user=rig[2][0], dry_run=False)["status"] == "activated"
    assert owner(rig, memory) == rig[2][0]


@pytest.mark.parametrize("forgotten", [False, True])
def test_attested_legacy_sources_can_be_forgotten_but_not_reused_for_writes(rig, forgotten):
    from llm_bawt.memory.ownership import MemoryAuthority
    memory = seed(rig, legacy=True, forgotten=forgotten)
    backend, _bot, users = rig
    table = backend._forgotten_table_name if forgotten else backend._messages_table_name
    with backend.engine.begin() as conn:
        source = conn.execute(text(f"SELECT id FROM {table}")).scalar_one()
        conn.execute(text(f"UPDATE {backend._memories_table_name} SET source_message_ids=jsonb_build_array(CAST(:id AS text))"),
                     {"id": source})
    assert run(rig, attest_user=users[0], dry_run=False)["status"] == "activated"
    assert owner(rig, memory) == users[0]
    authority = MemoryAuthority(user_id=users[0])
    assert not backend.add_memory(str(uuid4()), "Reuse guessed sources", source_message_ids=[source],
                                  authority=authority).persisted
    assert backend.delete_memories_by_source_message_ids([source], authority=MemoryAuthority(user_id=users[1])) == 0
    assert backend.delete_memories_by_source_message_ids([source], authority=authority) == 1


@pytest.mark.parametrize("kind", ["service", "missing_profile", "bot_author"])
def test_uncertain_evidence_never_infers_human_owner(rig, kind):
    memory = seed(rig)
    backend, bot, users = rig
    with backend.engine.begin() as conn:
        if kind == "service":
            conn.execute(text("UPDATE entity_profiles SET is_service=true WHERE entity_id=:user"),
                         {"user": users[0]})
        elif kind == "missing_profile":
            add_session(conn, bot, "synthetic-no-profile")
        else:
            session = add_session(conn, bot, users[0])
            add_message(conn, backend, session, "synthetic-bot", author_type="bot")
    assert run(rig, dry_run=False)["reason"] == "unproven_evidence_requires_attestation"
    assert owner(rig, memory) is None
    if kind == "service":
        assert run(rig, attest_user=users[0])["reason"] == "attestation_requires_human_profile"


@pytest.mark.parametrize("problem", ["missing_source", "malformed_sources", "foreign_explicit_owner"])
def test_row_provenance_cannot_be_laundered_by_single_human_session(rig, problem):
    memory = seed(rig)
    backend, _bot, users = rig
    with backend.engine.begin() as conn:
        if problem == "foreign_explicit_owner":
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id=:user WHERE id=:id"),
                         {"user": users[1], "id": memory})
        else:
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET source_message_ids=CAST(:sources AS jsonb) WHERE id=:id"),
                         {"sources": '["synthetic-unknown-source"]' if problem == "missing_source" else '{}',
                          "id": memory})
    assert run(rig, dry_run=False)["reason"] == "unproven_row_provenance_requires_attestation"
    if problem == "foreign_explicit_owner":
        assert run(rig, attest_user=users[0])["reason"] == "other_explicit_owner"


@pytest.mark.parametrize("forgotten", [False, True])
def test_explicit_mark_only_preserves_ambiguous_rows_and_never_uses_default_user(rig, monkeypatch, forgotten):
    import json
    from llm_bawt.memory.ownership import MemoryAuthority
    from scripts.memory_ownership import main
    backend, bot, users = rig
    monkeypatch.setattr(backend.config, "DEFAULT_USER", users[0])
    service = "synthetic-service"
    ProfileManager(backend.config).get_or_create_profile(EntityType.USER, service)
    with backend.engine.begin() as conn:
        conn.execute(text("UPDATE entity_profiles SET is_service=true WHERE entity_id=:user"), {"user": service})
        source_a = add_message(conn, backend, add_session(conn, bot, users[0]), users[0])
        source_b = add_message(conn, backend, add_session(conn, bot, users[1]), users[1])
        legacy = add_message(conn, backend, add_session(conn, bot, users[0], legacy=True), users[0], forgotten=forgotten)
        service_source = add_message(conn, backend, add_session(conn, bot, service), service)
        unknown_source = add_message(conn, backend, add_session(conn, bot, "synthetic-unknown"), "synthetic-unknown")
    cases = {"unsourced": [], "legacy": [legacy], "service": [service_source],
             "unknown": [unknown_source], "missing": ["missing-message"],
             "mixed": [source_a, source_b], "valid_a": [source_a], "valid_b": [source_b]}
    ids = {}
    for label, sources in cases.items():
        ids[label] = str(uuid4())
        assert backend.add_memory(ids[label], f"Synthetic {label}", source_message_ids=sources).persisted
    malformed = str(uuid4())
    assert backend.add_memory(malformed, "Synthetic malformed").persisted
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._memories_table_name} SET source_message_ids='{{}}'::jsonb WHERE id=:id"),
                     {"id": malformed})
    assert run(rig)["reason"] == "legacy_evidence_requires_attestation"
    assert run(rig, attest_user=users[0])["reason"] == "other_human_owner"
    preview = run(rig, mark_only=True)
    counts = preview["counts"]
    assert preview["status"] == "would_activate" and preview["mode"] == "mark_only"
    assert counts["stamp_memories"] == 0 and counts["human_owners"] == 2
    assert counts["restricted_memories"] == 7
    assert counts["restricted_unsourced"] == 1
    assert counts["restricted_legacy"] == 1
    assert counts["restricted_service_or_unknown"] == 4
    assert counts["restricted_mixed_human"] == 1
    assert run(rig)["status"] == "refused"  # dry run did not activate
    assert main(["activate", "--bot", bot, "--actor", "synthetic-operator", "--mark-only", "--apply"],
                engine=backend.engine) == 0
    assert run(rig)["mode"] == "mark_only"
    assert run(rig, attest_user=users[0], dry_run=False)["status"] == "already_active"
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT count(*) FROM {backend._memories_table_name} WHERE owner_user_id IS NOT NULL")).scalar_one() == 0
        receipt = conn.execute(text(f"SELECT counts FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                               {"bot": backend.bot_id_sanitized}).scalar_one()
        assert receipt == counts
    for user in users:
        authority = MemoryAuthority(user_id=user)
        assert len(backend.list_recent(n=20, authority=authority)) == 9
        assert backend.update_memory(ids["legacy"], content="Claimed legacy", authority=authority) is None
        assert not backend.delete_memory(ids["legacy"], authority=authority)
        assert not backend.add_memory(str(uuid4()), "Claimed legacy", source_message_ids=[legacy],
                                      authority=authority).persisted
    # The DEFAULT_USER's fresh identical write must not dedup into guessed history.
    new = backend.add_memory(str(uuid4()), "Synthetic legacy", authority=MemoryAuthority(user_id=users[0]))
    assert new.persisted and new.memory_id != ids["legacy"]
    assert all(user not in json.dumps(preview) for user in users)


def test_mark_only_and_attestation_cannot_be_combined(rig):
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(rig, mark_only=True, attest_user=rig[2][0])


def test_storage_alias_refuses_activation_even_with_attestation(rig):
    seed(rig)
    backend, bot, users = rig
    with backend.engine.begin() as conn:
        add_session(conn, bot.replace("-", ""), users[1])
    assert run(rig, attest_user=users[0], dry_run=False)["reason"] == "storage_identity_collision"


def test_preexisting_second_user_transaction_is_visible_after_lock_wait(rig):
    memory = seed(rig)
    backend, bot, users = rig
    with ThreadPoolExecutor(max_workers=1) as pool:
        with backend.engine.begin() as writer:
            add_session(writer, bot, users[1])
            holder = writer.execute(text("SELECT pg_backend_pid()")).scalar_one()
            future = pool.submit(run, rig, dry_run=False)
            wait_blocked_by(backend.engine, holder, future)
        result = future.result(timeout=10)
    assert result["counts"]["human_owners"] == 2
    assert result["counts"]["stamp_memories"] == 0
    assert owner(rig, memory) is None


def test_activation_refuses_a_potentially_stale_repeatable_read_snapshot(rig):
    seed(rig)
    backend, bot, _users = rig
    with backend.engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with pytest.raises(ValueError, match="READ COMMITTED"):
            migration.activate(conn, bot, actor_id="synthetic-operator")


def test_transaction_rollback_restores_marker_and_rows(rig):
    memory = seed(rig)
    backend, bot, _users = rig
    with backend.engine.connect() as conn:
        tx = conn.begin()
        migration.activate(conn, bot, actor_id="synthetic-operator", dry_run=False)
        tx.rollback()
    assert owner(rig, memory) is None
    assert run(rig)["status"] == "would_activate"


def test_activation_excludes_concurrent_second_user_session(rig):
    memory = seed(rig)
    backend, bot, users = rig
    pids = Queue()

    def second_user():
        with backend.engine.begin() as writer:
            pids.put(writer.execute(text("SELECT pg_backend_pid()")).scalar_one())
            add_session(writer, bot, users[1])

    with ThreadPoolExecutor(max_workers=1) as pool:
        with backend.engine.begin() as conn:
            migration.activate(conn, bot, actor_id="synthetic-operator", dry_run=False)
            holder = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
            future = pool.submit(second_user)
            pids.get(timeout=10)
            wait_blocked_by(backend.engine, holder, future)
        future.result(timeout=10)
    assert owner(rig, memory) == users[0]
    assert run(rig)["status"] == "already_active"


def test_indexes_attach_and_new_partition_inherits_then_down_preserves_evidence(rig):
    memory = seed(rig)
    backend, _bot, users = rig
    run(rig, dry_run=False)
    migration.build_owner_index(backend.engine)
    migration.build_owner_index(backend.engine)
    from llm_bawt.memory.postgresql import ensure_bot_partitions
    with backend.engine.begin() as conn:
        ensure_bot_partitions(conn, f"synthetic_new_{uuid4().hex[:8]}")
    migration.build_owner_index(backend.engine)
    with backend.engine.begin() as conn:
        assert conn.execute(text("SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(:idx)"),
                            {"idx": migration.OWNER_INDEX}).scalar_one()
        archive = migration.downgrade(conn)
        records = conn.execute(text("SELECT kind, data FROM memory_ownership_rollback WHERE archive_id=:id"),
                               {"id": archive}).all()
        assert any(kind == "owner" and data["id"] == memory and data["owner_user_id"] == users[0]
                   for kind, data in records)
        assert any(kind == "activation" and data["bot_id"] == backend.bot_id_sanitized
                   for kind, data in records)
        assert migration.downgrade(conn) is None
        # Re-expansion does not silently reactivate archived owners.
        migration.ensure_schema(conn)
    assert owner(rig, memory) is None
    assert run(rig)["status"] == "would_activate"


def test_new_profile_marks_but_existing_profile_is_not_reactivated(memory_rig):  # noqa: F811
    storage, backend, bot = memory_rig
    store = BotProfileStore(storage.config)
    store.upsert({"slug": bot})
    with backend.engine.begin() as conn:
        assert conn.execute(text(f"SELECT mode FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                            {"bot": backend.bot_id_sanitized}).scalar_one() == "new"
        conn.execute(text(f"DELETE FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                     {"bot": backend.bot_id_sanitized})
    store.upsert({"slug": bot, "description": "Existing profile update"})
    backend._ensure_tables_exist()
    with backend.engine.connect() as conn:
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()


def test_borrowed_profile_rollback_also_rolls_back_marker(memory_rig):  # noqa: F811
    from sqlmodel import Session
    storage, backend, bot = memory_rig
    store = BotProfileStore(storage.config)
    with Session(backend.engine) as session:
        store.upsert({"slug": bot}, session=session)
        assert session.connection().execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                            {"bot": backend.bot_id_sanitized}).first()
        session.rollback()
    with backend.engine.connect() as conn:
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()
        assert not conn.execute(text("SELECT 1 FROM bot_profiles WHERE slug=:bot"), {"bot": bot}).first()


def test_profile_recreated_over_retained_memory_does_not_activate(memory_rig):  # noqa: F811
    storage, backend, bot = memory_rig
    backend.add_memory(str(uuid4()), "Retained pre-profile memory")
    BotProfileStore(storage.config).upsert({"slug": bot})
    with backend.engine.connect() as conn:
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()


def test_new_profile_storage_collision_does_not_activate_legacy_identity(memory_rig):  # noqa: F811
    storage, backend, bot = memory_rig
    store = BotProfileStore(storage.config)
    store.upsert({"slug": bot})
    with backend.engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                     {"bot": backend.bot_id_sanitized})
    # The lossy partition map is identical even with another dash.
    store.upsert({"slug": bot + "-"})
    with backend.engine.connect() as conn:
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()


def test_direct_profile_creation_cannot_reuse_retired_alias(memory_rig):  # noqa: F811
    from llm_bawt.bot_slug_aliases import ensure_schema
    storage, backend, bot = memory_rig
    store = BotProfileStore(storage.config)
    target = "synthetic-target-" + uuid4().hex
    store.upsert({"slug": target})
    with backend.engine.begin() as conn:
        ensure_schema(conn)
        conn.execute(text("INSERT INTO bot_slug_aliases (old_slug, bot_slug) VALUES (:old, :target)"),
                     {"old": backend.bot_id_sanitized, "target": target})
    with pytest.raises(ValueError, match="retired alias"):
        store.upsert({"slug": bot})
    with backend.engine.connect() as conn:
        assert not conn.execute(text("SELECT 1 FROM bot_profiles WHERE slug=:bot"), {"bot": bot}).first()
        assert not conn.execute(text(f"SELECT 1 FROM {migration.ACTIVATIONS} WHERE bot_id=:bot"),
                                {"bot": backend.bot_id_sanitized}).first()


def test_retained_activation_blocks_http_bot_identity_reuse(memory_rig):  # noqa: F811
    from fastapi import HTTPException
    from llm_bawt.bot_create_transaction import BotCreateTransaction
    storage, backend, bot = memory_rig
    BotProfileStore(storage.config)
    with backend.engine.begin() as conn:
        migration.mark_new_bot(conn, bot)
        with pytest.raises(HTTPException) as raised:
            BotCreateTransaction._refuse_retained_identity(conn, bot, backend.bot_id_sanitized)
        assert raised.value.status_code == 409
