"""TASK-2045 guarded purge on a test-owned PostgreSQL schema, never a live bot."""
from __future__ import annotations

import threading

import pytest
from sqlalchemy import text

from llm_bawt import bot_purge_execute as ex
from llm_bawt import bot_purge_preview as preview
from tests.test_bot_data_purge_pg import purge_db, seed  # noqa: F401 (shared isolated fixture)

pytestmark = pytest.mark.integration


@pytest.fixture
def db(request):
    engine, _unused = request.getfixturevalue("purge_db")
    seed(engine, keep_profile=True)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE bot_profile_tags (bot_slug TEXT, tag_slug TEXT)")
        conn.exec_driver_sql("""INSERT INTO bot_profile_tags VALUES
            ('source','private'),('source','coder'),('dash','private')""")
        for ddl in ("ADD COLUMN status TEXT DEFAULT 'ok'", "ADD COLUMN end_reason TEXT",
                    "ADD COLUMN ended_at TIMESTAMP DEFAULT now()",
                    "ADD COLUMN created_at TIMESTAMP DEFAULT now()",
                    # Columns turn_admission.classify_target reads.
                    "ADD COLUMN agent_session_key TEXT", "ADD COLUMN agent_request_id TEXT",
                    "ADD COLUMN agent_backend TEXT", "ADD COLUMN agent_steer_caps TEXT",
                    "ADD COLUMN user_id TEXT"):
            conn.exec_driver_sql(f"ALTER TABLE turn_logs {ddl}")
    return engine


def count(engine, sql):
    with engine.connect() as conn:
        return conn.exec_driver_sql(sql).scalar_one()


def run(engine, *, delete_profile=True, digest=None, slug="source"):
    if digest is None:
        digest = preview.preview_bot_purge(engine, slug, delete_profile=delete_profile)["manifest_digest"]
    return ex.execute_bot_purge(engine, slug, manifest_digest=digest, delete_profile=delete_profile)


def assert_untouched(engine):
    assert count(engine, "SELECT count(*) FROM bot_profiles WHERE slug='source'") == 1
    assert count(engine, "SELECT count(*) FROM bot_profile_tags WHERE bot_slug='source'") == 2
    assert count(engine, "SELECT count(*) FROM messages WHERE bot_id='source'") == 1
    assert count(engine, "SELECT count(*) FROM sessions WHERE bot_id='source'") == 3


def test_delete_and_purge_removes_profile_tags_and_data_in_one_commit(db):
    manifest = preview.preview_bot_purge(db, "source", delete_profile=True)
    rows = {r["key"]: r for r in manifest["inventory"]}
    assert manifest["scope"] == "delete_profile_and_purge_data" and manifest["execution_enabled"] is True
    assert rows["bot_profiles"]["disposition"] == rows["bot_profile_tags"]["disposition"] == "deleted"
    assert rows["bot_profile_tags"]["count"] == 2

    result = run(db, digest=manifest["manifest_digest"])
    assert result["profile_deleted"] is True
    assert result["deleted_rows"]["bot_profiles"] == 1
    assert result["deleted_rows"]["bot_profile_tags"] == 2  # privacy tag included
    assert result["deleted_rows"]["messages"] == 1
    assert {r["key"] for r in result["retained"]} >= {"bot_profile_changes", "inter_bot_deliveries"}
    assert count(db, "SELECT count(*) FROM bot_profiles WHERE slug='source'") == 0
    assert count(db, "SELECT count(*) FROM bot_profile_tags WHERE bot_slug='source'") == 0
    assert count(db, "SELECT count(*) FROM sessions WHERE bot_id='source'") == 0
    # Other bots and retained audit are untouched.
    assert count(db, "SELECT count(*) FROM bot_profile_tags WHERE bot_slug='dash'") == 1
    assert count(db, "SELECT count(*) FROM messages WHERE bot_id='dash'") == 1
    assert count(db, "SELECT count(*) FROM bot_profile_changes WHERE bot_slug='source'") == 1


def test_standalone_purge_keeps_profile_and_tags(db):
    result = run(db, delete_profile=False)
    assert result["profile_deleted"] is False
    assert "bot_profiles" not in result["deleted_rows"]
    assert count(db, "SELECT count(*) FROM bot_profiles WHERE slug='source'") == 1
    assert count(db, "SELECT count(*) FROM bot_profile_tags WHERE bot_slug='source'") == 2
    assert count(db, "SELECT count(*) FROM messages WHERE bot_id='source'") == 0


def test_digest_is_action_specific(db):
    standalone = preview.preview_bot_purge(db, "source")["manifest_digest"]
    with pytest.raises(ex.PurgeStale):
        run(db, digest=standalone)
    assert_untouched(db)


@pytest.mark.parametrize("digest", ["", None, "not-a-digest"])
def test_missing_or_malformed_digest_is_refused_before_any_query(db, digest):
    with pytest.raises(ex.PurgeStale) as refused:
        ex.execute_bot_purge(db, "source", manifest_digest=digest, delete_profile=True)
    assert refused.value.status == 409
    assert_untouched(db)


@pytest.mark.parametrize("change", [
    "UPDATE bot_profiles SET last_session_id=NULL WHERE slug='source'",
    "INSERT INTO bot_profile_tags VALUES ('source','reviewer')",
    "INSERT INTO messages VALUES ('m-new','source','s-active')",
])
def test_stale_preview_is_409_and_nothing_is_deleted(db, change):
    digest = preview.preview_bot_purge(db, "source", delete_profile=True)["manifest_digest"]
    with db.begin() as conn:
        conn.exec_driver_sql(change)
    with pytest.raises(ex.PurgeStale) as refused:
        run(db, digest=digest)
    assert refused.value.status == 409
    assert count(db, "SELECT count(*) FROM bot_profiles WHERE slug='source'") == 1
    assert count(db, "SELECT count(*) FROM sessions WHERE bot_id='source'") == 3


def test_running_turn_is_423(db):
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO turn_logs(id,bot_id,status,ended_at) VALUES ('t-live','source','streaming',NULL)")
    with pytest.raises(ex.PurgeLocked) as refused:
        run(db)
    assert refused.value.status == 423 and refused.value.code == "active"
    assert_untouched(db)


@pytest.mark.parametrize("status", ["streaming", "reserved"])
def test_old_open_turn_still_blocks(db, status):
    # Age is not termination evidence: an abandoned open row must be recovered first.
    with db.begin() as conn:
        conn.exec_driver_sql(f"""INSERT INTO turn_logs(id,bot_id,status,ended_at,created_at)
            VALUES ('t-old','source','{status}',NULL, now() - interval '7 hours')""")
    with pytest.raises(ex.PurgeLocked) as refused:
        run(db)
    assert refused.value.code == "active"
    assert_untouched(db)


def advisory_locks(engine):
    return count(engine, "SELECT count(*) FROM pg_locks WHERE locktype='advisory'")  # timing-ok: post-hoc leaked-lock count after the purge settled, not a wait or poll


def test_turn_admitted_while_purge_waits_is_seen_and_refused(db):
    """Admission holds the bot's lock and commits an open turn while purge waits for it."""
    digest = preview.preview_bot_purge(db, "source", delete_profile=True)["manifest_digest"]
    admitter = db.connect()
    tx = admitter.begin()
    admitter.execute(text("SELECT pg_advisory_xact_lock(hashtext('source'))"))  # turn_admission.lock_target
    admitter.execute(text("INSERT INTO turn_logs(id,bot_id,status,ended_at) VALUES ('t-new','source','streaming',NULL)"))
    errors = []

    def purge():
        try:
            run(db, digest=digest)
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    worker = threading.Thread(target=purge)
    worker.start()
    worker.join(0.5)
    assert worker.is_alive()  # waiting on the admission fence, nothing deleted yet
    tx.commit()
    admitter.close()
    worker.join(10)
    # The new turn is visible to the snapshot: it changes the counted turn rows
    # (stale) and is an open turn (active). Either way nothing is deleted.
    assert len(errors) == 1 and isinstance(errors[0], ex.PurgeRefused)
    assert errors[0].code in {"stale_preview", "active"}
    assert_untouched(db)
    assert count(db, "SELECT count(*) FROM turn_logs WHERE id='t-new'") == 1
    assert advisory_locks(db) == 0


def test_admission_cannot_start_while_purge_runs(db, monkeypatch):
    """While the purge holds the fence, the admission lock is unavailable to others."""
    seen = []
    original = ex._check_active

    def probe(conn, slug, tables):
        with db.connect() as other:
            seen.append(other.execute(text("SELECT pg_try_advisory_xact_lock(hashtext('source'))")).scalar_one())
            other.rollback()
        return original(conn, slug, tables)

    monkeypatch.setattr(ex, "_check_active", probe)
    assert run(db)["profile_deleted"] is True
    assert seen == [False]
    assert advisory_locks(db) == 0


def test_admission_held_past_the_wait_is_busy_and_releases_nothing(db, monkeypatch):
    monkeypatch.setattr(ex, "ADMISSION_WAIT_SECONDS", 0.2)
    digest = preview.preview_bot_purge(db, "source", delete_profile=True)["manifest_digest"]
    admitter = db.connect()
    tx = admitter.begin()
    admitter.execute(text("SELECT pg_advisory_xact_lock(hashtext('source'))"))
    try:
        with pytest.raises(ex.PurgeLocked) as refused:
            run(db, digest=digest)
        assert refused.value.code == "busy"
    finally:
        tx.rollback()
        admitter.close()
    assert_untouched(db)
    assert advisory_locks(db) == 0
    assert run(db, digest=digest)["profile_deleted"] is True


def test_refusal_inside_the_snapshot_still_releases_the_fence(db):
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO turn_logs(id,bot_id,status,ended_at) VALUES ('t-live','source','streaming',NULL)")
    with pytest.raises(ex.PurgeLocked):
        run(db)
    assert advisory_locks(db) == 0


def test_undelivered_message_to_bot_is_423(db):
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO inter_bot_deliveries VALUES ('d-q','dash','source','QUEUED','{}',NULL)")
    with pytest.raises(ex.PurgeLocked):
        run(db)
    assert_untouched(db)


def test_reserved_identity_is_423(db):
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('system', NULL)")
    with pytest.raises(ex.PurgeLocked) as refused:
        run(db, slug="system")
    assert refused.value.code == "protected"
    assert count(db, "SELECT count(*) FROM bot_profiles WHERE slug='system'") == 1


@pytest.mark.parametrize("delete_profile", [True, False])
def test_shared_storage_refuses_both_actions(db, delete_profile):
    # A profile-less raw alias sharing the storage key makes partition rows unattributable.
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO sessions VALUES ('s-alias','Source!','owner','active')")
    with pytest.raises(ex.PurgeLocked) as refused:
        run(db, delete_profile=delete_profile)
    assert refused.value.code == "protected"
    assert_untouched(db)


def test_missing_profile_is_not_found(db):
    with db.begin() as conn:
        conn.exec_driver_sql("DELETE FROM bot_profiles WHERE slug='source'")
    with pytest.raises(ex.PurgeRefused) as refused:
        run(db)
    assert refused.value.code == "not_found"


def test_cross_bot_reference_rolls_back_everything(db):
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('foreign','dash','s-active')")
    with pytest.raises(ex.PurgeRefused) as refused:
        run(db)
    assert refused.value.code == "dependent"
    assert_untouched(db)


def test_concurrent_profile_writer_is_fenced_423_not_partial(db, monkeypatch):
    monkeypatch.setattr(ex, "LOCK_TIMEOUT", "200ms")
    digest = preview.preview_bot_purge(db, "source", delete_profile=True)["manifest_digest"]
    holder = db.connect()
    tx = holder.begin()
    holder.execute(text("UPDATE bot_profiles SET last_session_id=last_session_id WHERE slug='dash'"))
    try:
        with pytest.raises(ex.PurgeLocked) as refused:
            run(db, digest=digest)
        assert refused.value.code == "busy"
    finally:
        tx.rollback()
        holder.close()
    assert_untouched(db)
    assert run(db, digest=digest)["profile_deleted"] is True


def test_snapshot_is_taken_after_locks(db):
    """A writer that commits while purge waits on the fence is seen, so the digest is stale."""
    digest = preview.preview_bot_purge(db, "source", delete_profile=True)["manifest_digest"]
    holder = db.connect()
    tx = holder.begin()
    holder.execute(text("INSERT INTO bot_profile_tags VALUES ('source','reviewer')"))
    errors = []

    def purge():
        try:
            run(db, digest=digest)
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    worker = threading.Thread(target=purge)
    worker.start()
    worker.join(0.5)
    tx.commit()
    holder.close()
    worker.join(10)
    assert len(errors) == 1 and isinstance(errors[0], ex.PurgeStale)
    assert count(db, "SELECT count(*) FROM bot_profile_tags WHERE bot_slug='source'") == 3
