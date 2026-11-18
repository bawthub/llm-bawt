"""Private-PG proof of the operator history read cursor (TASK-893 slice 2)."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory import history_read_cursor as hrc
from llm_bawt.memory.postgresql import ensure_bot_partitions
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

USER, OTHER = "reader", "someone_else"


@pytest.fixture(scope="module")
def engine(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    InterBotDeliveryStore(require_test_db_config())  # bootstraps inter_bot_deliveries
    with engine.begin() as conn:
        hrc.ensure_schema(conn)
        hrc.ensure_schema(conn)  # idempotent
    try:
        yield engine
    finally:
        engine.dispose()


class Rig:
    def __init__(self, engine):
        self.engine = engine
        tag = uuid.uuid4().hex[:10]
        self.bot, self.peer, self.prot = f"hc{tag}", f"hcpeer{tag}", f"hcprot{tag}"
        self.excluded = [self.prot]
        self.sessions = {}

    def session(self, bot, user=USER):
        key = (bot, user)
        if key not in self.sessions:
            sid = str(uuid.uuid4())
            with self.engine.begin() as conn:
                ensure_bot_partitions(conn, bot)
                conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                                  "VALUES (:id, :bot, :user, 'active')"),
                             {"id": sid, "bot": bot, "user": user})
            self.sessions[key] = sid
        return self.sessions[key]

    def add(self, ts, *, bot=None, user=USER, mid=None, session="owned", content="hi"):
        bot = bot or self.bot
        mid = mid or str(uuid.uuid4())
        sid = self.session(bot, user) if session == "owned" else session
        with self.engine.begin() as conn:
            ensure_bot_partitions(conn, bot)
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                      author_entity_type, author_entity_id)
                VALUES (:bot, :id, 'user', :content, :ts, :sid, 'user', :user)
            """), {"bot": bot, "id": mid, "content": content, "ts": ts, "sid": sid, "user": user})
        return mid

    def consume(self, **kw):
        kw.setdefault("bot_id", self.bot)
        kw.setdefault("user_id", USER)
        kw.setdefault("until", 1000.0)
        return hrc.consume(self.engine, excluded_bot_ids=self.excluded, **kw)

    def cursor(self, bot=None, user=USER):
        with self.engine.connect() as conn:
            return conn.execute(text(
                "SELECT cursor_ts, cursor_message_id, processed_rows_total "
                "FROM history_read_cursors WHERE bot_id=:b AND user_id=:u"),
                {"b": bot or self.bot, "u": user}).first()

    def drain(self, limit, **kw):
        seen = []
        for _ in range(50):
            out = self.consume(limit=limit, dry_run=False, **kw)
            assert out["status"] == "advanced"
            seen.append(out)
            if not out["coverage"]["truncated"]:
                return seen
        raise AssertionError("did not drain")


@pytest.fixture
def rig(engine):
    return Rig(engine)


class _Explode:
    def __getattr__(self, name):
        raise AssertionError(f"database touched: {name}")


def test_protected_and_ambiguous_bots_refused_before_any_query(rig):
    for bot in (rig.prot, rig.prot.upper(), "hc-prot" + rig.prot[6:]):
        with pytest.raises(ValueError):
            hrc.consume(_Explode(), bot_id=bot, user_id=USER, excluded_bot_ids=rig.excluded)
        with pytest.raises(ValueError):
            hrc.fetch_after(_Explode(), bot_id=bot, user_id=USER,
                            excluded_bot_ids=rig.excluded, after=None, until=1)
        with pytest.raises(ValueError):
            hrc.reset_cursor(_Explode(), bot_id=bot, user_id=USER, excluded_bot_ids=rig.excluded)
    with pytest.raises(ValueError):  # one pair only, never a list
        hrc.consume(_Explode(), bot_id=[rig.bot], user_id=USER, excluded_bot_ids=rig.excluded)
    with pytest.raises(ValueError):
        hrc.consume(_Explode(), bot_id=rig.bot, user_id="system", excluded_bot_ids=rig.excluded)


def test_same_id_and_timestamp_in_two_partitions_each_cursor_consumes_its_own(rig):
    shared = str(uuid.uuid4())
    rig.add(5.0, bot=rig.bot, mid=shared, content="mine")
    rig.add(5.0, bot=rig.peer, mid=shared, content="peer's")
    a = rig.consume(limit=1, dry_run=False)
    b = rig.consume(bot_id=rig.peer, limit=1, dry_run=False)
    assert a["cursor_after"] == [5.0, shared] and b["cursor_after"] == [5.0, shared]
    assert a["coverage"]["rows_read"] == 1 and b["coverage"]["rows_read"] == 1
    assert rig.cursor()[2] == 1 and rig.cursor(rig.peer)[2] == 1
    # Neither cursor skipped or repeated: the next pages are empty for both.
    assert rig.consume(limit=1)["coverage"]["rows_read"] == 0
    assert rig.consume(bot_id=rig.peer, limit=1)["coverage"]["rows_read"] == 0


def test_equal_timestamp_keyset_neither_skips_nor_repeats(rig):
    ids = sorted(rig.add(7.0) for _ in range(5))
    rig.add(8.0)
    runs = rig.drain(limit=2)
    assert sum(r["coverage"]["rows_read"] for r in runs) == 6
    assert runs[0]["cursor_after"] == [7.0, ids[1]]
    assert runs[1]["cursor_after"] == [7.0, ids[3]]
    assert rig.cursor()[2] == 6


def test_truncation_does_not_overshoot(rig):
    first = rig.add(1.0)
    rig.add(2.0)
    rig.add(3.0)
    out = rig.consume(limit=1, dry_run=False, until=100.0)
    assert out["coverage"]["truncated"] is True
    assert out["cursor_after"] == [1.0, first]  # last processed row, never `until`
    assert out["coverage"]["window"]["covered_to"] == [1.0, first]
    tail = rig.consume(limit=10, dry_run=False, until=100.0)
    assert tail["coverage"]["truncated"] is False and tail["coverage"]["rows_read"] == 2


def test_cas_conflict_on_existing_cursor_commits_nothing(rig, monkeypatch):
    rig.add(1.0)
    rig.add(2.0)
    rig.consume(limit=1, dry_run=False)
    before = rig.cursor()
    real = hrc.fetch_after

    def racing(conn, **kw):
        monkeypatch.setattr(hrc, "fetch_after", real)
        assert rig.consume(limit=1, dry_run=False)["status"] == "advanced"  # other run wins
        return real(conn, **kw)

    monkeypatch.setattr(hrc, "fetch_after", racing)
    out = rig.consume(limit=1, dry_run=False)
    assert out["status"] == "conflict" and out["cursor_before"] == [before[0], before[1]]
    after = rig.cursor()
    assert after[2] == 2  # only the winner's count was committed


@pytest.mark.parametrize("reset_first", [False, True])
def test_cas_conflict_on_initial_null_cursor_and_first_creation(rig, monkeypatch, reset_first):
    rig.add(1.0)
    if reset_first:  # an existing row with a null cursor: the null-safe CAS path
        hrc.reset_cursor(rig.engine, bot_id=rig.bot, user_id=USER, excluded_bot_ids=rig.excluded)
    real = hrc.fetch_after

    def racing(conn, **kw):
        monkeypatch.setattr(hrc, "fetch_after", real)
        assert rig.consume(limit=5, dry_run=False)["status"] == "advanced"
        return real(conn, **kw)

    monkeypatch.setattr(hrc, "fetch_after", racing)
    out = rig.consume(limit=5, dry_run=False)
    assert out["status"] == "conflict" and out["cursor_before"] is None
    assert rig.cursor()[2] == 1


def test_dry_run_is_default_and_writes_nothing(rig):
    rig.add(1.0)
    out = rig.consume()
    assert out["status"] == "dry_run" and out["coverage"]["rows_read"] == 1
    assert rig.cursor() is None
    hrc.reset_cursor(rig.engine, bot_id=rig.bot, user_id=USER, excluded_bot_ids=rig.excluded)
    rig.consume()
    assert tuple(rig.cursor()) == (None, None, 0)


def test_session_owner_join_excludes_other_users_and_counts_unattributed(rig):
    mine = rig.add(1.0)
    rig.add(2.0, user=OTHER)  # another user's session on the same bot: neither read nor counted
    rig.add(3.0, session=None)  # no session at all
    rig.add(4.0, session=rig.session(rig.peer))  # a session of a different bot
    out = rig.consume(dry_run=False)
    assert out["coverage"]["rows_read"] == 1 and out["cursor_after"] == [1.0, mine]
    assert out["coverage"]["unattributed_partition_rows"] == 2
    # Ownership is the session's user, never author_entity_id.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE messages SET author_entity_id=:o WHERE id=:id"),
                     {"o": OTHER, "id": mine})
    hrc.reset_cursor(rig.engine, bot_id=rig.bot, user_id=USER, excluded_bot_ids=rig.excluded)
    assert rig.consume()["coverage"]["rows_read"] == 1
    assert rig.consume(user_id=OTHER)["coverage"]["rows_read"] == 1


def test_reset_and_rescan_recounts_as_processing(rig):
    ids = [rig.add(float(t)) for t in (1, 2, 3)]
    rig.consume(dry_run=False)
    assert tuple(rig.cursor()) == (3.0, ids[2], 3)
    hrc.reset_cursor(rig.engine, bot_id=rig.bot, user_id=USER, excluded_bot_ids=rig.excluded,
                     to=(1.0, ids[0]))
    again = rig.consume(dry_run=False)
    assert again["cursor_before"] == [1.0, ids[0]] and again["coverage"]["rows_read"] == 2
    assert tuple(rig.cursor()) == (3.0, ids[2], 5)  # processing count, not unique rows
    hrc.reset_cursor(rig.engine, bot_id=rig.bot, user_id=USER, excluded_bot_ids=rig.excluded)
    assert rig.consume(dry_run=False)["coverage"]["rows_read"] == 3


def test_delivery_origin_binds_to_canonical_pair(rig):
    bound = rig.add(1.0, content="Check the backups")
    stray = rig.add(2.0, content="I prefer short answers")
    with rig.engine.begin() as conn:
        for mid, target in ((bound, rig.bot), (stray, rig.peer)):
            conn.execute(text("""
                INSERT INTO inter_bot_deliveries (id, sender_bot_id, author_entity_type,
                    author_entity_id, target_bot_id, message, user_message_id, turn_id, status,
                    metadata_json)
                VALUES (:id, :t, 'user', :u, :t, 'x', :mid, :turn, 'DELIVERED',
                        '{"_request_identity_producer": "schedule"}')
            """), {"id": f"d-{uuid.uuid4().hex}", "t": target, "u": USER, "mid": mid,
                   "turn": f"turn-{uuid.uuid4().hex}"})
    out = rig.consume()
    # Another bot's delivery record never becomes this row's origin.
    assert out["coverage"]["classes"] == {"AUTOMATION/SCHEDULED/high": 1,
                                          "HUMAN/UNKNOWN/medium": 1}
