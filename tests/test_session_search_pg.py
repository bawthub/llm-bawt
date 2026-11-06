"""Real SQL proof in a private temporary PostgreSQL cluster, never live data.

Run: pytest -m integration tests/test_session_search_pg.py
Requires locally installed PostgreSQL binaries; no credentials or shared tables.
"""
import json

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory.session_search import SessionSearch
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def repository(private_pg, private_module_db):
    """Own schema in a module-owned database on the private cluster."""
    engine = create_engine(private_url(private_pg, private_module_db))
    try:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SHOW unix_socket_directories").scalar_one() == "/var/run/postgresql"
        yield from _populate_private_repository(engine)
    finally:
        engine.dispose()


def _populate_private_repository(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text, started_at timestamp, ended_at timestamp, session_metadata jsonb)"))
        rows = [
            ("one", "sample-a", "alice", "active", "2026-01-01", {"title": "Pocket", "agent_session_key_models": {"test": "test-model"}, "agent_session_keys": {"test": "secret-sdk-key"}, "reset_predecessor_ids": ["old"]}),
            ("two", "sample-b", "alice", "archived", "2026-02-01", {"title": "Pocket experiments", "provider": "test-provider"}),
            ("three", "sample-a", "bob", "active", "2026-03-01", {"title": "Pocket private"}),
            ("four", "sample-a", "alice", "deleted", "2026-04-01", {"title": "Pocket deleted"}),
            ("five", "sample-a", None, "archived", "2026-05-01", {"title": "Pocket unowned"}),
            ("six", "sample-a", "alice", "completed", "2026-06-01", {"title": "100%_literal", "agent_session_key_models": ["malformed"], "reset_predecessor_ids": "malformed"}),
            ("seven", "sample-a", "alice", "archived", "2026-07-01", {"title": {"bad": "shape"}, "agent_session_key_models": {"bad": {"nested": "secret"}, "ok": "another-model"}}),
        ]
        for id_, bot, user, status, started, meta in rows:
            conn.execute(text("INSERT INTO sessions VALUES (:id,:bot,:user,:status,:started,NULL,CAST(:meta AS jsonb))"), {"id": id_, "bot": bot, "user": user, "status": status, "started": started, "meta": json.dumps(meta)})
        _populate_first_messages(conn)
    yield SessionSearch(engine)


LONG = ("Investigate the flaky upload path " + "with plenty of context " * 20
        + "and finally the needle-past-300 word")
assert LONG.index("needle-past-300") > 300


def _populate_first_messages(conn):
    """TASK-2061: user carol; messages keyed by the partition bot key, as in production."""
    conn.execute(text("CREATE TABLE messages (bot_id text, id text, role text, content text, timestamp float8, session_id text, author_entity_type text, author_entity_id text)"))
    conn.execute(text("CREATE TABLE inter_bot_deliveries (id text, user_message_id text, target_bot_id text, author_entity_type text, author_entity_id text)"))
    conn.execute(text("CREATE TABLE prompt_occurrences (run_id text, job_id text, delivery_id text, session_id text)"))
    conn.execute(text("CREATE TABLE prompt_schedules (job_id text, owner_user_id text, creator_bot_id text)"))
    sessions = [
        # id, bot, status, started, metadata
        ("long", "sample-a", "archived", "2026-08-01", {"title": LONG[:59] + "…", "title_source": "auto"}),
        ("named", "sample-a", "archived", "2026-08-02", {"title": "Renamed thread", "title_source": "user"}),
        ("bot-start", "sample-b", "archived", "2026-08-03", {}),
        ("sched", "sample-b", "archived", "2026-08-04", {"title": "Hourly watchdog", "title_source": "automation"}),
        ("stale", "sample-a", "archived", "2026-08-05", {"title": "sup", "title_source": "auto"}),
        ("empty", "sample-a", "archived", "2026-08-06", {}),
        ("mixed", "sample-a", "archived", "2026-08-07", {}),
    ]
    for id_, bot, status, started, meta in sessions:
        conn.execute(text("INSERT INTO sessions VALUES (:id,:bot,'carol',:status,:started,NULL,CAST(:meta AS jsonb))"),
                     {"id": id_, "bot": bot, "status": status, "started": started, "meta": json.dumps(meta)})
    messages = [
        # bot key (hyphen dropped), id, role, content, ts, session, author type, author id
        ("samplea", "l0", "user", "/new", 1.0, "long", "user", "carol"),
        ("samplea", "l1", "user", "\t /new", 2.0, "long", "user", "carol"),
        ("samplea", "l2", "user", "   ", 3.0, "long", "user", "carol"),
        ("samplea", "l3", "user", "```\n# \n```", 4.0, "long", "user", "carol"),
        ("samplea", "l4", "user", LONG + "\nsecond line keyword-after-newline", 5.0, "long", "user", "carol"),
        ("samplea", "l5", "user", "later message mentions zebra", 6.0, "long", "user", "carol"),
        ("samplea", "n1", "user", "first words of the renamed one", 1.0, "named", "user", "carol"),
        ("sampleb", "b1", "user", "Message from bot 'snark': please review", 1.0, "bot-start", "bot", "snark"),
        ("sampleb", "s1", "user", "Run the hourly watchdog", 1.0, "sched", "bot", "snark"),
        ("samplea", "t0", "assistant", "sup", 1.0, "stale", "bot", "sample-a"),
        ("samplea", "t1", "user", "Look into the outage", 2.0, "stale", None, None),
        # Marker plus non-ASCII Python whitespace: the old writer skipped it.
        ("samplea", "x1", "user", "#\u00a0", 1.0, "mixed", "bot", "gavel"),
        ("samplea", "x2", "user", "Please investigate searchable-needle", 2.0, "mixed", "user", "carol"),
    ]
    for row in messages:
        conn.execute(text("INSERT INTO messages VALUES (:b,:i,:r,:c,:t,:s,:at,:ai)"),
                     dict(zip(("b", "i", "r", "c", "t", "s", "at", "ai"), row)))
    conn.execute(text("INSERT INTO inter_bot_deliveries VALUES ('d1','s1','sample-b','bot','snark')"))
    conn.execute(text("INSERT INTO prompt_occurrences VALUES ('o1','job-1','d1','sched')"))
    conn.execute(text("INSERT INTO prompt_schedules VALUES ('job-1','carol','snark')"))


def test_owned_ranked_search_and_metadata_projection(repository):
    page = repository.search(user_id="alice", query="pocket")
    assert [row["id"] for row in page["sessions"]] == ["one", "two"]
    assert page["total_count"] == 2
    first = page["sessions"][0]
    assert first["models"] == "test-model" and first["has_predecessor"]
    assert "session_metadata" not in first
    assert repository.search(user_id="alice", query="secret-sdk-key")["total_count"] == 0
    assert repository.search(user_id="alice", query="test-model")["total_count"] == 1
    assert repository.search(user_id="alice", query="test-provider")["total_count"] == 1


def test_paging_has_real_total_even_past_end(repository):
    assert repository.search(user_id="alice", query="pocket", limit=1)["total_count"] == 2
    assert repository.search(user_id="alice", query="pocket", limit=1, offset=1)["sessions"][0]["id"] == "two"
    assert repository.search(user_id="alice", query="pocket", offset=30) == {"sessions": [], "total_count": 2}


def test_archive_alias_bot_time_sort_and_empty_query(repository):
    assert repository.search(user_id="alice", status="active")["total_count"] == 1
    archived = repository.search(user_id="alice", status="archived", sort_by="oldest")
    assert [row["id"] for row in archived["sessions"]] == ["two", "six", "seven"]
    assert all(row["status"] == "archived" for row in archived["sessions"])
    assert repository.search(user_id="alice", bot_id="sample-b")["total_count"] == 1
    assert repository.search(user_id="alice", since=1780272000)["total_count"] == 2
    assert repository.search(user_id="alice", sort_by="recent")["sessions"][0]["id"] == "seven"
    assert repository.search(user_id="alice", query="sample-a")["total_count"] == 3


def test_literal_queries_and_malformed_metadata(repository):
    assert repository.search(user_id="alice", query="%_")["sessions"][0]["id"] == "six"
    assert repository.search(user_id="alice", query="' OR TRUE --")["total_count"] == 0
    assert repository.search(user_id="alice", query="nested")["total_count"] == 0
    hit = repository.search(user_id="alice", query="another-model")["sessions"][0]
    assert hit["title"] == "" and hit["models"] == "another-model"
    assert repository.search(user_id="unknown")["total_count"] == 0


def _by_id(page):
    return {row["id"]: row for row in page["sessions"]}


def test_full_first_message_is_searchable_past_old_cuts(repository):
    for needle in ("needle-past-300", "keyword-after-newline", "investigate the flaky"):
        page = repository.search(user_id="carol", query=needle)
        assert list(_by_id(page)) == ["long"], needle
    row = _by_id(repository.search(user_id="carol", query="needle-past-300"))["long"]
    assert row["first_message_text"] == LONG + "\nsecond line keyword-after-newline"
    assert row["initiated_by"] == {"kind": "user", "id": "carol"}
    # The legacy stored copy is still returned for older clients, uncut by search.
    assert row["title"] == LONG[:59] + "…" and row["title_source"] == "auto"


def test_only_the_first_eligible_message_matches_not_later_ones(repository):
    assert repository.search(user_id="carol", query="zebra")["total_count"] == 0


def test_stored_auto_copy_is_not_a_match_source(repository):
    # "sup" survives only in the stale auto copy and an assistant row.
    assert "stale" not in _by_id(repository.search(user_id="carol", query="sup"))
    stale = _by_id(repository.search(user_id="carol", query="outage"))["stale"]
    assert stale["initiated_by"] == {"kind": "unknown", "id": None}


def test_marker_with_python_whitespace_does_not_displace_the_real_request(repository):
    mixed = _by_id(repository.search(user_id="carol", query="searchable-needle"))["mixed"]
    assert mixed["first_message_text"] == "Please investigate searchable-needle"
    assert mixed["initiated_by"] == {"kind": "user", "id": "carol"}


def _retired_writer_title(content):
    """The retired auto-title rule (f1252ef9 derive_thread_title), uncut."""
    import re

    text_ = (content or "").strip()
    if not text_ or text_.startswith("/"):
        return None
    text_ = re.sub(r"```[a-zA-Z0-9_-]*", " ", text_).replace("`", " ")
    for line in text_.splitlines():
        line = line.strip().lstrip("#>*- ").strip()
        if line:
            return re.sub(r"\s+", " ", line).strip() or None
    return None


ELIGIBILITY_CASES = [
    "hello", "", " ", "\u00a0", "/new", " \u3000/new", "#", "# ", "#\u00a0", "#\x1c", "#\u00a0#",
    "\u00a0#", "- * > #", "-\u2003-", "```", "```python\n```", "```\n# \n```", "`#`", "``x``",
    "#\u2028real", "#\x1dreal", "#\x1f#", ">\x85>", "# \r\n> \r\nok", "\x1f", "a\u00a0",
    "```js\n# heading", "   \t\n\n  *", "**bold**", "\u2029", "#\u2029\u00a0#",
]


def test_sql_eligibility_matches_the_retired_writer(repository):
    from llm_bawt.sessions.first_message import eligible_sql, first_message_params

    with repository.engine.connect() as conn:
        for case in ELIGIBILITY_CASES:
            sql = f"SELECT {eligible_sql('m')} FROM (SELECT 'user'::text AS role, CAST(:c AS text) AS content) m"
            got = conn.execute(text(sql), {"c": case, **first_message_params()}).scalar_one()
            assert got is (_retired_writer_title(case) is not None), repr(case)


def test_given_names_match_and_rank_as_labels(repository):
    named = _by_id(repository.search(user_id="carol", query="renamed thread"))["named"]
    assert named["rank"] == 4 and named["title_source"] == "user"
    assert _by_id(repository.search(user_id="carol", query="hourly watchdog"))["sched"]["rank"] == 4


def test_initiator_filter_and_schedule_provenance(repository):
    sched = _by_id(repository.search(user_id="carol", initiated_by="schedule"))
    assert list(sched) == ["sched"] and sched["sched"]["initiated_by"] == {"kind": "schedule", "id": "job-1"}
    assert list(_by_id(repository.search(user_id="carol", initiated_by="bot"))) == ["bot-start"]
    assert "empty" not in _by_id(repository.search(user_id="carol", initiated_by="unknown"))
    with pytest.raises(ValueError):
        repository.search(user_id="carol", initiated_by="robot")


def test_page_only_lookup_matches_full_path(repository):
    fast = _by_id(repository.search(user_id="carol", limit=200))
    full = _by_id(repository.search(user_id="carol", limit=200, sort_by="title"))
    assert fast.keys() == full.keys()
    for key in fast:
        assert fast[key]["first_message_text"] == full[key]["first_message_text"]
        assert fast[key]["initiated_by"] == full[key]["initiated_by"]
    assert fast["empty"]["first_message_text"] is None and fast["empty"]["initiated_by"] is None


def test_batched_first_message_lookup_is_owner_keyed(repository):
    from llm_bawt.sessions.first_message import fetch_first_messages

    with repository.engine.connect() as conn:
        found = fetch_first_messages(conn, [
            {"id": "long", "bot_id": "sample-a", "user_id": "carol"},
            {"id": "bot-start", "bot_id": "sample-b", "user_id": "carol"},
            {"id": "empty", "bot_id": "sample-a", "user_id": "carol"},
            {"id": "named", "bot_id": "sample-a", "user_id": "mallory"},  # wrong owner
            {"id": "sched", "bot_id": "sample-a", "user_id": "carol"},  # wrong bot
        ])
    assert found["long"]["first_message_text"].startswith("Investigate the flaky")
    assert found["bot-start"]["initiated_by"] == {"kind": "bot", "id": "snark"}
    assert found["empty"] == {"first_message_text": None, "initiated_by": None}
    assert "named" not in found and "sched" not in found


def test_thread_list_query_searches_full_history_before_limit(repository):
    from types import SimpleNamespace

    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    with repository.engine.begin() as conn:  # the list query also reads archived_at
        conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at timestamp"))
    manager = object.__new__(PostgreSQLShortTermManager)
    manager.bot_id = "sample-a"
    manager._backend = SimpleNamespace(engine=repository.engine, _messages_table_name="messages")

    def ids(**kwargs):
        return [row["id"] for row in manager.list_sessions(user_id="carol", **kwargs)]

    # Past every old preview cut, after a newline, and in a later message.
    assert ids(query="NEEDLE-past-300") == ["long"]
    assert ids(query="keyword-after-newline") == ["long"]
    assert ids(query="zebra") == ["long"]
    assert ids(query="renamed thread") == ["named"]
    # The stale auto copy "sup" is not a source; the assistant reply is.
    assert ids(query="sup") == ["stale"]
    assert ids(query="%_") == []
    assert len(ids(query="", limit=200)) == 5  # sample-a sessions of carol
    assert ids(query="zebra", limit=1) == ["long"]
