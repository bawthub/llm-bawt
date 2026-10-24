"""TASK-1323 ``messages_find`` on real PostgreSQL: scope, budget, handles.

Private PostgreSQL, isolated schema, synthetic fixtures only (no protected
data). Covers the §5.0 denial tests that apply to ``find`` (1, 2, 3, 5, 6, 8,
9, 10), the bot-filter no-oracle rule, the §5.1 budget invariants I1 (wire
payload ≤ BUDGET) and I2 (fill never moves a primary anchor), and the §5.3
handle contract.
"""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotPolicyResolver, TrustedCaller
from llm_bawt.bot_tags import catalog
from llm_bawt.memory.message_tsv import ensure_message_tsv
from llm_bawt.retrieval import find as find_mod
from llm_bawt.retrieval.find import BUDGET, FindRequest, MessageFinder
from llm_bawt.retrieval.handles import HandleStore
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config
from tests.test_fts_query_pg import assert_no_stored_vector, result_columns

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

NOW = 1_800_000_000.0
USER = "nick"
# slug -> include_in_global_search. "empty" is public and holds no messages.
# "priv" is private through the legacy column, "tagged" through a tag only.
BOTS = {"pub": True, "pub2": True, "empty": True, "priv": False, "tagged": True}
NOTHING = {"status": "no_match", "rungs": [["exact_all", 0], ["exact_any", 0]]}


class World:
    def __init__(self, engine):
        self.engine = engine
        self.handles = HandleStore()
        self.finder = MessageFinder(engine, self.handles, clock=lambda: NOW)
        self._n = 0

    def session(self, sid, bot, user=USER, status="active"):
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO sessions VALUES (:id, :bot, :user, :status)"),
                         {"id": sid, "bot": bot, "user": user, "status": status})
        return sid

    def message(self, partition, session, content, *, role="user", author=("user", USER),
                age=60.0, mid=None):
        self._n += 1
        mid = mid or f"m{self._n}"
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO messages (id, bot_id, session_id, role, content, timestamp, "
                              "author_entity_type, author_entity_id) "
                              "VALUES (:id, :bot, :sid, :role, :c, :ts, :at, :ai)"),
                         {"id": mid, "bot": partition, "sid": session, "role": role, "c": content,
                          "ts": NOW - age - self._n * 1e-3, "at": author[0], "ai": author[1]})
        return mid

    def flag(self, bot, public):
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE bot_profiles SET include_in_global_search = :p WHERE slug = :s"),
                         {"p": public, "s": bot})

    def access(self, caller="pub", user=USER):
        return BotPolicyResolver(self.engine).resolve_sync(TrustedCaller(caller, user, "turn"))

    def raw(self, caller="pub", user=USER, *, trigger_id=None, fill=True, **request) -> str:
        return self.finder.find(self.access(caller, user), FindRequest(**request),
                                trigger_id=trigger_id, fill=fill)

    def find(self, *args, **kwargs) -> dict:
        return json.loads(self.raw(*args, **kwargs))


@pytest.fixture
def world(private_app_pg):
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "msgfind_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    with eng.begin() as conn:
        conn.execute(text("""CREATE TABLE messages (
            id text, bot_id text, session_id text, role text, content text,
            timestamp double precision, author_entity_type text, author_entity_id text)"""))
        ensure_message_tsv(conn, table_was_new=True)
        conn.execute(text("CREATE TABLE bot_profiles (slug text, include_in_global_search boolean)"))
        for slug, public in BOTS.items():
            conn.execute(text("INSERT INTO bot_profiles VALUES (:s, :p)"), {"s": slug, "p": public})
        catalog.ensure_schema(conn)
        catalog.seed(conn)
        catalog.assign(conn, "tagged", ["private"], "test")
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text)"))
    w = World(eng)
    for bot in ("pub", "pub2", "priv", "tagged"):
        w.session(f"s_{bot}", bot)
    try:
        yield w
    finally:
        eng.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def no_handle(resp: dict) -> dict:
    return {k: v for k, v in resp.items() if k != "handle"}


def card_ids(resp: dict) -> set[str]:
    return {c["id"] for c in resp.get("best", []) + resp.get("latest", [])} | {
        a for c in resp.get("best", []) + resp.get("latest", []) for a in c["also"]}


def nothing(token: str) -> dict:
    return {**NOTHING, "absent": [token]}


# -- §5.0 denial tests -----------------------------------------------------

def test_1_protected_legacy_row_changes_nothing(world):
    """A blank-author user row naming a protected bot leaves no trace anywhere."""
    world.message("pub", "s_pub", "TASK-777 shipped today")
    world.message("pub", "s_pub", "TASK-777 bot reply", role="assistant", author=("bot", "pub"))
    world.session("s_pub_b", "pub")
    legacy = world.message("pub", "s_pub_b", "Message from bot 'priv': TASK-777 secret",
                           author=(None, None))
    leaked = world.find(query="TASK-777", speaker="user")
    with world.engine.begin() as conn:
        conn.execute(text("DELETE FROM messages WHERE id = :id"), {"id": legacy})
    control = world.find(query="TASK-777", speaker="user")

    assert leaked["status"] == "ok" and leaked["total"] == 1
    assert leaked["outside_filters"] == {"speaker": 1}
    assert no_handle(leaked) == no_handle(control)
    assert legacy not in json.dumps(leaked) and "secret" not in json.dumps(leaked)


def test_2_malformed_or_unknown_sender_is_omitted(world):
    world.message("pub", "s_pub", "TASK-778 from a ghost", author=("bot", "ghost"))
    world.message("pub", "s_pub", "TASK-778 unknown type", author=("robot", USER))
    world.message("pub", "s_pub", "Message from bot 'pub2': TASK-778", author=(None, None))
    world.message("pub", "s_pub", "TASK-778 ghost reply", role="assistant", author=("bot", "ghost"))
    assert world.find(query="TASK-778") == nothing("TASK-778")


def test_3_policy_flip_drops_deliveries_before_the_next_page(world):
    """Each page re-resolves policy: a delivery author excluded after page 1
    never appears on page 2."""
    users = [world.message("pub", world.session(f"u{i}", "pub"), f"flip note {i}", age=60 + i)
             for i in range(12)]
    deliveries = [world.message("pub", world.session(f"d{i}", "pub"), f"flip delivery {i}",
                                author=("bot", "pub2"), age=1000 + i) for i in range(2)]
    first = world.find()
    assert first["status"] == "ok" and first["total"] == 14
    assert [c["id"] for c in first["latest"]] == users[:10]
    world.flag("pub2", False)
    second = world.find(handle=first["handle"])
    assert [c["id"] for c in second["latest"]] == users[10:]
    assert second["total"] == 12
    assert not card_ids(second) & set(deliveries)
    assert second["facets"]["speaker"] == {"user": 12}


def test_5_self_access_and_no_reach_into_a_protected_partition(world):
    own = world.message("priv", "s_priv", "TASK-780 own reply", role="assistant",
                        author=("bot", "priv"))
    world.session("s_priv_b", "priv")
    world.message("priv", "s_priv_b", "TASK-780 legacy reply", role="assistant",
                  author=(None, None))
    public = world.message("pub", "s_pub", "TASK-780 public note")
    world.message("tagged", "s_tagged", "TASK-780 tag-private reply", role="assistant",
                  author=("bot", "tagged"))

    mine = world.find("priv", query="TASK-780")
    assert mine["total"] == 2 and card_ids(mine) == {own, public}
    assert mine["facets"]["bot"] == {"priv": 1, "pub": 1}

    theirs = world.find("pub", query="TASK-780")
    assert card_ids(theirs) == {public}
    # No existence oracle: a protected bot, an unknown bot and an empty
    # in-scope bot all answer the same no_match.
    by_private = world.find("pub", query="TASK-780", bot="priv")
    assert by_private == world.find("pub", query="TASK-780", bot="nosuch")
    assert by_private == world.find("pub", query="TASK-780", bot="tagged")
    assert by_private == world.find("pub", query="TASK-780", bot="empty") == nothing("TASK-780")


def test_5_handle_cannot_carry_a_caller_into_a_protected_partition(world):
    for i in range(12):
        world.message("priv", world.session(f"p{i}", "priv"), f"private note {i}",
                      role="assistant", author=("bot", "priv"))
    first = world.find("priv")
    assert world.find("pub", handle=first["handle"]) == {
        "status": "unavailable", "reason": "handle_expired"}


def test_6_user_pin(world):
    world.session("s_eve", "pub", user="eve")
    world.message("pub", "s_eve", "TASK-781 eve said", author=("user", "eve"))
    world.message("pub", "s_eve", "TASK-781 nick in eve's session")
    world.message("pub", "s_eve", "TASK-781 reply to eve", role="assistant", author=("bot", "pub"))
    assert world.find(query="TASK-781") == nothing("TASK-781")


def test_6_dead_sessions_are_out_of_scope(world):
    world.session("s_gone", "pub", status="deleted")
    world.message("pub", "s_gone", "TASK-786 deleted session")
    assert world.find(query="TASK-786") == nothing("TASK-786")


def test_8_trigger_message_never_appears(world):
    trigger = world.message("pub", "s_pub", "find TASK-782 for me", age=1)
    world.session("s_pub_b", "pub")
    earlier = world.message("pub", "s_pub_b", "TASK-782 was about retries")
    resp = world.find(query="TASK-782", trigger_id=trigger)
    assert resp["total"] == 1 and card_ids(resp) == {earlier}
    assert trigger not in json.dumps(resp)
    browse = world.find(trigger_id=trigger)
    assert trigger not in card_ids(browse)


def test_9_mismatched_session(world):
    world.message("pub", "s_priv", "TASK-783 pub row in a priv session")
    world.message("priv", "s_pub", "TASK-783 priv row in a pub session", role="assistant",
                  author=("bot", "priv"))
    world.message("pub", "s_pub2", "TASK-783 pub row in a pub2 session")
    assert world.find("priv", query="TASK-783") == nothing("TASK-783")
    assert world.find("pub", query="TASK-783") == nothing("TASK-783")


def test_10_partial_author(world):
    for author in (("", "pub"), ("bot", ""), ("robot", "pub"), (None, "pub"), ("bot", None)):
        world.message("pub", "s_pub", "TASK-784 partial", role="assistant", author=author)
    assert world.find(query="TASK-784") == nothing("TASK-784")

    # A role=user row authored by the partition bot is a delivery, not a bot reply.
    world.message("pub", "s_pub", "TASK-785 self delivery", author=("bot", "pub"))
    assert world.find(query="TASK-785")["facets"]["speaker"] == {"delivery": 1}
    assert world.find(query="TASK-785", speaker="bot")["status"] == "no_match"


# -- cards, filters, budget ------------------------------------------------

def test_session_collapse_also_ids_and_snippets(world):
    ids = [world.message("pub", "s_pub", f"TASK-790 step {i}", age=60 + i) for i in range(5)]
    resp = world.find(query="TASK-790")
    assert resp["total"] == 5 and len(resp["best"]) == 1 and resp["latest"] == []
    card = resp["best"][0]
    assert card["hits"] == 5 and len(card["also"]) == 3
    assert {card["id"], *card["also"]} <= set(ids)
    assert "TASK-790" in card["snippet"] and card["matched"] == ["TASK-790"]


def test_outside_filters_relax_one_caller_filter(world):
    world.message("pub", "s_pub", "TASK-791 asked")
    world.message("pub", "s_pub", "TASK-791 answered", role="assistant", author=("bot", "pub"))
    world.message("pub2", "s_pub2", "TASK-791 elsewhere")
    resp = world.find(query="TASK-791", bot="pub", speaker="user")
    assert resp["total"] == 1
    assert resp["outside_filters"] == {"bot": 1, "speaker": 1}


def test_i1_payload_never_exceeds_budget(world):
    filler = "lorem ipsum dolor " * 160
    for i in range(40):
        bot = "pub" if i % 2 else "pub2"
        sid = world.session(f"big{i}", bot)
        for j in range(3):
            world.message(bot, sid, f"{filler} budget_token_{i} TASK-792 {filler}",
                          age=60 + i * 10 + j)
    for request in ({"query": "TASK-792 budget_token_7"}, {"query": "lorem ipsum"}, {},
                    {"query": "TASK-792", "speaker": "user", "since": NOW - 10_000}):
        raw = world.raw(**request)
        assert len(raw) <= BUDGET, (request, len(raw))
        resp = json.loads(raw)
        assert resp["status"] == "ok"
        # The budget is met by fitting real snippets, never by serving blank cards.
        cards = resp["best"] + resp["latest"]
        assert cards and all(c["snippet"] for c in cards), request


def test_i2_fill_never_moves_primary_anchors(world):
    for i in range(2):
        world.message("pub", world.session(f"both{i}", "pub"), f"TASK-900 touched alpha_beta {i}",
                      age=500 + i)
    for i in range(5):
        world.message("pub", world.session(f"id{i}", "pub"), f"TASK-900 only {i}", age=60 + i)
    for i in range(3):
        world.message("pub2", world.session(f"sn{i}", "pub2"), f"alpha_beta only {i}", age=90 + i)
    bare = world.find(query="TASK-900 alpha_beta", fill=False)
    full = world.find(query="TASK-900 alpha_beta")
    assert bare["rung"] == full["rung"] == "exact_all"
    n = len(bare["best"])
    assert n == 2
    assert full["best"][:n] == bare["best"]
    assert full["latest"] == bare["latest"]
    for key in ("total", "facets", "omitted"):
        assert full[key] == bare[key]
    filled = full["best"][n:]
    assert filled and {c["rung"] for c in filled} <= {"exact_any", "parts", "parts_loose"}
    assert not card_ids({"best": filled}) & card_ids(bare)


def test_overflow_fails_closed(world, monkeypatch):
    world.message("pub", "s_pub", "TASK-793 something")
    monkeypatch.setattr(find_mod, "BUDGET", 150)
    assert world.find(query="TASK-793") == {"status": "unavailable", "reason": "response_overflow"}


def test_stored_vector_never_leaves_sql(world):
    world.message("pub", "s_pub", "TASK-794 stop button pressed")
    with result_columns(world.engine) as columns:
        world.find(query="TASK-794 stop button", speaker="user")
        world.find(query="stop button")
        world.find()
    assert_no_stored_vector(columns)


# -- handles (§5.3) --------------------------------------------------------

def browse_world(world, n):
    return [world.message("pub", world.session(f"h{i}", "pub"), f"handle note {i}", age=60 + i)
            for i in range(n)]


def test_handle_pages_are_disjoint_then_exhausted(world):
    ids = browse_world(world, 15)
    first = world.find()
    second = world.find(handle=first["handle"])
    assert [c["id"] for c in first["latest"]] == ids[:10]
    assert [c["id"] for c in second["latest"]] == ids[10:]
    assert second["latest"][0]["ref"] == "c1.0"
    assert world.find(handle=first["handle"]) == {"status": "no_match", "reason": "exhausted"}


def test_dense_session_cannot_hide_other_sessions(world):
    """Loopy r1 blocker: 205 hot hits in one session must not use up the scan
    on any page. Every session surfaces across pages, each exactly once, and
    the hot session's card counts all of its hits."""
    world.session("dense", "pub")
    with world.engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (id, bot_id, session_id, role, content, "
                          "timestamp, author_entity_type, author_entity_id) "
                          "VALUES (:id, 'pub', 'dense', 'user', :content, :ts, 'user', :author)"),
                     [{"id": f"dense-{i}", "content": "TASK-796 hot session",
                       "ts": NOW - 60 - i * 0.001, "author": USER} for i in range(205)])
    others = {f"o{i}": world.message("pub", world.session(f"o{i}", "pub"),
                                     f"TASK-796 older {i}", age=1000 + i) for i in range(12)}
    first = world.find(query="TASK-796")
    assert first["total"] == 217
    assert first["best"][0]["session"] == "dense" and first["best"][0]["hits"] == 205
    assert len(first["best"]) == 6 and len(first["latest"]) == 4
    second = world.find(handle=first["handle"])
    pages = [c["session"] for r in (first, second) for c in r["best"] + r["latest"]]
    assert sorted(pages) == sorted(["dense", *others])
    assert set(others.values()) <= card_ids(first) | card_ids(second)
    assert world.find(handle=first["handle"]) == {"status": "no_match", "reason": "exhausted"}


def test_handle_is_owner_bound(world):
    browse_world(world, 12)
    handle = world.find()["handle"]
    expired = {"status": "unavailable", "reason": "handle_expired"}
    assert world.find("pub2", handle=handle) == expired
    assert world.find("pub", "eve", handle=handle) == expired
    assert world.find(handle="not-a-handle") == expired
    assert world.find(handle=handle)["status"] == "ok"


def test_handle_expires(world):
    browse_world(world, 12)
    now = [0.0]
    world.handles = world.finder.handles = HandleStore(ttl=60, clock=lambda: now[0])
    handle = world.find()["handle"]
    now[0] = 61
    assert world.find(handle=handle) == {"status": "unavailable", "reason": "handle_expired"}


def test_page_limit(world, monkeypatch):
    browse_world(world, 25)
    monkeypatch.setattr(find_mod, "MAX_PAGES", 2)
    handle = world.find()["handle"]
    assert world.find(handle=handle)["status"] == "ok"
    assert world.find(handle=handle) == {"status": "no_match", "reason": "page_limit"}


def test_concurrent_page_loses_cas(world):
    class Busy(HandleStore):
        def replace(self, *args, **kwargs):
            return False

    browse_world(world, 12)
    world.finder.handles = Busy()
    handle = world.find()["handle"]
    assert world.find(handle=handle) == {"status": "unavailable", "reason": "handle_busy"}


def test_handle_with_other_arguments_is_invalid(world):
    browse_world(world, 12)
    handle = world.find()["handle"]
    assert world.find(handle=handle, query="x") == {"status": "unavailable",
                                                    "reason": "invalid_request"}


def test_no_match_issues_no_handle(world):
    world.handles._entries.clear()
    assert "handle" not in world.find(query="TASK-795")
    assert not world.handles._entries  # nothing stored for a page that never served
