"""Bot tag rules with no database (TASK-1329, BOT_TAGS.md §3, §5, §10.2).

Fixture bots only: ``fixture-private`` (tag), ``fixture-column`` (legacy column
false, no tag), ``fixture-quiet`` (no_global_search) and ``fixture-inactive``.
"""
from __future__ import annotations

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.policy import TrustedCaller
from llm_bawt.bot_tags import BehaviorTag, BotTags, TagPolicyUnavailable, TagSnapshot, sql_allowed
from llm_bawt.bot_tags import facade
from llm_bawt.bot_tags.vocabulary import OPT_IN_BEHAVIOR_TAGS

CATALOG = ["coder", "inactive", "no_global_search", "private", "reviewer"]
TAGS = {
    "fixture-public": ["reviewer"],
    "fixture-private": ["private"],
    "fixture-column": [],
    "fixture-quiet": ["no_global_search"],
    "fixture-inactive": ["inactive"],
}


def rows(**changes):
    out = [{"slug": slug, "include_in_global_search": slug != "fixture-column",
            "tags": tags, "catalog": CATALOG} for slug, tags in sorted(TAGS.items())]
    for row in out:
        row.update(changes.get(row["slug"], {}))
    return out


@pytest.fixture
def snap():
    return TagSnapshot.from_rows(rows())


# --- §3 effective rules -------------------------------------------------------


@pytest.mark.parametrize("bot", ["fixture-private", "fixture-column"])
def test_private_by_tag_or_column_implies_no_global_search(snap, bot):
    assert snap.has(bot, BehaviorTag.PRIVATE)
    assert snap.has(bot, BehaviorTag.NO_GLOBAL_SEARCH)
    assert snap.excluded_reason(bot) == "private bot"
    assert bot in snap.private_bots and bot not in snap.searchable_bots


def test_sets_split_private_quiet_and_inactive(snap):
    assert snap.private_bots == {"fixture-private", "fixture-column"}
    # Inactive is a dispatch/discovery rule, not a search rule.
    assert snap.searchable_bots == {"fixture-public", "fixture-inactive"}
    assert not snap.has("fixture-quiet", BehaviorTag.PRIVATE)
    assert snap.excluded_reason("fixture-quiet") is None
    assert snap.excluded_reason("fixture-inactive") == "inactive bot"
    assert snap.has("fixture-public", "reviewer") and not snap.has("fixture-public", "coder")


def test_an_unknown_bot_is_private_but_not_inactive(snap):
    assert snap.has("fixture-unknown", BehaviorTag.PRIVATE)
    assert snap.has("fixture-unknown", BehaviorTag.NO_GLOBAL_SEARCH)
    assert not snap.has("fixture-unknown", BehaviorTag.INACTIVE)
    assert snap.excluded_reason("fixture-unknown") == "private bot"


def test_bot_slugs_are_normalized_before_lookup(snap):
    assert snap.has("  Fixture-Private ", BehaviorTag.PRIVATE)
    assert snap.excluded_reason("FIXTURE-INACTIVE") == "inactive bot"


def test_an_unknown_tag_raises_instead_of_answering(snap):
    with pytest.raises(ValueError, match="unknown bot tag"):
        snap.has("fixture-public", "privat")


def test_unavailable_is_private_and_inactive_for_everyone():
    snap = TagSnapshot.unavailable()
    for bot in ("fixture-public", "fixture-unknown"):
        assert all(snap.has(bot, tag) for tag in BehaviorTag if tag not in OPT_IN_BEHAVIOR_TAGS)
        # An opt-in grant fails closed the other way (TASK-1362).
        assert not snap.has(bot, BehaviorTag.EXTERNAL)
        assert snap.excluded_reason(bot) == "private bot"
    assert snap.searchable_bots == frozenset()
    # A role answer would be a guess: it raises.
    with pytest.raises(TagPolicyUnavailable):
        snap.has("fixture-public", "reviewer")


def test_legacy_rows_without_tag_columns_mean_no_tags():
    snap = TagSnapshot.from_rows([{"slug": "fixture-public", "include_in_global_search": True},
                                  {"slug": "fixture-column", "include_in_global_search": False}])
    assert snap.private_bots == {"fixture-column"}
    assert snap.searchable_bots == {"fixture-public"}


@pytest.mark.parametrize("bad", [
    {"fixture-public": {"slug": "Fixture-Public"}},
    {"fixture-public": {"slug": " fixture-public"}},
    {"fixture-public": {"include_in_global_search": None}},
    {"fixture-public": {"include_in_global_search": 1}},
    {"fixture-public": {"tags": ["ghost"]}},
    {"fixture-public": {"tags": "private"}},
    {"fixture-public": {"tags": ["private", "private"]}},
    {"fixture-public": {"catalog": ["inactive", "no_global_search", "private"]}},
], ids=["case", "space", "null-flag", "int-flag", "unknown-tag", "string-array",
        "duplicate-tag", "inconsistent-catalog"])
def test_malformed_snapshot_rows_are_refused(bad):
    with pytest.raises(TagPolicyUnavailable):
        TagSnapshot.from_rows(rows(**bad))


def test_a_catalog_missing_a_behaviour_tag_or_a_duplicate_bot_is_refused():
    catalog = ["no_global_search", "private"]
    with pytest.raises(TagPolicyUnavailable, match="behaviour"):
        TagSnapshot.from_rows([{"slug": "fixture-public", "include_in_global_search": True,
                                "tags": [], "catalog": catalog}])
    row = {"slug": "fixture-public", "include_in_global_search": True}
    with pytest.raises(TagPolicyUnavailable, match="malformed"):
        TagSnapshot.from_rows([row, dict(row)])


def test_a_catalog_not_yet_seeded_with_an_opt_in_tag_grants_it_to_nobody(snap):
    """Code ahead of the seed (or a rollback window) must not poison the snapshot."""
    assert "external" not in CATALOG
    assert snap.available
    assert not any(snap.has(bot, BehaviorTag.EXTERNAL) for bot in [*TAGS, "fixture-unknown"])
    assert snap.has("fixture-inactive", BehaviorTag.INACTIVE)


def test_an_opt_in_tag_is_granted_only_where_assigned():
    catalog = sorted([*CATALOG, "external"])
    snap = TagSnapshot.from_rows([
        {"slug": "fixture-outside", "include_in_global_search": True,
         "tags": ["external", "inactive"], "catalog": catalog},
        {"slug": "fixture-public", "include_in_global_search": True, "tags": [], "catalog": catalog},
    ])
    assert snap.has("fixture-outside", BehaviorTag.EXTERNAL)
    assert not snap.has("fixture-public", BehaviorTag.EXTERNAL)
    # An unknown bot is private, never external.
    assert not snap.has("fixture-unknown", BehaviorTag.EXTERNAL)


def test_an_opt_in_tag_still_must_be_in_the_catalog_to_be_assigned():
    with pytest.raises(TagPolicyUnavailable, match="unknown tag"):
        TagSnapshot.from_rows([{"slug": "fixture-outside", "include_in_global_search": True,
                                "tags": ["external"], "catalog": CATALOG}])


def test_more_than_the_bound_is_refused():
    many = [{"slug": f"fixture-{i}", "include_in_global_search": True} for i in range(facade.MAX_BOTS + 1)]
    with pytest.raises(TagPolicyUnavailable, match="too many"):
        TagSnapshot.from_rows(many)


# --- §5.3 SQL fragment ------------------------------------------------------


@pytest.mark.parametrize("column", ["target_bot_id", "d.target_bot_id; --", "lower(d.x)", ""])
def test_sql_allowed_refuses_an_unqualified_or_unsafe_column(column):
    with pytest.raises(ValueError, match="unsafe"):
        sql_allowed(column, [BehaviorTag.PRIVATE])


def test_sql_allowed_refuses_a_bad_prefix_empty_exclude_or_unknown_tag():
    with pytest.raises(ValueError, match="unsafe"):
        sql_allowed("d.bot_id", [BehaviorTag.PRIVATE], prefix="x; drop")
    with pytest.raises(ValueError, match="at least one"):
        sql_allowed("d.bot_id", [])
    with pytest.raises(ValueError):
        sql_allowed("d.bot_id", ["ghost"])


def test_no_global_search_also_excludes_private_and_requires_a_profile():
    sql, params = sql_allowed("d.target_bot_id", [BehaviorTag.NO_GLOBAL_SEARCH], prefix="tgt")
    assert params == {"tgt_tags": ["no_global_search", "private"]}
    assert "tgt_t.bot_slug = lower(d.target_bot_id)" in sql
    assert "FROM bot_profiles tgt_p" in sql and "tgt_p.include_in_global_search" in sql


def test_inactive_alone_is_one_clause_with_no_profile_requirement():
    sql, params = sql_allowed("s.bot_id", [BehaviorTag.INACTIVE])
    assert params == {"tagx_tags": ["inactive"]}
    assert "bot_profiles" not in sql and sql.count("EXISTS") == 1


# --- BotAccess: readable vs searchable --------------------------------------


def access(caller_bot: str | None = "fixture-public"):
    caller = TrustedCaller(caller_bot, "owner", "turn") if caller_bot else None
    return BotPolicyResolver._compile(rows(), caller)


def test_readable_excludes_private_and_searchable_also_excludes_quiet():
    view = access()
    assert view.readable_bots == {"fixture-public", "fixture-quiet", "fixture-inactive"}
    assert view.searchable_bots == {"fixture-public", "fixture-inactive"}


def test_a_quiet_bot_is_targetable_but_not_an_aggregate_filter_for_others():
    view = access()
    assert view.require_target("fixture-quiet") == "fixture-quiet"
    for bot in ("fixture-quiet", "fixture-private", "fixture-column", "fixture-unknown"):
        with pytest.raises(BotAccessError, match="access_denied"):
            view.aggregate_scope(bot)
    unfiltered = {p.bot_id for p in view.aggregate_scope().partitions}
    assert unfiltered == {"fixture-public", "fixture-inactive"}


def test_a_quiet_or_private_caller_still_sees_itself():
    quiet = access("fixture-quiet")
    assert [p.bot_id for p in quiet.aggregate_scope("fixture-quiet").partitions] == ["fixture-quiet"]
    assert quiet.aggregate_scope("fixture-quiet").partitions[0].authority == "caller"
    private = access("fixture-private")
    assert private.require_target("fixture-private") == "fixture-private"
    with pytest.raises(BotAccessError, match="access_denied"):
        access().require_target("fixture-private")


@pytest.mark.parametrize("bad_rows", [[], rows(**{"fixture-public": {"tags": ["ghost"]}})],
                         ids=["empty", "malformed"])
def test_the_resolver_refuses_an_empty_or_malformed_snapshot(bad_rows):
    with pytest.raises(BotAccessError, match="policy_unavailable"):
        BotPolicyResolver._compile(bad_rows, None)


# --- Sync facade and the snapshot read --------------------------------------


class _Conn:
    def __init__(self, *, in_tx: bool, fail: bool = False):
        self.in_tx, self.fail, self.calls = in_tx, fail, []

    def in_transaction(self):
        return self.in_tx

    def begin_nested(self):
        self.calls.append("SAVEPOINT")
        return self._ctx()

    def begin(self):
        self.calls.append("BEGIN")
        return self._ctx()

    def _ctx(self):
        conn = self

        class _Ctx:
            def __enter__(self):
                return conn

            def __exit__(self, exc_type, *a):
                conn.calls.append("ROLLBACK" if exc_type else "RELEASE/COMMIT")
                return False
        return _Ctx()

    def execute(self, statement):
        sql = str(statement).strip()
        self.calls.append(sql.split()[0] + " " + sql.split()[1])
        if sql.startswith("SELECT") and self.fail:
            raise RuntimeError("relation bot_profile_tags does not exist")

        class _Result:
            def mappings(self_inner):
                return [{"slug": "fixture-public", "include_in_global_search": True,
                         "tags": [], "catalog": CATALOG}]
        return _Result()


def test_inside_a_transaction_the_read_is_a_savepoint_with_no_set_local():
    conn = _Conn(in_tx=True)
    assert facade.read_snapshot_rows_on(conn)[0]["slug"] == "fixture-public"
    assert conn.calls == ["SAVEPOINT", "SELECT p.slug,", "RELEASE/COMMIT"]


def test_outside_a_transaction_the_read_is_its_own_bounded_read_only_one():
    conn = _Conn(in_tx=False)
    facade.read_snapshot_rows_on(conn)
    assert conn.calls == ["BEGIN", "SET TRANSACTION", "SET LOCAL", "SET LOCAL",
                          "SELECT p.slug,", "RELEASE/COMMIT"]


def test_a_failed_savepoint_read_is_private_and_rolls_back_only_the_savepoint():
    conn = _Conn(in_tx=True, fail=True)
    snap = BotTags().snapshot(conn)
    assert not snap.available and snap.has("fixture-public", BehaviorTag.INACTIVE)
    assert conn.calls == ["SAVEPOINT", "SELECT p.slug,", "ROLLBACK"]


def test_a_facade_with_no_engine_or_a_broken_one_is_private_and_inactive(monkeypatch):
    monkeypatch.setattr(BotTags, "_resolve_engine", lambda self: None)
    assert BotTags().excluded_reason("fixture-public") == "private bot"

    class _Broken:
        def connect(self):
            raise OSError("connection refused")

    tags = BotTags(_Broken())
    assert tags.has("fixture-public", BehaviorTag.PRIVATE)
    assert tags.has("fixture-public", BehaviorTag.INACTIVE)
