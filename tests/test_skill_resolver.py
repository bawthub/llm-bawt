"""The skill resolver (TASK-1309): precedence, exclusions, caps and the manifest."""
from __future__ import annotations

from pathlib import Path

from agent_bridge.skill_sources import SkillBody, SkillEntry, SkillSource
from llm_bawt.skills.harness import capabilities_for
from llm_bawt.skills.prompts import render_section
from llm_bawt.skills.resolver import (
    CATALOG_UNREADABLE,
    NO_EFFECT,
    NOT_IN_BUNDLE,
    OVER_CAP,
    PROTECTED_BOT,
    SOURCE_MISSING,
    UNSUPPORTED_HARNESS,
    Assignment,
    BotFacts,
    Caps,
    canonical_mode,
    newly_over_cap,
    resolve,
)

CAPS = Caps(per_skill=100, total=150)


class _Source(SkillSource):
    """Skills with fixed token sizes; ``read_body`` honours ``missing_body``."""

    kind = "fake"

    def __init__(self, sizes: dict[str, int], missing_body: tuple[str, ...] = ()):
        self.sizes = sizes
        self.missing_body = missing_body

    def list(self):
        return tuple(SkillEntry(ref, ref, "", self.kind, Path("/skills") / ref)
                     for ref in sorted(self.sizes))

    def read_body(self, ref):
        if ref not in self.sizes or ref in self.missing_body:
            return None
        return SkillBody(ref, f"body of {ref}", self.sizes[ref], f"sha-{ref}-{self.sizes[ref]}")


def _facts(**overrides) -> BotFacts:
    base = dict(slug="fixture-one", bot_type="agent", harness="claude-code", bundle=None,
                tags=frozenset({"coder", "reviewer"}), private=False)
    return BotFacts(**{**base, **overrides})


def _bot(ref, mode, position=0, slug="fixture-one"):
    return Assignment("bot", slug, ref, mode, position)


def _tag(tag, ref, mode, position=0):
    return Assignment("tag", tag, ref, mode, position)


def _by_ref(resolution):
    return {e.ref: e for e in resolution.entries}


def test_a_bot_with_no_rows_keeps_harness_discovery():
    # Default is every skill enabled: nothing to resolve, no allow-list, so
    # skills added to the tree later reach the bot without a row.
    got = resolve(_facts(), [], _Source({"a": 10}), CAPS)
    assert got.entries == () and got.manifest() is None


def test_bot_row_beats_tag_rows_and_disable_beats_inject_beats_enable_across_tags():
    rows = [_tag("coder", "a", "enable"), _tag("reviewer", "a", "inject"),
            _tag("coder", "b", "inject"), _tag("reviewer", "b", "disable"),
            _tag("coder", "c", "disable"), _bot("c", "enable"),
            _tag("coder", "d", "inject"), _bot("d", "disable"),
            _tag("other-tag", "e", "disable"), _bot("f", "disable", slug="fixture-two")]
    got = _by_ref(resolve(_facts(), rows, _Source({r: 10 for r in "abcdef"}), CAPS))
    assert sorted(got) == ["a", "b", "c", "d"]
    assert (got["a"].mode, got["a"].origin, got["a"].applied) == ("inject", "tag:reviewer", True)
    assert (got["b"].mode, got["b"].origin, got["b"].applied) == ("disable", "tag:reviewer", True)
    # A bot enable over a tag disable is the deliberate override.
    assert (got["c"].mode, got["c"].origin, got["c"].reason) == ("enable", "bot", None)
    assert (got["d"].mode, got["d"].origin, got["d"].applied) == ("disable", "bot", True)


def test_an_enable_with_no_disable_under_it_is_marked_no_effect():
    rows = [_bot("a", "enable"), _tag("coder", "b", "enable")]
    got = _by_ref(resolve(_facts(), rows, _Source({"a": 1, "b": 1}), CAPS))
    assert {r: (e.applied, e.reason) for r, e in got.items()} == {
        "a": (True, NO_EFFECT), "b": (True, NO_EFFECT)}


def _global(ref, mode, position=0):
    return Assignment("global", "*", ref, mode, position)


def test_the_global_row_reaches_every_bot_below_bot_and_tag_rows():
    rows = [_global("a", "disable"), _global("b", "disable"), _tag("coder", "b", "enable"),
            _global("c", "disable"), _bot("c", "enable"), _global("d", "inject"),
            _tag("coder", "d", "disable"), _global("e", "disable"), _bot("e", "disable"),
            Assignment("global", "elsewhere", "f", "disable")]
    source = _Source({r: 10 for r in "abcdef"})
    got = _by_ref(resolve(_facts(), rows, source, CAPS))
    assert {r: (e.mode, e.origin, e.applied, e.reason) for r, e in got.items()} == {
        "a": ("disable", "global", True, None),
        # An enable over a global disable is an override, from a tag or a bot.
        "b": ("enable", "tag:coder", True, None),
        "c": ("enable", "bot", True, None),
        "d": ("disable", "tag:coder", True, None),
        "e": ("disable", "bot", True, None),
    }
    # A bot with none of those tags still gets every global row ("f" has no
    # valid global scope_id, so it applies nowhere).
    plain = _by_ref(resolve(_facts(slug="fixture-two", tags=frozenset()), rows, source, CAPS))
    assert {r: (e.mode, e.origin) for r, e in plain.items()} == {
        "a": ("disable", "global"), "b": ("disable", "global"), "c": ("disable", "global"),
        "d": ("inject", "global"), "e": ("disable", "global")}
    # Private bots and unsupported harnesses still get nothing from it.
    assert not resolve(_facts(private=True), rows, source, CAPS).applied
    assert resolve(_facts(harness="local"), rows, source, CAPS).manifest() is None


def test_a_global_enable_over_nothing_is_no_effect():
    got = _by_ref(resolve(_facts(), [_global("a", "enable")], _Source({"a": 1}), CAPS))
    assert (got["a"].applied, got["a"].reason) == (True, NO_EFFECT)


def test_legacy_rows_keep_global_rows_and_change_nothing_without_them():
    from llm_bawt.skills.resolver import legacy_rows

    facts = [_facts(), _facts(slug="fixture-two", tags=frozenset())]
    legacy = [_tag("coder", "a", "injected"), _bot("b", "off")]
    converted = legacy_rows(legacy, frozenset({"fixture-one"}), facts, ("a", "b", "c"))
    assert all(r.scope_type != "global" for r in converted)
    assert legacy_rows([*legacy, _global("c", "disable")], frozenset({"fixture-one"}), facts,
                       ("a", "b", "c")) == [*converted[:1], _global("c", "disable"), *converted[1:]]


def test_legacy_mode_names_resolve_like_the_new_ones():
    assert [canonical_mode(m) for m in ("available", "off", "injected", "enable", "bogus")] == [
        "enable", "disable", "inject", "enable", None]
    source = _Source({"a": 10, "b": 10, "c": 10})
    legacy = resolve(_facts(), [_bot("a", "inject"), _bot("b", "off")], source, CAPS)
    current = resolve(_facts(), [_bot("a", "inject"), _bot("b", "disable")], source, CAPS)
    assert legacy.manifest() == current.manifest()
    assert resolve(_facts(), [_bot("a", "bogus")], source, CAPS).entries == ()


def test_blanket_exclusions_keep_rows_visible_with_a_reason():
    rows = [_bot("a", "inject"), _bot("b", "disable")]
    source = _Source({"a": 10, "b": 10})
    cases = [(_facts(private=True), PROTECTED_BOT), (_facts(harness="local"), UNSUPPORTED_HARNESS),
             (_facts(harness=None), UNSUPPORTED_HARNESS), (_facts(bot_type="chat"), UNSUPPORTED_HARNESS),
             (_facts(bundle="pkg"), NOT_IN_BUNDLE)]
    for facts, reason in cases:
        got = resolve(facts, rows, source, CAPS)
        assert [(e.ref, e.applied, e.reason) for e in got.entries] == [
            ("a", False, reason), ("b", False, reason)], facts
        assert got.applied == () and got.injected == () and got.disabled == frozenset()
        # Loopy r1 B4: such a bot keeps its own skills; no selection reaches a harness.
        assert got.skip_reason == reason and got.manifest() is None, facts


def test_private_wins_over_every_other_exclusion():
    got = resolve(_facts(private=True, harness="local", bundle="pkg"), [_bot("a", "inject")],
                  _Source({"a": 10}), CAPS)
    assert got.entries[0].reason == PROTECTED_BOT


def test_missing_skills_are_reported_not_applied():
    rows = [_bot("gone", "enable"), _bot("unreadable", "inject"), _bot("absent", "inject"),
            _bot("nowhere", "disable")]
    got = _by_ref(resolve(_facts(), rows, _Source({"unreadable": 5}, ("unreadable",)), CAPS))
    assert {ref: (e.applied, e.reason) for ref, e in got.items()} == {
        "gone": (False, SOURCE_MISSING), "unreadable": (False, SOURCE_MISSING),
        "absent": (False, SOURCE_MISSING), "nowhere": (False, SOURCE_MISSING)}


def test_an_unreadable_catalog_applies_no_disable_and_sends_no_allow_list():
    got = resolve(_facts(), [_bot("a", "disable"), _bot("b", "inject")], _Source({}), CAPS)
    assert _by_ref(got)["a"].reason == CATALOG_UNREADABLE
    assert got.catalog_unreadable and got.disabled == frozenset()
    assert got.manifest() is None  # hiding every skill is the worse failure


def test_injected_skills_are_admitted_in_position_order_under_both_caps():
    sizes = {"first": 90, "second": 70, "third": 40, "huge": 101, "small": 10}
    rows = [_bot("small", "inject", 4), _bot("third", "inject", 2),
            _bot("second", "inject", 1), _bot("first", "inject", 0),
            _bot("huge", "inject", 3)]
    got = resolve(_facts(), rows, _Source(sizes), CAPS)
    assert [e.ref for e in got.entries] == ["first", "second", "third", "huge", "small"]
    assert [e.ref for e in got.injected] == ["first", "third", "small"]
    assert got.injected_tokens == 140
    assert got.over_cap == {"second", "huge"}
    skipped = _by_ref(got)["second"]
    assert (skipped.applied, skipped.reason, skipped.tokens) == (False, OVER_CAP, 70)
    assert skipped.body_text is None and skipped.base_dir is None
    admitted = _by_ref(got)["first"]
    assert admitted.body_text == "body of first" and admitted.base_dir == "/skills/first"


def test_equal_positions_order_by_ref():
    rows = [_bot("b", "inject"), _bot("a", "inject")]
    got = resolve(_facts(), rows, _Source({"a": 1, "b": 1}), CAPS)
    assert [e.ref for e in got.injected] == ["a", "b"]


def test_inject_without_a_disable_renders_bodies_but_keeps_discovery():
    got = resolve(_facts(), [_bot("a", "inject")], _Source({"a": 10, "b": 10}), CAPS)
    assert [e.ref for e in got.injected] == ["a"] and got.manifest() is None


def test_the_allow_list_is_the_catalog_minus_disables_and_is_stable():
    rows = [_bot("a", "inject"), _bot("b", "enable"), _bot("c", "disable")]
    source = _Source({"a": 10, "b": 10, "c": 10, "d": 10})
    first = resolve(_facts(), rows, source, CAPS).manifest()
    again = resolve(_facts(), list(reversed(rows)), source, CAPS).manifest()
    assert first == again
    # "d" has no row and is still allowed: the catalog, not the rows, is the base.
    assert first["entries"] == [{"ref": "a", "mode": "inject", "source": "fake"},
                                {"ref": "b", "mode": "enable", "source": "fake"},
                                {"ref": "d", "mode": "enable", "source": "fake"}]
    assert first["injected"] == [{"ref": "a", "content_sha": "sha-a-10"}]
    edited = resolve(_facts(), rows, _Source({"a": 11, "b": 10, "c": 10, "d": 10}), CAPS).manifest()
    added = resolve(_facts(), rows, _Source({"a": 10, "b": 10, "c": 10, "d": 10, "e": 1}),
                    CAPS).manifest()
    demoted = resolve(_facts(), [_bot("a", "enable"), *rows[1:]], source, CAPS).manifest()
    assert {"ref": "e", "mode": "enable", "source": "fake"} in added["entries"]
    assert len({first["manifest_sha"], edited["manifest_sha"], added["manifest_sha"],
                demoted["manifest_sha"]}) == 4


def test_newly_over_cap_ignores_skills_already_over():
    source = _Source({"a": 120, "b": 60, "c": 60})
    before = resolve(_facts(), [_bot("a", "inject"), _bot("b", "inject", 1)], source, CAPS)
    after = resolve(_facts(), [_bot("a", "inject"), _bot("b", "inject", 1),
                               _bot("c", "inject", 2)], source, CAPS)
    assert before.over_cap == {"a"}
    assert newly_over_cap(before, after) == set()
    tighter = resolve(_facts(), [_bot("a", "inject"), _bot("b", "inject", 1),
                                 _bot("c", "inject", 2)], source, Caps(100, 100))
    assert newly_over_cap(before, tighter) == {"c"}


def test_harness_capabilities_fall_back_to_unsupported():
    assert capabilities_for("claude-code").supported
    assert capabilities_for("codex").injection == "until_first_compaction"
    assert capabilities_for("codex").reinject_on_resume == "unverified"
    assert capabilities_for("local").supported is False
    unknown = capabilities_for("something-new")
    assert (unknown.harness, unknown.supported, unknown.injection) == (
        "something-new", False, "not_applied")
    assert capabilities_for(None).harness == "chat"


def test_render_section_substitutes_once_and_keeps_braces():
    body = "Use {ref} and {body} literally; also {unknown} and {{x}}.\n"
    rendered = render_section("## Pre", "### {ref} @ {base_dir}\n{body}",
                              [("one", "/s/one", body), ("two", "/s/two", "plain")])
    assert rendered == ("## Pre\n\n### one @ /s/one\n"
                        "Use {ref} and {body} literally; also {unknown} and {{x}}."
                        "\n\n### two @ /s/two\nplain")


def test_render_section_renders_nothing_without_skills_or_a_body_slot():
    assert render_section("## Pre", "{body}", []) == ""
    assert render_section("## Pre", "### {ref}", [("one", "/s", "b")]) == ""
    assert render_section("  ", "{body}", [("one", "/s", "b")]) == "b"


def test_the_app_catalog_and_the_bridge_discover_the_same_tree(monkeypatch, tmp_path):
    # A disable is enforced as an allow-list of catalog refs; if the app read a
    # different tree than the bridge's CLI discovery, the list would hide or leak skills.
    from agent_bridge.skill_selection import claude_skill_names, legacy_claude_skill_roots
    from agent_bridge.skill_sources import AmbientSource

    first, second = tmp_path / "first", tmp_path / "second"
    for root, names in ((first, ("alpha", "beta")), (second, ("beta", "gamma"))):
        for name in names:
            (root / name).mkdir(parents=True)
            (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n")
    (first / "not-a-skill").mkdir()
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", f"{first}:{second}")

    source = AmbientSource()
    assert source.roots == legacy_claude_skill_roots() == (first, second)
    assert tuple(sorted(e.ref for e in source.list())) == claude_skill_names(None) == (
        "alpha", "beta", "gamma")


def test_a_partly_unreadable_catalog_applies_no_disable(tmp_path):
    # One readable root and one missing root: the listing is partial, and an
    # allow-list built from it would hide the missing root's skills.
    from agent_bridge.skill_sources import AmbientSource

    root = tmp_path / "mounted"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n")
    source = AmbientSource((root, tmp_path / "not-mounted"))
    assert [e.ref for e in source.list()] == ["alpha"] and source.complete() is False

    resolution = resolve(_facts(), [Assignment("bot", "fixture-one", "alpha", "disable", 0)],
                         source, Caps(per_skill=100, total=100))

    assert resolution.catalog_unreadable is True
    assert resolution.manifest() is None
