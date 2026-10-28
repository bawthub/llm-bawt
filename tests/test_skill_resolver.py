"""The skill resolver (TASK-1309): precedence, exclusions, caps and the manifest."""
from __future__ import annotations

from pathlib import Path

from agent_bridge.skill_sources import SkillBody, SkillEntry, SkillSource
from llm_bawt.skills.harness import capabilities_for
from llm_bawt.skills.prompts import render_section
from llm_bawt.skills.resolver import (
    NOT_IN_BUNDLE,
    OVER_CAP,
    PROTECTED_BOT,
    SOURCE_MISSING,
    UNSUPPORTED_HARNESS,
    Assignment,
    BotFacts,
    Caps,
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
                tags=frozenset({"coder", "reviewer"}), private=False, managed=True)
    return BotFacts(**{**base, **overrides})


def _bot(ref, mode, position=0, slug="fixture-one"):
    return Assignment("bot", slug, ref, mode, position)


def _tag(tag, ref, mode, position=0):
    return Assignment("tag", tag, ref, mode, position)


def _by_ref(resolution):
    return {e.ref: e for e in resolution.entries}


def test_unmanaged_bot_resolves_to_nothing_and_has_no_manifest():
    got = resolve(_facts(managed=False), [_bot("a", "injected")], _Source({"a": 10}), CAPS)
    assert got.entries == () and got.manifest() is None


def test_bot_row_beats_tag_rows_and_the_strongest_tag_mode_wins():
    rows = [_tag("coder", "a", "available"), _tag("reviewer", "a", "injected"),
            _tag("coder", "b", "injected"), _bot("b", "available"),
            _tag("coder", "c", "injected"), _bot("c", "off"),
            _tag("other-tag", "d", "injected"), _bot("e", "available", slug="fixture-two")]
    got = _by_ref(resolve(_facts(), rows, _Source({r: 10 for r in "abcde"}), CAPS))
    assert sorted(got) == ["a", "b", "c"]
    assert (got["a"].mode, got["a"].origin, got["a"].applied) == ("injected", "tag:reviewer", True)
    assert (got["b"].mode, got["b"].origin) == ("available", "bot")
    assert (got["c"].mode, got["c"].applied, got["c"].reason) == ("off", False, None)


def test_off_at_tag_scope_is_never_honoured():
    got = resolve(_facts(), [_tag("coder", "a", "off")], _Source({"a": 10}), CAPS)
    assert got.entries == ()


def test_blanket_exclusions_keep_rows_visible_with_a_reason():
    rows = [_bot("a", "injected"), _bot("b", "available")]
    source = _Source({"a": 10, "b": 10})
    cases = [(_facts(private=True), PROTECTED_BOT), (_facts(harness="local"), UNSUPPORTED_HARNESS),
             (_facts(harness=None), UNSUPPORTED_HARNESS), (_facts(bot_type="chat"), UNSUPPORTED_HARNESS),
             (_facts(bundle="pkg"), NOT_IN_BUNDLE)]
    for facts, reason in cases:
        got = resolve(facts, rows, source, CAPS)
        assert [(e.ref, e.applied, e.reason) for e in got.entries] == [
            ("a", False, reason), ("b", False, reason)], facts
        assert got.applied == () and got.injected == ()
        # Loopy r1 B4: such a bot keeps its own skills; no selection reaches a harness.
        assert got.skip_reason == reason and got.manifest() is None, facts


def test_private_wins_over_every_other_exclusion():
    got = resolve(_facts(private=True, harness="local", bundle="pkg"), [_bot("a", "injected")],
                  _Source({"a": 10}), CAPS)
    assert got.entries[0].reason == PROTECTED_BOT


def test_missing_skills_are_reported_not_applied():
    rows = [_bot("gone", "available"), _bot("unreadable", "injected"), _bot("absent", "injected")]
    got = _by_ref(resolve(_facts(), rows, _Source({"unreadable": 5}, ("unreadable",)), CAPS))
    assert {ref: (e.applied, e.reason) for ref, e in got.items()} == {
        "gone": (False, SOURCE_MISSING), "unreadable": (False, SOURCE_MISSING),
        "absent": (False, SOURCE_MISSING)}


def test_injected_skills_are_admitted_in_position_order_under_both_caps():
    sizes = {"first": 90, "second": 70, "third": 40, "huge": 101, "small": 10}
    rows = [_bot("small", "injected", 4), _bot("third", "injected", 2),
            _bot("second", "injected", 1), _bot("first", "injected", 0),
            _bot("huge", "injected", 3)]
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
    rows = [_bot("b", "injected"), _bot("a", "injected")]
    got = resolve(_facts(), rows, _Source({"a": 1, "b": 1}), CAPS)
    assert [e.ref for e in got.injected] == ["a", "b"]


def test_manifest_is_stable_and_tracks_content_and_modes():
    rows = [_bot("a", "injected"), _bot("b", "available"), _bot("c", "off")]
    source = _Source({"a": 10, "b": 10, "c": 10})
    first = resolve(_facts(), rows, source, CAPS).manifest()
    again = resolve(_facts(), list(reversed(rows)), source, CAPS).manifest()
    assert first == again
    assert first["entries"] == [{"ref": "a", "mode": "injected", "source": "fake"},
                                {"ref": "b", "mode": "available", "source": "fake"}]
    assert first["injected"] == [{"ref": "a", "content_sha": "sha-a-10"}]
    edited = resolve(_facts(), rows, _Source({"a": 11, "b": 10, "c": 10}), CAPS).manifest()
    demoted = resolve(_facts(), [_bot("a", "available"), *rows[1:]], source, CAPS).manifest()
    assert len({first["manifest_sha"], edited["manifest_sha"], demoted["manifest_sha"]}) == 3


def test_managed_bot_with_no_rows_has_an_empty_manifest():
    manifest = resolve(_facts(), [], _Source({}), CAPS).manifest()
    assert manifest["entries"] == [] and manifest["injected"] == []
    assert len(manifest["manifest_sha"]) == 64


def test_newly_over_cap_ignores_skills_already_over():
    source = _Source({"a": 120, "b": 60, "c": 60})
    before = resolve(_facts(), [_bot("a", "injected"), _bot("b", "injected", 1)], source, CAPS)
    after = resolve(_facts(), [_bot("a", "injected"), _bot("b", "injected", 1),
                               _bot("c", "injected", 2)], source, CAPS)
    assert before.over_cap == {"a"}
    assert newly_over_cap(before, after) == set()
    tighter = resolve(_facts(), [_bot("a", "injected"), _bot("b", "injected", 1),
                                 _bot("c", "injected", 2)], source, Caps(100, 100))
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
