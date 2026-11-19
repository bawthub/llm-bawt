"""TASK-893 slice 6: approved guidance in the prompt, rendered per turn."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_bawt.core.prompt_builder import PromptBuilder, SectionPosition
from llm_bawt.core.prompt_manifest import PER_TURN, PROMPT_MANIFEST
from llm_bawt.core.prompt_sections import PromptSectionsMixin
from llm_bawt.memory import guidance_feed as gf
from llm_bawt.setting_definitions import SETTING_DEFINITIONS


def _pref(pid, statement, status="active", **kw):
    return {"id": pid, "statement": statement, "status": status, "evidence": [], **kw}


def _version(*prefs, mandate="", row=7, version=3):
    body = json.dumps({"v": 1, "mandate": mandate, "preferences": list(prefs)})
    return {"row_id": row, "version": version, "body": body, "sha256": f"sha-{row}"}


# -- pure renderer -----------------------------------------------------------

def test_mandate_and_active_only_in_document_order():
    v = _version(_pref("a", "short answers", status="superseded", superseded_by="b"),
                 _pref("b", "detailed answers", supersedes="a"),
                 _pref("c", "no emoji"), mandate="Be direct.")
    r = gf.render(v)
    assert r.body == (f"{gf.HEADING}\nBe direct.\n\n{gf.PREFS_HEADING}\n"
                      "- detailed answers\n- no emoji")
    assert "short answers" not in r.body
    assert r.provenance["version_row_id"] == 7 and r.provenance["source_sha256"] == "sha-7"
    assert r.provenance["dropped_count"] == 0 and r.provenance["active_count"] == 2
    assert r.provenance["rendered_sha256"] != r.provenance["source_sha256"]


def test_same_version_renders_identical_bytes():
    v = _version(_pref("a", "x"), mandate="m")
    assert gf.render(v).body == gf.render(v).body


def test_no_guidance_or_empty_doc_renders_nothing():
    assert gf.render(None).body is None
    assert gf.render(_version()).body is None


def test_malformed_statement_type_is_refused_not_coerced():
    v = _version(_pref("a", ["not", "text"]))
    with pytest.raises(ValueError):
        gf.render(v)


def test_oversized_mandate_omits_whole_section():
    v = _version(_pref("a", "x"), mandate="m" * 50)
    r = gf.render(v, max_chars=40)
    assert r.body is None and r.provenance["reason"] == "mandate_over_cap"


def test_whole_statements_dropped_from_the_end_never_truncated():
    v = _version(_pref("a", "first"), _pref("b", "y" * 500), _pref("c", "third"))
    r = gf.render(v, max_chars=len(gf.HEADING) + 200)
    assert r.body.endswith("- first") and "yyy" not in r.body and "third" not in r.body
    assert r.provenance["dropped_count"] == 2
    assert len(r.body) <= len(gf.HEADING) + 200


def test_cap_holds_for_default_limit():
    v = _version(*[_pref(str(i), "z" * 300) for i in range(30)], mandate="m")
    r = gf.render(v)
    assert len(r.body) <= gf.MAX_CHARS and r.provenance["dropped_count"] > 0


# -- declaration ---------------------------------------------------------------

def test_section_is_per_turn_for_all_bots_and_setting_defaults_off():
    spec = next(s for s in PROMPT_MANIFEST if s.name == "user_guidance")
    assert spec.stage == PER_TURN and set(spec.applies_to) == {"chat", "agent"}
    assert SETTING_DEFINITIONS[gf.SETTING].default is False


# -- reused owner: each turn rebuilds from a fresh copy -------------------------

class _Owner(PromptSectionsMixin):
    def __init__(self, monkeypatch):
        self.bot = SimpleNamespace(requires_memory=True)
        self.bot_id, self.user_id = "fixture-bot", "fixture-user"
        self.config = SimpleNamespace()
        self._db_available = True
        self.settings = {gf.SETTING: True}
        self.current = None
        self.fail = False
        stable = PromptBuilder()
        stable.add_section("user_context", "## About the User\nx",
                           position=SectionPosition.USER_CONTEXT)
        stable.add_section("bot_traits", "## Traits", position=SectionPosition.BOT_TRAITS)
        self._prompt_builder = stable  # built once, as at initialization

        def load(engine, *, bot_id, user_id, max_chars=gf.MAX_CHARS):
            if self.fail:
                raise RuntimeError("db down")
            if self.current == "protected":
                return gf.Rendered(None, {"reason": "protected_bot"})
            return gf.render(self.current, max_chars)
        monkeypatch.setattr(gf, "load", load)
        monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda cfg: object())

    def _resolve_setting(self, key, fallback=None):
        return self.settings.get(key, fallback)

    def turn(self):
        b = self._prompt_builder.copy()
        self._sec_user_guidance(b, "")
        return b


def _names(builder):
    return [s.name for s in builder.sections]


def test_reused_owner_sees_approval_on_next_turn(monkeypatch):
    o = _Owner(monkeypatch)
    o.current = _version(_pref("a", "short answers"), row=1)
    first = o.turn()
    assert "short answers" in first.build()
    assert _names(first)[:3] == ["user_context", "user_guidance", "bot_traits"]
    o.current = _version(_pref("a", "short answers", status="superseded", superseded_by="b"),
                         _pref("b", "detailed answers", supersedes="a"), row=2)
    nxt = o.turn()
    out = nxt.build()
    assert "detailed answers" in out and "short answers" not in out
    assert nxt._sections["user_guidance"].metadata["provenance"]["version_row_id"] == 2
    assert "user_guidance" not in o._prompt_builder._sections  # cache untouched


@pytest.mark.parametrize("change", ["disable", "protect", "fail", "no_memory", "string_true"])
def test_reused_owner_drops_section_when_gate_closes(monkeypatch, change):
    o = _Owner(monkeypatch)
    o.current = _version(_pref("a", "short answers"))
    assert "short answers" in o.turn().build()
    if change == "disable":
        o.settings[gf.SETTING] = False
    elif change == "protect":
        o.current = "protected"
    elif change == "fail":
        o.fail = True
    elif change == "no_memory":
        o.bot.requires_memory = False
    else:
        o.settings[gf.SETTING] = "true"  # only a real boolean enables it
    assert "short answers" not in o.turn().build()
