"""A managed skill selection reaches both harnesses (TASK-1309, Loopy r1 B4).

The app pins the resolver's manifest with the session's prompt; the
publisher validates it onto the command; Claude gets an allow-list and
Codex a deny-list of everything else it would discover. Hermetic.
"""
from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_bridge.skill_codex import codex_config_overrides, codex_skills
from agent_bridge.skill_selection import (
    CLAUDE_DEFAULT_SETTING_SOURCES,
    SELECTION_FIELD,
    ManagedSelection,
    SkillSelectionError,
    claude_skill_names,
    claude_skill_options,
)
from llm_bawt.models.message import Message
from llm_bawt.service import system_prompt_pin as pin_mod
from llm_bawt.service.system_prompt_pin import PinDecision, apply_session_pin, pin_backend_kwargs

MANIFEST = {"harness": "claude-code", "manifest_sha": "a" * 64, "injected": [],
            "entries": [{"ref": "alpha", "mode": "available", "source": "ambient"},
                        {"ref": "gamma", "mode": "injected", "source": "ambient"},
                        {"ref": "alpha", "mode": "available", "source": "ambient"}]}


# ── the selection ───────────────────────────────────────────────────────────

def test_selection_round_trips_through_the_command_field():
    selection = ManagedSelection.from_manifest(MANIFEST)
    assert selection.refs == ("alpha", "gamma") and selection.manifest_sha == "a" * 64
    assert ManagedSelection.from_field(selection.to_field()) == selection
    assert ManagedSelection.from_field(None) is None and ManagedSelection.from_field("") is None


@pytest.mark.parametrize("raw", [
    "not json", "[]", json.dumps({"entries": []}), json.dumps({"manifest_sha": "x"}),
    json.dumps({"entries": [{"ref": "../escape"}], "manifest_sha": "x"}),
    json.dumps({"entries": ["alpha"], "manifest_sha": "x"}),
])
def test_a_malformed_selection_fails_closed(raw):
    with pytest.raises(SkillSelectionError):
        ManagedSelection.from_field(raw)


def test_claude_gets_an_allow_list_and_keeps_its_settings_sources():
    selection = ManagedSelection.from_manifest(MANIFEST)
    assert claude_skill_options(None, selection) == {
        "skills": ["alpha", "gamma"], "setting_sources": list(CLAUDE_DEFAULT_SETTING_SOURCES)}
    assert "local" in CLAUDE_DEFAULT_SETTING_SOURCES
    assert claude_skill_names(None, selection) == ("alpha", "gamma")
    empty = ManagedSelection.from_manifest({**MANIFEST, "entries": []})
    assert claude_skill_options(None, empty)["skills"] == []  # managed, nothing chosen: hide all
    with pytest.raises(SkillSelectionError, match="exclusive"):
        claude_skill_options("tenant", selection)


# ── Codex deny-list ─────────────────────────────────────────────────────────

def _skill(directory: Path, name: str | None = None) -> None:
    directory.mkdir(parents=True)
    front = f"---\nname: {json.dumps(name)}\n---\n" if name else ""
    (directory / "SKILL.md").write_text(f"{front}body\n", encoding="utf-8")


@pytest.fixture
def codex_env(tmp_path):
    home, codex_home = tmp_path / "home", tmp_path / "codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        '[marketplaces.loc]\nsource_type = "local"\n'
        '[marketplaces.remote]\nsource_type = "git"\n')
    _skill(codex_home / "skills" / "alpha", "alpha-skill")       # user skill, frontmatter name
    _skill(codex_home / "skills" / "quoted", 'say "hi" \\ back')  # needs TOML escaping
    _skill(codex_home / "skills" / ".system")                     # hidden: never discovered
    (codex_home / "skills" / "no-skill-file").mkdir()
    _skill(home / ".agents" / "skills" / "beta")                  # user skill, dir name
    _skill(home / ".agents" / "skills" / "ünï")                   # non-ASCII name
    cache = codex_home / "plugins" / "cache"
    _skill(cache / "loc" / "toolkit" / "1.0" / "skills" / "gamma", "gamma")
    _skill(cache / "remote" / "ext" / "2.0" / "skills" / "alpha", "alpha")  # same ref, not local
    _skill(cache / "remote" / "ext" / "2.0" / "skills" / "delta")
    return {"HOME": str(home), "CODEX_HOME": str(codex_home)}


def test_codex_discovers_user_and_plugin_skills_by_their_codex_names(codex_env):
    assert codex_skills(codex_env) == [
        ("alpha-skill", "alpha", True), ("beta", "beta", True), ("ext:alpha", "alpha", False),
        ("ext:delta", "delta", False), ('say "hi" \\ back', "quoted", True),
        ("toolkit:gamma", "gamma", True), ("ünï", "ünï", True)]


def _parsed(overrides: list[str]) -> dict:
    return tomllib.loads("\n".join(overrides))["skills"]


def test_codex_denies_exactly_the_discovered_complement_of_the_selection(codex_env):
    selection = ManagedSelection.from_manifest(MANIFEST)  # alpha + gamma
    overrides = codex_config_overrides(selection, env=codex_env)
    parsed = _parsed(overrides)
    assert parsed["bundled"] == {"enabled": False}
    assert all(item["enabled"] is False for item in parsed["config"])
    denied = {item["name"] for item in parsed["config"]}
    discovered = {name for name, _ref, _selectable in codex_skills(codex_env)}
    allowed = {name for name, ref, selectable in codex_skills(codex_env)
               if selectable and ref in selection.refs}
    assert allowed == {"alpha-skill", "toolkit:gamma"}
    # Al's drift contract (TASK-1243): denied == discovered - allowed, nothing else.
    assert denied == discovered - allowed
    assert "ext:alpha" in denied  # a non-local plugin is never selectable by an ambient ref


def test_codex_unmanaged_gets_no_overrides_and_full_selection_denies_only_bundled(codex_env):
    assert codex_config_overrides(None, env=codex_env) == []
    every = ManagedSelection.from_manifest({**MANIFEST, "entries": [
        {"ref": ref} for ref in ("alpha", "beta", "quoted", "gamma", "delta")]})
    only_unselectable = _parsed(codex_config_overrides(every, env=codex_env))
    assert {i["name"] for i in only_unselectable["config"]} == {"ext:alpha", "ext:delta", "ünï"}
    empty_home = {"HOME": codex_env["HOME"] + "-none", "CODEX_HOME": codex_env["CODEX_HOME"] + "-none"}
    assert codex_config_overrides(every, env=empty_home) == ["skills.bundled.enabled=false"]


def test_codex_overrides_go_before_resume_on_the_real_sdk_command():
    from openai_codex_sdk.exec import CodexExec, CodexExecArgs

    from codex_bridge.exec_patch import with_config_overrides

    args = CodexExec(executable_path="codex", env={})._build_command_args(
        CodexExecArgs(input="x", model="m", thread_id="thread-1"))
    out = with_config_overrides(args, ("skills.bundled.enabled=false", "a=1"))
    assert out[:6] == ["exec", "--experimental-json", "--config", "skills.bundled.enabled=false",
                       "--config", "a=1"]
    assert out[6:] == args[2:] and out.index("resume") > 5
    assert with_config_overrides(args, ()) is args
    with pytest.raises(RuntimeError, match="unexpected codex command"):
        with_config_overrides(["resume", "x"], ("a=1",))


# ── app → command ───────────────────────────────────────────────────────────

def _publish(**kwargs):
    from agent_bridge.subscriber import RedisSubscriber

    subscriber = RedisSubscriber("redis://localhost:6379/0")
    captured: dict = {}

    async def fake_xadd(_stream, fields, **_kwargs):
        captured.update(fields)
        return b"1"

    subscriber._pub_redis = SimpleNamespace(xadd=fake_xadd)
    asyncio.run(subscriber.send_command(session_key="s", message="hi", request_id="req-t", **kwargs))
    return captured


def test_publisher_validates_the_selection_onto_the_command():
    fields = _publish(skill_selection=MANIFEST)
    assert ManagedSelection.from_field(fields[SELECTION_FIELD]) == ManagedSelection.from_manifest(MANIFEST)
    assert SELECTION_FIELD not in _publish()
    with pytest.raises(SkillSelectionError):
        _publish(skill_selection={"entries": "nope"})
    with pytest.raises(SkillSelectionError, match="exclusive"):
        _publish(skill_selection=MANIFEST, skill_bundle="tenant")


def test_claude_bridge_reads_the_selection_field():
    from claude_code_bridge.send_request import SendRequest

    raw = ManagedSelection.from_manifest(MANIFEST).to_field()
    assert SendRequest.from_fields({"session_key": "b:s", SELECTION_FIELD: raw}).skill_selection == raw
    assert SendRequest.from_fields({"session_key": "b:s"}).skill_selection is None


def test_injected_skills_ride_their_own_command_field():
    # TASK-1691: independent of the selection, so an inject-only bot names them too.
    from agent_bridge.skill_selection import INJECTED_FIELD
    from claude_code_bridge.send_request import SendRequest

    fields = _publish(injected_skills=["git-commits", "peer-review"])
    assert fields[INJECTED_FIELD] == "git-commits,peer-review" and SELECTION_FIELD not in fields
    assert INJECTED_FIELD not in _publish(injected_skills=[])
    with pytest.raises(SkillSelectionError):
        _publish(injected_skills=["../x"])
    assert SendRequest.from_fields(
        {"session_key": "b:s", INJECTED_FIELD: "git-commits"}).injected_skills == "git-commits"


def test_client_forwards_the_selection_into_the_backend_config():
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    backend = ClaudeCodeBackend()
    client = object.__new__(AgentBackendClient)
    client._backend_name, client._backend, client.config = backend.name, backend, None
    client._bot_config = {"bot_id": "b", "model": "m"}
    seen: list[dict] = []
    backend.stream_raw = lambda prompt, config, **kw: seen.append(config) or iter(())
    with patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", return_value=0):
        list(client.stream_raw([Message("user", "hi")], skill_selection=MANIFEST,
                               injected_skills=["git-commits"]))
        list(client.stream_raw([Message("user", "hi")]))
    assert seen[0]["skill_selection"] == MANIFEST and "skill_selection" not in seen[1]
    assert seen[0]["injected_skills"] == ["git-commits"] and "injected_skills" not in seen[1]
    assert "skill_selection" not in client._bot_config  # request-local, never shared


# ── session pin ─────────────────────────────────────────────────────────────

def _resolution(manifest):
    return SimpleNamespace(manifest=lambda: manifest)


def _ctx(resolution=None):
    return SimpleNamespace(
        svc=SimpleNamespace(_turn_log_store=SimpleNamespace(engine=object())),
        bot_id="fixture-one", user_prompt="hi", bridge_reset_action=None,
        thread_binding={"thread_session_id": "sess", "thread_resume_id": "resume"},
        llm_bawt=SimpleNamespace(_skill_resolution=resolution),
    )


def _decision(manifest):
    return PinDecision(text="P", pin_turn_id="t", sha256="s", action="kept", reason=None,
                       is_boundary=False, rendered_sha256="s", skill_manifest=manifest)


def test_a_pinned_turn_sends_the_pinned_selection_not_the_current_one():
    pinned = {**MANIFEST, "manifest_sha": "b" * 64}
    ctx = _ctx(_resolution(MANIFEST))
    decision = _decision(pinned)
    assert pin_backend_kwargs(ctx, decision, agent=True) == {
        "system_prompt_pin": decision, "skill_selection": pinned}
    # A session pinned unmanaged stays unmanaged until its next boundary.
    assert pin_backend_kwargs(ctx, _decision(None), agent=True) == {
        "system_prompt_pin": _decision(None)}


def test_an_unpinned_turn_sends_this_render_and_chat_turns_send_nothing():
    assert pin_backend_kwargs(_ctx(_resolution(MANIFEST)), None, agent=True) == {
        "skill_selection": MANIFEST}
    assert pin_backend_kwargs(_ctx(_resolution(None)), None, agent=True) == {}
    assert pin_backend_kwargs(_ctx(None), None, agent=True) == {}
    assert pin_backend_kwargs(_ctx(_resolution(MANIFEST)), None, agent=False) == {}


class _Client:
    _backend = SimpleNamespace(pins_system_prompt=True)

    def compose_system_prompt(self, base):
        return base


def test_injected_skills_follow_the_held_prompt_not_the_manifest():
    # TASK-1691: an inject-only bot has no manifest but still names its skills.
    ctx = _ctx(_resolution(None))
    ctx.llm_bawt._injected_skill_refs = ("git-commits",)
    assert pin_backend_kwargs(ctx, None, agent=True) == {"injected_skills": ["git-commits"]}
    held = PinDecision(text="P", pin_turn_id="t", sha256="s", action="kept", reason=None,
                       is_boundary=False, rendered_sha256="s", injected_skills=("peer-review",))
    assert pin_backend_kwargs(ctx, held, agent=True)["injected_skills"] == ["peer-review"]
    # The harness may still hold older bytes: claim nothing.
    unverified = PinDecision(text="P", pin_turn_id="t", sha256="s", action="pinned",
                             reason="missing_row", is_boundary=True, rendered_sha256="s",
                             snapshot_unverified=True, injected_skills=("peer-review",))
    assert "injected_skills" not in pin_backend_kwargs(ctx, unverified, agent=True)


def test_the_hook_pins_this_renders_manifest(monkeypatch):
    captured = {}
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve",
                        lambda self, **kw: captured.update(kw))
    apply_session_pin(_ctx(_resolution(MANIFEST)), _Client(), [Message("system", "x")], turn_id="t")
    assert captured["skill_manifest"] == MANIFEST
    assert captured["injected_skills"] == ()


def test_a_corrupt_pinned_selection_fails_the_turn_instead_of_using_the_current(monkeypatch):
    def corrupt(self, **kwargs):
        raise SkillSelectionError("skill selection needs entries and a manifest_sha")

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve", corrupt)
    with pytest.raises(SkillSelectionError):
        apply_session_pin(_ctx(_resolution(MANIFEST)), _Client(), [Message("system", "x")], turn_id="t")
