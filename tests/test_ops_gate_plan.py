"""The first gate plan is closed and inert; it grants no dispatch authority."""
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from llm_bawt.ops.gate_plan import GATE_PLAN_SCHEMA, GatePlan
from llm_bawt.ops.gate_receipt import (
    DEFAULT_GATE_RECEIPT, GATE_RECEIPT_KEY, gate_callback, gate_receipt_prompt,
)

SHA = "1234567890abcdef" * 2 + "12345678"


def test_plan_roundtrip_and_immutability():
    plan = GatePlan(repository="llm-bawt", source_sha=SHA)
    assert plan.to_dict() == {"schema": GATE_PLAN_SCHEMA, "repository": "llm-bawt",
                              "source_sha": SHA, "target": "test-all"}
    assert GatePlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(FrozenInstanceError):
        plan.source_sha = "a" * 40


@pytest.mark.parametrize("sha", [None, True, 123, "", "HEAD", "origin/master", "a" * 39,
                                "a" * 41, "A" * 40, "g" * 40, " " + SHA, SHA + "\n"])
def test_exact_lowercase_sha_only(sha):
    with pytest.raises(ValueError, match="source_sha"):
        GatePlan(repository="llm-bawt", source_sha=sha)


@pytest.mark.parametrize("repo", ["bawthub", "agent-skills", "../llm-bawt", "/repo", True, None])
def test_unsupported_repos_fail_closed(repo):
    with pytest.raises(ValueError, match="unsupported gate repository"):
        GatePlan(repository=repo, source_sha=SHA)


@pytest.mark.parametrize("target", ["gate", "test-target", "test", "test-all;echo bad", None, True])
def test_no_custom_targets(target):
    with pytest.raises(ValueError, match="fixed"):
        GatePlan(repository="llm-bawt", source_sha=SHA, target=target)


@pytest.mark.parametrize("extra", ["argv", "command", "path", "environment", "mounts", "patches", "attempt"])
def test_no_extra_fields_or_caller_retry_attempt(extra):
    with pytest.raises(ValueError, match="exactly"):
        GatePlan.from_dict({**GatePlan("llm-bawt", SHA).to_dict(), extra: "untrusted"})


@pytest.mark.parametrize("value", [None, [], "{}", {}, {"schema": 1},
    {"schema": GATE_PLAN_SCHEMA, "repository": "llm-bawt", "source_sha": SHA}])
def test_decoder_never_coerces_or_defaults(value):
    with pytest.raises(ValueError):
        GatePlan.from_dict(value)


def test_unknown_schema_is_refused():
    with pytest.raises(ValueError, match="schema"):
        GatePlan.from_dict({**GatePlan("llm-bawt", SHA).to_dict(), "schema": "v2"})


@pytest.mark.parametrize("value", [None, "", " ", True, "a" * 257])
@pytest.mark.parametrize("field", ["bot_id", "user_id", "session_id"])
def test_callback_needs_identity_and_a_session(field, value):
    destination = {"bot_id": "fixture-reviewer", "user_id": "fixture-user", "session_id": "fixture-session"}
    destination[field] = value
    with pytest.raises(ValueError, match="identity and session"):
        gate_callback(GatePlan("llm-bawt", SHA), **destination)


def test_gate_framing_is_registered_and_live_with_seed_fallback(monkeypatch):
    from llm_bawt import prompt_registry
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    definition = DEFAULT_PROMPT_DEFINITIONS[GATE_RECEIPT_KEY]
    assert definition.default_body() == DEFAULT_GATE_RECEIPT and definition.required_vars == ()
    seen = []
    resolved = SimpleNamespace(body="custom gate wake")
    monkeypatch.setattr(prompt_registry, "get_prompt_resolver", lambda: SimpleNamespace(
        resolve=lambda key, **kw: seen.append((key, kw)) or resolved))
    assert gate_receipt_prompt("fixture-reviewer") == "custom gate wake"
    assert seen == [(GATE_RECEIPT_KEY, {"scope_type": "bot", "scope_id": "fixture-reviewer"})]
    resolved.body = " "
    assert gate_receipt_prompt("fixture-reviewer") == DEFAULT_GATE_RECEIPT
    monkeypatch.setattr(prompt_registry, "get_prompt_resolver", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert gate_receipt_prompt("fixture-reviewer") == DEFAULT_GATE_RECEIPT


def test_gate_plan_is_not_registered_as_a_docker_action():
    from llm_bawt.ops.executor import validate_spec

    with pytest.raises(ValueError):
        validate_spec(json.dumps({"action": "batch_gate", **GatePlan("llm-bawt", SHA).to_dict()}))
