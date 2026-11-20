"""Utility-model routing of imported Google tasks (TASK-905); the model is stubbed."""
import json

from llm_bawt.integrations import google_tasks_route as route
from llm_bawt.integrations.google_tasks_route import DEFAULT_SYSTEM, MIN_CONFIDENCE, PROMPT_KEY, decide

BOTS = [{"id": "juno", "name": "Juno", "description": "Claude coder"},
        {"id": "pixel", "name": "Pixel", "description": "Assistant with memory"}]
PROJECTS = ["Timeline", "Home"]


def model(answer, seen=None):
    def complete(prompt, system):
        if seen is not None:
            seen.append((json.loads(prompt), system))
        return answer if isinstance(answer, str) else json.dumps(answer)
    return complete


def fixed(text="routing instructions"):
    return lambda: text


def item(title, notes=""):
    return {"linkId": "l1", "title": title, "notes": notes}


def run(title, answer, seen=None, system=None):
    return decide(item(title), BOTS, PROJECTS, complete=model(answer, seen), system=system or fixed())


def test_misheard_name_routes():
    seen = []
    out = run("June fix the timeline scroll",
              {"bot_id": "juno", "project": "timeline", "title": "fix the timeline scroll", "confidence": 0.9,
               "reason": "'June' is a mishearing of Juno"}, seen)
    assert out == {"linkId": "l1", "botId": "juno", "project": "Timeline",
                   "reason": "'June' is a mishearing of Juno"}
    prompt, system = seen[0]
    assert prompt["bots"] == BOTS and prompt["projects"] == PROJECTS
    assert prompt["item"]["title"] == "June fix the timeline scroll"
    assert system == "routing instructions"
    assert "misheard" in DEFAULT_SYSTEM


def test_routing_never_rewrites_the_title():
    for raw in ("Pickle remind me about the vet", "Vera never publish the draft", "Nora no refunds"):
        out = run(raw, {"bot_id": "juno", "project": None, "title": "publish the draft", "confidence": 0.9, "reason": "r"})
        assert out["botId"] == "juno" and "title" not in out, raw


def test_low_confidence_unknown_bot_bad_types_or_output_keep_the_default():
    nan = float("nan")
    cases = [
        {"bot_id": "juno", "project": None, "title": "t", "confidence": MIN_CONFIDENCE - 0.01, "reason": "maybe"},
        {"bot_id": "someone-else", "project": None, "title": "t", "confidence": 0.99, "reason": "guess"},
        {"bot_id": None, "project": "Nonexistent", "title": "t", "confidence": 0.99, "reason": "?"},
        {"bot_id": "juno", "project": None, "title": "t", "confidence": nan, "reason": "nan"},
        {"bot_id": "juno", "project": None, "title": "t", "confidence": 2, "reason": "too sure"},
        {"bot_id": "juno", "project": None, "title": "t", "confidence": "0.9", "reason": "string"},
        {"bot_id": "juno", "project": None, "title": "t", "confidence": True, "reason": "bool"},
        {"bot_id": ["juno"], "project": {"x": 1}, "title": "t", "confidence": 0.9, "reason": 5},
        "not json", ["a list"],
    ]
    for answer in cases:
        out = run("June maybe", answer)
        assert out["botId"] is None and out["project"] is None, answer
        assert "default" in out["reason"]


def test_model_failure_or_timeout_falls_back_without_raising():
    def boom(prompt, system):
        raise TimeoutError("slow")
    out = decide(item("Juno fix it"), BOTS, PROJECTS, complete=boom, system=fixed())
    assert out["botId"] is None and "default" in out["reason"]


def test_unloadable_or_blank_prompt_routes_nothing_and_skips_the_model():
    def never(prompt, system):
        raise AssertionError("model called without instructions")

    def broken():
        raise RuntimeError("registry down")
    for system in (fixed(""), fixed("  "), lambda: None, broken):
        out = decide(item("Juno fix it"), BOTS, PROJECTS, complete=never, system=system)
        assert out["botId"] is None and "default" in out["reason"]


def test_registry_key_resolves_to_the_default_body(monkeypatch):
    from llm_bawt.prompt_registry import PromptResolver
    assert PromptResolver.definition_for(PROMPT_KEY).default_body() == DEFAULT_SYSTEM

    class Resolver:
        def render(self, **kw):
            assert kw["key"] == PROMPT_KEY
            return "override"
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: Resolver())
    assert route.system_prompt() == "override"

    def down():
        raise RuntimeError("db")
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", down)
    assert route.system_prompt() is None


def test_project_only_route_keeps_the_original_title():
    out = run("Order more dog food",
              {"bot_id": None, "project": "Home", "title": "Dog food", "confidence": 0.9, "reason": "household"})
    assert out["project"] == "Home" and out["botId"] is None
