"""TASK-1697: the renderer undoes Compose's ``$$`` escape before the Engine sees it."""
from __future__ import annotations

import json

from llm_bawt.ops.compose_render import ComposeRenderer, unescape_dollars

TEST = "curl -s -o /dev/null http://localhost:5173/; [ $$? != 7 ] && curl -fsS http://x/live"


def test_unescape_is_the_inverse_of_compose_escaping():
    assert unescape_dollars({"t": ["CMD-SHELL", TEST], "n": 3, "e": {"A": "a$$$$b"}}) == {
        "t": ["CMD-SHELL", TEST.replace("$$?", "$?")], "n": 3, "e": {"A": "a$$b"}}
    assert unescape_dollars("no dollars") == "no dollars"


def test_render_hands_the_engine_unescaped_strings(monkeypatch):
    spec = {"compose_service": "frontend"}
    renderer = ComposeRenderer(object(), spec, tag="t")
    rendered = {"services": {"frontend": {"healthcheck": {"test": ["CMD-SHELL", TEST]}}}}
    outputs = iter([json.dumps(rendered), "frontend " + "a" * 64])
    monkeypatch.setattr(renderer, "_run", lambda *_a: next(outputs))
    monkeypatch.setattr("llm_bawt.ops.compose_render.renderer_argv", lambda *_a, **_k: [])
    test = renderer.render({}, "img")["config"]["services"]["frontend"]["healthcheck"]["test"][1]
    assert "[ $? != 7 ]" in test and "$$" not in test
