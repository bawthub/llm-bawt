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


def test_every_engine_bound_field_is_unescaped():
    # Snark's audit ask: env values (secrets with `$`), command, entrypoint and
    # labels ride the same rendered dict as the healthcheck.
    svc = {"environment": {"PW": "a$$b$$$$c"}, "command": ["sh", "-c", "echo $$HOME"],
           "entrypoint": ["/bin/sh", "-c", "x=$$1"], "labels": {"l": "p$$q"},
           "healthcheck": {"test": ["CMD-SHELL", "[ $$? = 0 ]"]}}
    assert unescape_dollars(svc) == {
        "environment": {"PW": "a$b$$c"}, "command": ["sh", "-c", "echo $HOME"],
        "entrypoint": ["/bin/sh", "-c", "x=$1"], "labels": {"l": "p$q"},
        "healthcheck": {"test": ["CMD-SHELL", "[ $? = 0 ]"]}}


def test_label_keys_and_values_reach_translate_unescaped(monkeypatch):
    # Loopy r1 B2: real `compose config` renders label key `xx$yy` as `xx$$yy`.
    from llm_bawt.ops.compose_config import translate
    from tests.test_ops_compose_config import REF, _rendered
    from tests.test_ops_image_deploy import SPEC

    # The pinned profile refuses `command`; the field test above covers it.
    rendered = _rendered(labels={"xx$$yy": "v$$w"}, environment={"PW": "a$$b"})
    renderer = ComposeRenderer(object(), SPEC, tag="t")
    outputs = iter([json.dumps(rendered), "frontend-prod " + "a" * 64])
    monkeypatch.setattr(renderer, "_run", lambda *_a: next(outputs))
    monkeypatch.setattr("llm_bawt.ops.compose_render.renderer_argv", lambda *_a, **_k: [])
    desired = translate(renderer.render({}, REF)["config"], SPEC, REF)
    assert desired["labels"].get("xx$yy") == "v$w" and "xx$$yy" not in desired["labels"]
    assert desired["env"]["PW"] == "a$b"
    assert "$$" not in json.dumps(desired)
