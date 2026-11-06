"""TASK-2116: where each bot's new turns run, resolved like the dispatcher."""

import asyncio
from types import SimpleNamespace as Row

from llm_bawt.bot_location import location_runtimes, resolve_locations


def bot(slug, backend="claude-code", house=None, pinned=False, bot_type="agent"):
    return Row(slug=slug, agent_backend=backend, house=house, house_pinned=pinned,
               bot_type=bot_type)


def resolve(rows, leases=None, *, fail=False):
    calls = []

    async def read(key):
        calls.append(key)
        if fail:
            raise ConnectionError("redis down")
        return (leases or {}).get(key)

    return asyncio.run(resolve_locations(rows, read)), calls


def test_backend_decides_the_bridge_not_the_model():
    out, calls = resolve([bot("vex"), bot("codex", "codex"),
                          bot("mira", None, bot_type="chat")])
    assert out == {"vex": {"runtime": "claude-code-bridge"},
                   "codex": {"runtime": "codex-bridge"},
                   "mira": {"runtime": "app"}}
    assert calls == []  # no house, no Redis read


def test_live_house_lease_runs_in_the_house():
    out, calls = resolve([bot("probe", house="H1", pinned=True)],
                         {"agent:house:h1:lease:claude-code": "7"})
    assert out["probe"] == {"runtime": "house", "house": "h1", "pinned": True}
    assert calls == ["agent:house:h1:lease:claude-code"]
    assert location_runtimes(out["probe"]) == {"house", "h1"}


def test_dead_house_falls_back_to_the_mansion_bridge():
    out, _ = resolve([bot("probe", "codex", house="h2")])
    assert out["probe"] == {"runtime": "codex-bridge", "house": "h2", "house_live": False}
    assert location_runtimes(out["probe"]) == {"codex-bridge"}


def test_unreadable_lease_is_unknown_not_mansion():
    out, _ = resolve([bot("probe", house="h1")], fail=True)
    assert out["probe"] == {"runtime": "house", "house": "h1", "house_live": None}


def test_invalid_house_goes_to_the_mansion_like_the_dispatcher():
    out, calls = resolve([bot("probe", house="not a house!")])
    assert out["probe"] == {"runtime": "claude-code-bridge"}
    assert calls == []
