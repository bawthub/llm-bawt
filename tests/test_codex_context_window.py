"""The per-run CLI compaction cap is also the usage gauge denominator."""
import asyncio

import pytest

from codex_bridge.context_window import compact_config_overrides, effective_context_window
from codex_bridge.exec_patch import CONFIG_OVERRIDES, with_config_overrides


@pytest.mark.parametrize("value", [None, "", "bad", "1.5", 1.5, True, False, 0, -1, "0",
                                  "225000\nother=true", 2**63])
def test_absent_or_invalid_compact_threshold_has_no_override(value):
    assert compact_config_overrides(value) == ()


@pytest.mark.parametrize("value", [225000, "225000"])
def test_valid_compact_threshold_merges_without_losing_skills(value):
    overrides = ("skills.bundled.enabled=false",) + compact_config_overrides(value)
    command = with_config_overrides(["exec", "--experimental-json", "resume", "thread"], overrides)
    assert command == ["exec", "--experimental-json", "--config", "skills.bundled.enabled=false",
                       "--config", "model_auto_compact_token_limit=225000", "resume", "thread"]


@pytest.mark.parametrize("window,limit,expected", [
    (475000, 225000, 225000), (100000, 225000, 100000), (475000, None, 475000),
    (None, 225000, None), (0, 225000, None), (475000, "bad", 475000),
])
def test_effective_window(window, limit, expected):
    assert effective_context_window(window, compact_config_overrides(limit)) == expected


def test_concurrent_per_run_overrides_do_not_leak():
    async def run():
        ready = asyncio.Event()

        async def turn(limit):
            token = CONFIG_OVERRIDES.set(compact_config_overrides(limit))
            try:
                ready.set()
                await ready.wait()
                await asyncio.sleep(0)
                return effective_context_window(475000, CONFIG_OVERRIDES.get())
            finally:
                CONFIG_OVERRIDES.reset(token)

        return await asyncio.gather(turn(225000), turn(None), turn(150000))

    assert asyncio.run(run()) == [225000, 475000, 150000]
