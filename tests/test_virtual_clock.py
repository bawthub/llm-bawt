"""tests/virtual_clock.py: exact elapsed time with no real waiting (TASK-1457)."""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from tests import virtual_clock


def test_timers_keep_their_order_and_take_no_real_time():
    async def main():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.8):
                await asyncio.sleep(30)
        await asyncio.wait_for(asyncio.sleep(0.3), timeout=1)
        return loop.time() - started

    real_started = time.monotonic()
    assert virtual_clock.run(main()) == pytest.approx(1.1)
    # A real wait would take 1.1s; any bound below that proves none happened.
    assert time.monotonic() - real_started < 1.1  # timing-ok: the bound is the real wait it rules out


def test_a_followed_module_reads_the_virtual_clock_until_the_run_ends():
    module = types.ModuleType("followed")
    module.time = time

    async def main():
        monotonic, wall = module.time.monotonic(), module.time.time()
        await asyncio.sleep(2)
        return module.time.monotonic() - monotonic, module.time.time() - wall

    assert virtual_clock.run(main(), follow=[module]) == (pytest.approx(2), pytest.approx(2))
    assert module.time is time


def test_a_thread_or_a_wait_on_nothing_fails_instead_of_racing_or_hanging():
    async def thread():
        await asyncio.to_thread(time.sleep, 0)

    with pytest.raises(RuntimeError, match="real thread"):
        virtual_clock.run(thread())
    with pytest.raises(RuntimeError, match="deadlock"):
        virtual_clock.run(asyncio.Event().wait())
