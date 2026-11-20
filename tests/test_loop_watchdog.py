"""Exercise the production metrics route and watchdog against real ASGI requests."""

import asyncio
import json
import logging
import threading

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from llm_bawt.service.loop_watchdog import LoopStallWatchdog, monitor_serving_loop
from llm_bawt.service.routes import health


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    return app


def test_blocking_request_captures_its_stack_before_returning(caplog):
    async def exercise():
        app = _app()
        captured = threading.Event()
        logger = logging.getLogger("llm_bawt.service.loop_watchdog")

        class Capture(logging.Handler):
            def emit(self, record):
                captured.set()

        @app.get("/blocked")
        async def blocked_handler():
            # The independent thread must capture this stack to release the handler.
            return {"captured_during_request": captured.wait(0.7)}

        signal = Capture()
        logger.addHandler(signal)
        caplog.set_level(logging.WARNING, logger=logger.name)
        try:
            monitor = LoopStallWatchdog(heartbeat=0.01, threshold=0.15)
            async with monitor_serving_loop(app, watchdog=monitor), AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                assert (await client.get("/blocked")).json() == {"captured_during_request": True}
                # Give the overdue heartbeat a chance to record the completed delay.
                await asyncio.sleep(0.02)
                first = await client.get("/metrics")
                second = await client.get("/metrics")
                assert first.status_code == 200
                assert first.headers["cache-control"] == "no-store"
                assert first.json()["event_loop_stalls_total"] == second.json()["event_loop_stalls_total"] == 1
                assert first.json()["event_loop_lag_max_ms"] >= 150
                assert first.json()["event_loop_lag_p99_ms"] >= 150
                assert first.json()["event_loop_window_seconds"] == 60
                assert all(type(value) in {float, int} for value in first.json().values())
            assert not hasattr(app.state, "loop_watchdog")
            assert not any(thread.name == "event-loop-watchdog" for thread in threading.enumerate())
        finally:
            logger.removeHandler(signal)

    asyncio.run(exercise())
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("event_loop_stall ")]
    assert len(warnings) == 1
    payload = json.loads(warnings[0].split(" ", 1)[1])
    assert payload["lag_ms_at_capture"] >= 150
    assert any(frame["function"] == "blocked_handler" for frame in payload["frames"])
    assert all(set(frame) == {"module", "file", "line", "function"} for frame in payload["frames"])


def test_health_handler_keeps_serving_other_requests_during_sync_work(monkeypatch):
    async def exercise():
        app = _app()
        entered = threading.Event()
        release = threading.Event()
        original_health = health.HealthResponse

        def slow_health_response():
            entered.set()
            assert release.wait(0.6), "health handler blocked the serving loop"
            return original_health()

        monkeypatch.setattr(health, "HealthResponse", slow_health_response)
        async with monitor_serving_loop(app), AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            in_flight = asyncio.create_task(client.get("/health"))
            try:
                assert await asyncio.to_thread(entered.wait, 0.6)
                metrics = await client.get("/metrics")
                assert metrics.json()["event_loop_stalls_total"] == 0
                assert not in_flight.done()
            finally:
                release.set()
                assert (await in_flight).status_code == 200

    asyncio.run(exercise())


def test_metrics_without_serving_lifespan_is_unavailable():
    async def exercise():
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
            response = await client.get("/metrics")
            assert response.status_code == 503
            assert response.headers["cache-control"] == "no-store"
            assert response.json() == {"error": "metrics_unavailable"}

    asyncio.run(exercise())
