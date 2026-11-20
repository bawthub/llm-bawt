"""Exercise the production metrics route and watchdog against real ASGI requests."""

import asyncio
import io
import json
import logging
import threading

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from llm_bawt.service import loop_watchdog
from llm_bawt.service.loop_watchdog import LoopStallWatchdog, monitor_serving_loop
from llm_bawt.service.routes import health

HANG_GUARD = 5


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    return app


def test_stall_captures_in_flight_stack_without_holding_app_log_sink(monkeypatch):
    async def exercise():
        app = _app()
        writing = threading.Event()
        recovered = asyncio.Event()
        release = threading.Event()
        output = []
        timed_out = threading.Event()
        root = logging.getLogger()

        def gate_write(line):
            output.append(line)
            writing.set()
            if not release.wait(HANG_GUARD):
                timed_out.set()

        class AppStream(io.StringIO):
            def write(self, text):
                if "event_loop_stall " in text:
                    # If the watchdog regresses to root logging, it holds the
                    # exact handler lock the next request needs until release.
                    gate_write(text.encode())
                return super().write(text)

        app_stream = AppStream()
        app_handler = logging.StreamHandler(app_stream)
        root.addHandler(app_handler)

        def stalled_stderr(fd, line):
            assert fd == 2
            gate_write(line)
            return len(line)

        monkeypatch.setattr(loop_watchdog, "_write_fd", stalled_stderr)

        @app.get("/blocked")
        async def blocked_handler():
            return {"captured_during_request": writing.wait(HANG_GUARD)}

        @app.get("/ordinary")
        async def ordinary_handler():
            root.warning("ordinary request ran")
            return {"ok": True}

        monitor = LoopStallWatchdog(heartbeat=0.01, threshold=0.15)
        beat = monitor._beat

        def record_recovery():
            lag = beat()
            if writing.is_set() and lag >= 0.15:
                recovered.set()
            return lag

        monkeypatch.setattr(monitor, "_beat", record_recovery)
        try:
            async with monitor_serving_loop(app, watchdog=monitor), AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                try:
                    assert (await client.get("/blocked")).json() == {"captured_during_request": True}
                    assert (await client.get("/ordinary")).json() == {"ok": True}
                    assert "ordinary request ran" in app_stream.getvalue()
                    assert not timed_out.is_set(), "watchdog held the ordinary request's log handler lock"
                    await asyncio.wait_for(recovered.wait(), HANG_GUARD)
                    first = await client.get("/metrics")
                    second = await client.get("/metrics")
                    assert first.status_code == 200
                    assert first.headers["cache-control"] == "no-store"
                    assert first.json()["event_loop_stalls_total"] >= 1
                    assert second.json()["event_loop_lag_max_ms"] >= first.json()["event_loop_lag_max_ms"] >= 150
                    assert 0 <= first.json()["event_loop_lag_p99_ms"] <= first.json()["event_loop_lag_max_ms"]
                    assert first.json()["event_loop_window_seconds"] == 60
                    assert all(type(value) in {float, int} for value in first.json().values())
                finally:
                    release.set()
            assert not hasattr(app.state, "loop_watchdog")
            assert not monitor._watcher.is_alive()
        finally:
            root.removeHandler(app_handler)
            app_handler.close()

        assert len(output) == 1
        prefix = b"WARNING event_loop_stall "
        assert output[0].startswith(prefix) and len(output[0]) <= 4096
        payload = json.loads(output[0][len(prefix):])
        assert payload["lag_ms_at_capture"] >= 150
        assert any(frame["function"] == "blocked_handler" for frame in payload["frames"])
        assert all(set(frame) == {"module", "file", "line", "function"} for frame in payload["frames"])

    asyncio.run(exercise())


def test_health_handler_keeps_serving_other_requests_during_sync_work(monkeypatch):
    async def exercise():
        app = _app()
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original_health = health.HealthResponse

        def slow_health_response():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(HANG_GUARD), "health handler blocked the serving loop"
            return original_health()

        monkeypatch.setattr(health, "HealthResponse", slow_health_response)
        async with monitor_serving_loop(app), AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            in_flight = asyncio.create_task(client.get("/health"))
            try:
                await asyncio.wait_for(entered.wait(), HANG_GUARD)
                assert (await client.get("/metrics")).status_code == 200
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
