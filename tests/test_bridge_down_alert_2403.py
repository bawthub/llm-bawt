"""TASK-2403: the app tells the user when claude-code-bridge stays down."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from llm_bawt.service import bridge_down_alert as mod
from llm_bawt.service.bridge_down_alert import BridgeDownAlert, probe_health


class Harness:
    def __init__(self):
        self.now, self.healthy, self.sent = 0.0, True, []

        async def probe():
            return self.healthy

        async def send(title, message, key):
            self.sent.append((title, key))
            return {"result": "sent"}

        self.alert = BridgeDownAlert(probe, send, clock=lambda: self.now)

    def at(self, seconds, healthy):
        self.now, self.healthy = seconds, healthy
        asyncio.run(self.alert.step())
        return [title for title, _key in self.sent]


def test_one_alert_after_five_minutes_hourly_resend_and_one_recovery():
    h = Harness()
    assert h.at(0, True) == []
    assert h.at(60, False) == []                      # outage starts
    assert h.at(60 + 299, False) == []                # not yet 5 min
    assert h.at(60 + 300, False) == ["claude-code-bridge is down"]
    assert h.at(60 + 300 + 3599, False) == ["claude-code-bridge is down"]  # no resend inside the hour
    assert h.at(60 + 300 + 3600, False) == ["claude-code-bridge is down"] * 2
    assert h.at(60 + 300 + 3660, True)[-1] == "claude-code-bridge recovered"
    assert len(h.sent) == 3
    assert h.at(10_000, True) and len(h.sent) == 3   # no repeat recovery
    keys = [key for _t, key in h.sent]
    assert len(set(keys)) == 3                        # store dedupe never swallows a resend


def test_a_short_blip_sends_nothing():
    h = Harness()
    h.at(0, False)
    h.at(240, False)
    h.at(300 - 1, True)
    assert h.sent == []


def test_a_failing_send_does_not_break_the_loop():
    h = Harness()

    async def boom(*_a):
        raise RuntimeError("ha down")

    h.alert.send = boom
    h.at(0, False)
    h.at(300, False)  # logged, not raised
    assert h.alert.alerted_at == 300


class _Health(BaseHTTPRequestHandler):
    body = b'{"redis": true, "command_consumer": true}'

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_a):
        pass


def test_probe_against_a_fake_health_source_and_a_stopped_one():
    server = HTTPServer(("127.0.0.1", 0), _Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/health"
    try:
        assert asyncio.run(probe_health(url)) is True
        _Health.body = json.dumps({"redis": False}).encode()
        assert asyncio.run(probe_health(url)) is False
    finally:
        server.shutdown()
        server.server_close()
    assert asyncio.run(probe_health(url)) is False  # stopped: connection refused


def test_thresholds_match_the_card():
    assert (mod.DOWN_AFTER_S, mod.RESEND_S, mod.CHECK_INTERVAL_S) == (300, 3600, 60)
