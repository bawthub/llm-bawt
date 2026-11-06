"""TASK-2105 S3: app health change version and its publishers."""

from __future__ import annotations

import threading
import time

import httpx

from llm_bawt.service import health_changes as hc
from llm_bawt.service.health_changes import HealthChanges
from llm_bawt.service.maintenance_health import MaintenanceHealth
from llm_bawt.service.providers import ollama


def test_publish_coalesces_a_burst_into_one_bump():
    changes = HealthChanges(delay=0.05)
    start = changes.version
    for _ in range(5):
        changes.publish("write")
    assert changes.version == start  # not before the window
    assert changes.wait_for_change(start, 1.0) == start + 1
    time.sleep(0.1)
    assert changes.version == start + 1


def test_wait_returns_at_once_for_a_stale_seen_and_times_out_otherwise():
    changes = HealthChanges(delay=0)
    v = changes.version
    assert changes.wait_for_change(0, 5.0) == v
    t = time.monotonic()
    assert changes.wait_for_change(v, 0.05) == v
    assert time.monotonic() - t >= 0.04


def test_expiry_timer_targets_the_earliest_future_expiry():
    changes = HealthChanges(delay=0)
    now = 1_000_000
    changes.arm_expiry([None, now - 5, now + 60_000, now + 9_000], now_ms=now)
    assert changes._expiry[0] == now + 9_000
    # Same target is left alone; nothing in the future cancels it.
    timer = changes._expiry[1]
    changes.arm_expiry([now + 9_000], now_ms=now)
    assert changes._expiry[1] is timer
    changes.arm_expiry([None], now_ms=now)
    assert changes._expiry is None and timer.finished.is_set()  # cancelled


def test_expiry_fires_a_publish():
    changes = HealthChanges(delay=0)
    v = changes.version
    now = time.time() * 1000
    # arm_expiry adds 1s; aim just behind now so it fires within ~1s.
    changes.arm_expiry([now + 10], now_ms=now)
    assert changes.wait_for_change(v, 3.0) == v + 1


def test_maintenance_publishes_only_on_flips(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(hc._changes, "publish", calls.append)
    registry = MaintenanceHealth()
    kw = dict(model="m", provider="p", credential=None)
    registry.record_success("job", **kw)
    registry.record_success("job", **kw)
    registry.record_failure("job", "auth", "bad", **kw)
    registry.record_failure("job", "auth", "bad", **kw)
    registry.record_success("job", **kw)
    assert calls == ["maintenance job ok", "maintenance job failing", "maintenance job ok"]


def test_ollama_flip_and_chat_connect_error_publish(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(hc._changes, "publish", calls.append)
    ollama.clear_health_cache()
    root = "http://ollama.test:11434"
    ollama._store_probe(root, None, ("0.5", None, False))
    ollama._store_probe(root, None, ("0.5", None, False))
    assert calls == []  # first observation and a repeat are not changes
    # A chat request to an untracked server is ignored.
    ollama.report_unreachable("http://other.test/v1", httpx.ConnectError("x"))
    assert calls == []
    ollama.report_unreachable(f"{root}/v1", httpx.ConnectError("refused"))
    assert calls == ["ollama ok -> offline"]
    hit = ollama._health_cache[(root, None)][1]
    assert hit[0] is None and hit[2] is True
    ollama.report_unreachable(f"{root}/v1", httpx.ConnectError("refused"))
    assert len(calls) == 1
    ollama.clear_health_cache()


def test_reporting_transport_reraises_and_reports(monkeypatch):
    from llm_bawt.clients import openai_client

    seen: list[str] = []
    monkeypatch.setattr(ollama, "report_unreachable", lambda url, err: seen.append(url))
    transport = openai_client._ConnectReportingTransport("http://127.0.0.1:9/v1")
    client = httpx.Client(transport=transport)
    try:
        client.get("http://127.0.0.1:9/v1/models")
    except httpx.ConnectError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ConnectError")
    assert seen == ["http://127.0.0.1:9/v1"]


def test_reachability_loop_stops():
    stop = threading.Event()
    stop.set()
    ollama.reachability_loop(object(), stop)  # returns at once
