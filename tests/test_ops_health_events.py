"""TASK-2105: Docker events push health changes; snapshots are reconciliation."""
from threading import Event

from llm_bawt.ops.health_collector import OpsHealthCollector, relevant_event
from llm_bawt.ops.health_state import HealthSample, OpsHealthState


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 1
        return self.t


class FakeEventsClient:
    def __init__(self, windows, stop):
        self.windows, self.stop, self.calls = list(windows), stop, []

    def events(self, **kwargs):
        self.calls.append(kwargs)
        if not self.windows:
            self.stop.set()
            return iter(())
        window = self.windows.pop(0)
        if callable(window):
            window = window()
        if isinstance(window, Exception):
            raise window
        return iter(window)

    def close(self):
        pass


def _collector(state, client):
    return OpsHealthCollector(state, lambda: None, events_client_factory=lambda: client)


def test_relevant_event_ignores_exec_noise():
    assert relevant_event({"Type": "container", "Action": "die"})
    assert relevant_event({"Type": "container", "Action": "health_status: unhealthy"})
    assert relevant_event({"Type": "image", "Action": "tag"})
    assert not relevant_event({"Type": "container", "Action": "exec_start: sh"})
    assert not relevant_event("garbage")


def test_clean_window_confirms_post_change_sample_and_windows_are_contiguous():
    state = OpsHealthState(clock=Clock())
    stop, wake = Event(), Event()
    resynced = HealthSample(None, "ok")

    def resync():  # the woken collector publishes after (re)connect
        nonlocal resynced
        resynced = HealthSample(state.sample_time(), "ok")
        state.publish("docker", resynced)
        return []
    client = FakeEventsClient([resync, []], stop)
    collector = _collector(state, client)
    collector.watch_docker(stop, wake, window=5)
    assert wake.is_set()  # connect forces a resync
    assert collector.docker_events_live
    assert state._samples["docker"].sampled.monotonic > resynced.sampled.monotonic
    assert client.calls[0]["until"] - 1 == client.calls[1]["since"]  # 1s overlap


def test_event_wakes_collector_and_does_not_refresh_pre_change_sample():
    state = OpsHealthState(clock=Clock())
    stop, wake = Event(), Event()
    stale = HealthSample(state.sample_time(), "ok")
    state.publish("docker", stale)
    client = FakeEventsClient([[{"Type": "container", "Action": "stop"}], []], stop)
    _collector(state, client).watch_docker(stop, wake, window=5)
    assert wake.is_set()
    assert state._samples["docker"] is stale  # older than the change: ages out


def test_stream_failure_falls_back_to_polling_and_resyncs():
    state = OpsHealthState(clock=Clock())
    stop, wake = Event(), Event()
    client = FakeEventsClient([OSError("gone")], stop)
    collector = _collector(state, client)
    collector.docker_events_live = True
    original_wait = stop.wait

    def wait(timeout=None):
        stop.set()
        return original_wait(0)
    stop.wait = wait
    collector.watch_docker(stop, wake, window=5, backoff=0)
    assert wake.is_set()
    assert not collector.docker_events_live
    assert collector.docker_interval() == 5.0


def test_run_wakes_early_on_event():
    stop, wake = Event(), Event()
    calls = []

    def collect():
        calls.append(1)
        if len(calls) == 1:
            wake.set()  # an event arrived mid-collection
        else:
            stop.set()
    OpsHealthCollector.run(collect, stop, interval=60, wake=wake)
    assert len(calls) == 2


def test_failed_collection_is_retried_fast_and_never_restamped():
    state = OpsHealthState(clock=Clock())
    stop, wake = Event(), Event()
    down = None

    def fail():
        nonlocal down
        down = HealthSample(state.sample_time(), "down", "container observation unavailable")
        state.publish("docker", down)
        return []
    collector = _collector(state, FakeEventsClient([fail, []], stop))
    collector.watch_docker(stop, wake, window=5)
    assert collector.docker_events_live
    assert state._samples["docker"] is down  # not kept looking current
    assert collector.docker_interval() == 5.0  # stream live, but retry the failure


def test_run_exits_when_stop_lands_before_wake_clear():
    wake, calls = Event(), []

    class LateStop:  # stop flips after the loop's check, before clear()
        n = 0

        def is_set(self):
            self.n += 1
            return self.n > 1

    OpsHealthCollector.run(lambda: calls.append(1), LateStop(), interval=60, wake=wake)
    assert calls == []


def test_change_version_bumps_on_content_not_restamp():
    state = OpsHealthState(clock=Clock())
    v0 = state.wait_for_change(-1, 0)
    state.publish("docker", HealthSample(state.sample_time(), "ok"))
    v1 = state.wait_for_change(v0, 0)
    assert v1 != v0
    # Same content, new timestamp: a re-stamp, not a change.
    state.publish("docker", HealthSample(state.sample_time(), "ok"))
    assert state.wait_for_change(v1, 0) == v1
    state.publish("docker", HealthSample(state.sample_time(), "down", "gone"))
    assert state.wait_for_change(v1, 0) != v1


def test_change_wait_wakes_on_publish_and_stop():
    from threading import Thread
    state = OpsHealthState(clock=Clock())
    seen = state.wait_for_change(-1, 0)
    got = []
    waiter = Thread(target=lambda: got.append(state.wait_for_change(seen, 5)))
    waiter.start()
    state.publish("database", HealthSample(state.sample_time(), "ok"))
    waiter.join(2)
    assert got and got[0] != seen
    seen = got[0]
    waiter = Thread(target=lambda: got.append(state.wait_for_change(seen, 5)))
    waiter.start()
    state.stop()
    waiter.join(2)
    assert not waiter.is_alive()


def test_changes_endpoint_long_polls_and_rejects_bad_versions():
    from fastapi.testclient import TestClient
    from llm_bawt.ops import health_http
    state = OpsHealthState(clock=Clock())
    client = TestClient(health_http.create_health_app(state))
    version = client.get("/v1/health/changes/0").json()["version"]
    health_http.CHANGES_HOLD_SECONDS = 0.05
    try:
        assert client.get(f"/v1/health/changes/{version}").json() == {"schemaVersion": 1, "version": version}
    finally:
        health_http.CHANGES_HOLD_SECONDS = 25.0
    assert client.get("/v1/health/changes/x1").status_code == 404
    assert client.get("/v1/health/changes/1?a=b").status_code == 404


def test_changes_endpoint_refuses_beyond_the_waiter_cap(monkeypatch):
    from fastapi.testclient import TestClient
    from llm_bawt.ops import health_http
    state = OpsHealthState(clock=Clock())
    monkeypatch.setattr(health_http, "MAX_CHANGE_WAITERS", 0)
    client = TestClient(health_http.create_health_app(state))
    # At the cap it refuses at once, so the client backs off instead of spinning.
    assert client.get("/v1/health/changes/1", timeout=5).status_code == 503


def test_catalog_rereads_on_notify_and_closes_listener():
    stop = Event()
    reads, waits = [], []

    class Waiter:
        closed = False

        def wait(self, stop_, timeout):
            waits.append(timeout)
            if len(waits) == 2:
                stop_.set()
            return True

        def close(self):
            Waiter.closed = True

    collector = OpsHealthCollector(OpsHealthState(clock=Clock()), lambda: None)
    collector.collect_catalog = lambda: reads.append(1)
    collector.watch_catalog(stop, Waiter())
    assert reads == [1, 1] and waits == [60.0, 60.0] and Waiter.closed


def test_operation_writes_notify_the_catalog_channel():
    from llm_bawt.ops import store as store_mod
    src = open(store_mod.__file__).read()
    assert src.count("self._notify_catalog(session, row.slug)") == 2
