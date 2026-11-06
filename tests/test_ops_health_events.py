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
