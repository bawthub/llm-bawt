"""Bounded callback recovery; no HTTP, containers or real sleeps."""
import json
from urllib.error import HTTPError, URLError

import pytest

from scripts import heavy_queue


@pytest.fixture
def callback_rig(tmp_path, monkeypatch):
    job = dict(id="fixture", notify_bot="requester", notify_url="http://example.test")
    receipt = dict(status="PASSED", exit_code=0, log_path="/log", receipt_path=str(tmp_path / "receipt.json"),
                   callback_delivery_id=None)
    sleeps = []
    monkeypatch.setattr(heavy_queue.time, "sleep", sleeps.append)
    heavy_queue.save(tmp_path / "job.json", job)
    heavy_queue.save(tmp_path / "receipt.json", receipt)
    return job, receipt, tmp_path / "receipt.json", sleeps


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_http_error_stops_after_one_attempt(callback_rig, monkeypatch, code):
    job, receipt, path, sleeps = callback_rig
    calls = []

    def fail(job, receipt):
        calls.append(job["id"])
        raise HTTPError("http://secret.invalid", code, "secret body", {}, None)

    monkeypatch.setattr(heavy_queue, "notify", fail)
    assert not heavy_queue.notify_with_retry(job, receipt, path)
    saved = json.loads(path.read_text())
    assert saved["status"] == "PASSED" and saved["exit_code"] == 0
    assert saved["callback_error"] == "HTTPError"
    assert saved["callback_attempts"] == 1
    assert saved["callback_delivery_id"] is None
    assert calls == ["fixture"] and sleeps == []
    assert "secret" not in path.read_text()


@pytest.mark.parametrize("error", [
    HTTPError("http://example.test", 408, "timeout", {}, None),
    HTTPError("http://example.test", 429, "throttle", {}, None),
    HTTPError("http://example.test", 503, "unavailable", {}, None),
    URLError("offline"), RuntimeError("no durable delivery ID"),
])
def test_transient_and_malformed_responses_have_finite_budget(callback_rig, monkeypatch, error):
    job, receipt, path, sleeps = callback_rig
    calls = []

    def fail(job, receipt):
        calls.append(job["id"])
        raise error

    monkeypatch.setattr(heavy_queue, "notify", fail)
    assert not heavy_queue.notify_with_retry(job, receipt, path)
    assert len(calls) == heavy_queue.CALLBACK_ATTEMPTS == 5
    assert sleeps == [5, 10, 15, 20]
    saved = json.loads(path.read_text())
    assert saved["callback_attempts"] == 5
    assert saved["callback_error"] == type(error).__name__
    assert saved["status"] == "PASSED"


def test_transient_then_success_clears_error(callback_rig, monkeypatch):
    job, receipt, path, sleeps = callback_rig
    calls = []

    def notify(job, receipt):
        calls.append(job["id"])
        if len(calls) == 1:
            raise URLError("lost response")
        return "delivery-fixture"

    monkeypatch.setattr(heavy_queue, "notify", notify)
    assert heavy_queue.notify_with_retry(job, receipt, path)
    saved = json.loads(path.read_text())
    assert saved["callback_delivery_id"] == "delivery-fixture"
    assert "callback_error" not in saved
    assert calls == ["fixture", "fixture"] and sleeps == [5]


def test_manual_retry_only_notifies_existing_job(callback_rig, monkeypatch):
    job, receipt, path, sleeps = callback_rig
    receipt["callback_error"] = "HTTPError"
    heavy_queue.save(path, receipt)
    calls = []
    monkeypatch.setattr(heavy_queue, "notify", lambda job, receipt: calls.append(job["id"]) or "delivery-manual")
    monkeypatch.setattr(heavy_queue, "run_job", lambda *args: pytest.fail("job replayed"))
    assert heavy_queue.main(["notify", str(path.parent)]) == 0
    saved = json.loads(path.read_text())
    assert saved["callback_delivery_id"] == "delivery-manual"
    assert "callback_error" not in saved
    assert saved["status"] == "PASSED"
    assert calls == [job["id"]] and sleeps == []


def test_manual_failure_records_error_without_retry(callback_rig, monkeypatch):
    job, receipt, path, sleeps = callback_rig

    def fail(*args):
        raise URLError("offline")

    monkeypatch.setattr(heavy_queue, "notify", fail)
    assert heavy_queue.main(["notify", str(path.parent)]) == 1
    assert json.loads(path.read_text())["callback_error"] == "URLError"
    assert sleeps == []
