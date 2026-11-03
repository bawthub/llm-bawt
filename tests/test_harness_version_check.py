"""TASK-1975: weekly harness version check keeps one open upgrade task."""

import json

import httpx
import pytest

from llm_bawt.service.jobs import harness_version_check as hvc

CHANGELOG = """# Changelog

## 0.2.177

- Updated bundled Claude CLI to version 2.1.317

## 0.2.176

- Updated bundled Claude CLI to version 2.1.316
"""
LATEST = {
    "claude-agent-sdk": "0.2.177",
    "openai-codex": "0.163.0",
    "openai-codex-cli-bin": "0.163.0",
    "openai-codex-sdk": "0.1.11",
}


def pypi_client(fail=False):
    def handler(request):
        if fail:
            raise httpx.ConnectError("offline", request=request)
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(200, text=CHANGELOG)
        name = request.url.path.split("/")[2]
        return httpx.Response(200, json={"info": {"version": LATEST[name]}})

    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeTasks:
    def __init__(self):
        self.tasks = []
        self.calls = []

    def client(self):
        def handler(request):
            self.calls.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json={"tasks": self.tasks})
            body = json.loads(request.content)
            if request.method == "POST":
                task = {"id": f"t{len(self.tasks)}", "shortId": f"TASK-{len(self.tasks)}", **body}
                self.tasks.append(task)
                return httpx.Response(201, json=task)
            tid = request.url.path.rsplit("/", 1)[1]
            next(t for t in self.tasks if t["id"] == tid).update(body)
            return httpx.Response(200, json={})

        return httpx.Client(base_url="http://bawthub", transport=httpx.MockTransport(handler))


@pytest.fixture
def pins(tmp_path):
    path = tmp_path / "constraints.txt"
    path.write_text(
        "# pins\nclaude-agent-sdk==0.2.176\nhttpx==0.28.1\nopenai-codex==0.163.0\n"
        "openai-codex-cli-bin==0.163.0\nopenai-codex-sdk==0.1.11\n"
    )
    return str(path)


def run(pins, tasks, **kw):
    return hvc.run_harness_version_check(
        constraints_path=pins, http_client=pypi_client(), task_client=tasks.client(), **kw
    )


def test_creates_one_queued_task_then_no_duplicate(pins):
    tasks = FakeTasks()
    first = run(pins, tasks, agent_bot_id="owner")
    assert first["behind"] == ["claude-agent-sdk"]
    sdk = next(p for p in first["packages"] if p["name"] == "claude-agent-sdk")
    assert (sdk["pinned_cli"], sdk["latest_cli"]) == ("2.1.316", "2.1.317")
    assert first["task"]["action"] == "created"
    [task] = tasks.tasks
    assert task["status"] == "QUEUED" and task["agentBotId"] == "owner"
    assert "2.1.316 -> 2.1.317" in task["description"]

    second = run(pins, tasks)
    assert second["task"]["action"] == "none"
    assert len(tasks.tasks) == 1


def test_updates_open_task_when_versions_move(pins):
    tasks = FakeTasks()
    run(pins, tasks)
    LATEST["openai-codex"] = "0.164.0"
    try:
        result = run(pins, tasks)
    finally:
        LATEST["openai-codex"] = "0.163.0"
    assert result["task"]["action"] == "updated"
    assert len(tasks.tasks) == 1 and "openai-codex 0.164.0" in tasks.tasks[0]["title"]


def test_completed_task_is_not_reused(pins):
    tasks = FakeTasks()
    run(pins, tasks)
    tasks.tasks[0]["status"] = "COMPLETED"
    assert run(pins, tasks)["task"]["action"] == "created"
    assert len(tasks.tasks) == 2


def test_current_pins_write_no_task(pins):
    tasks = FakeTasks()
    LATEST["claude-agent-sdk"] = "0.2.176"
    try:
        result = run(pins, tasks)
    finally:
        LATEST["claude-agent-sdk"] = "0.2.177"
    assert result["behind"] == [] and tasks.calls == []


def test_network_failure_fails_run_without_task(pins):
    tasks = FakeTasks()
    with pytest.raises(httpx.ConnectError):
        hvc.run_harness_version_check(
            constraints_path=pins, http_client=pypi_client(fail=True), task_client=tasks.client()
        )
    assert tasks.calls == []


def test_default_constraints_file_has_the_harness_pins():
    pins = hvc.read_pins(hvc.DEFAULT_CONSTRAINTS, hvc.DEFAULT_PACKAGES)
    assert set(pins) == set(hvc.DEFAULT_PACKAGES)
