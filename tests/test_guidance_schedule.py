"""TASK-893 slice 7: the guidance_run scheduled job is one explicit pair."""
import json
from types import SimpleNamespace

import pytest

from llm_bawt.service.background_tasks import BackgroundTasksMixin
from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob
from llm_bawt.service.tasks import TaskType, create_guidance_run_task


def _task(bot_id, cfg):
    job = ScheduledJob(job_type=JobType.GUIDANCE_RUN, bot_id=bot_id,
                       config_json=json.dumps(cfg) if cfg is not None else None)
    return JobScheduler(None, SimpleNamespace(config=None))._create_task_for_job(job)


def test_explicit_pair_builds_a_committing_task():
    task = _task("examplebot", {"user_id": "alice", "max_evidence": 50})
    assert task.task_type is TaskType.GUIDANCE_RUN
    assert (task.bot_id, task.user_id, task.payload["max_evidence"]) == ("examplebot", "alice", 50)


@pytest.mark.parametrize("bot_id,cfg", [("*", {"user_id": "alice"}), ("examplebot", None),
                                        ("examplebot", {"user_id": " "})])
def test_wildcard_or_missing_person_never_runs(bot_id, cfg):
    assert _task(bot_id, cfg) is None


@pytest.mark.parametrize("user_id", ["", "system", "bot:examplebot"])
def test_handler_refuses_non_person(user_id):
    with pytest.raises(ValueError):
        BackgroundTasksMixin._process_guidance_run(
            SimpleNamespace(config=None), create_guidance_run_task("examplebot", user_id))


def test_handler_commits_through_run_pair(monkeypatch):
    from llm_bawt.memory import guidance, guidance_run
    from llm_bawt.memory import history_read_cursor
    from llm_bawt.utils import db

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    engine = SimpleNamespace(begin=lambda: _Conn())
    monkeypatch.setattr(db, "get_shared_engine", lambda cfg: engine)
    monkeypatch.setattr(history_read_cursor, "ensure_schema", lambda c: None)
    monkeypatch.setattr(guidance, "ensure_schema", lambda c: None)
    seen = {}
    monkeypatch.setattr(guidance_run, "run_pair", lambda e, **kw: seen.update(kw) or {"status": "ok"})
    out = BackgroundTasksMixin._process_guidance_run(
        SimpleNamespace(config=None), create_guidance_run_task("examplebot", "alice", 7))
    assert out == {"status": "ok"}
    assert seen == {"bot_id": "examplebot", "user_id": "alice", "max_evidence": 7,
                    "dry_run": False, "proposed_by": "guidance-schedule"}


def _run_with(monkeypatch, proposal):
    from llm_bawt.memory import guidance_run
    from llm_bawt.service import state_events
    test_handler_commits_through_run_pair(monkeypatch)
    monkeypatch.setattr(guidance_run, "run_pair", lambda e, **kw: {"status": "done", "proposal": proposal})
    marked = []
    monkeypatch.setattr(state_events, "mark", lambda svc, *t: marked.extend(t))
    out = BackgroundTasksMixin._process_guidance_run(
        SimpleNamespace(config=None), create_guidance_run_task("examplebot", "alice"))
    return out, marked


def test_pending_proposal_marks_guidance_topic(monkeypatch):
    out, marked = _run_with(monkeypatch, {"status": "pending", "proposal_id": 3})
    assert marked == ["guidance"] and "error" not in out


def test_no_change_is_quiet_success(monkeypatch):
    out, marked = _run_with(monkeypatch, {"status": "no_change"})
    assert marked == [] and "error" not in out


@pytest.mark.parametrize("proposal", [{"status": "invalid_draft", "error": "timeout"},
                                      {"status": "rejected", "error": "stale base"},
                                      {"status": "aborted", "reason": "protected"}])
def test_noncommitting_draft_reports_failure(monkeypatch, proposal):
    out, marked = _run_with(monkeypatch, proposal)
    assert marked == [] and out["error"].startswith(f"guidance draft {proposal['status']}")
    assert out["proposal"] == proposal
