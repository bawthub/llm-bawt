"""TASK-2274: small defects in shared bridge, Codex and local-model bridges."""
from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge import houses
from agent_bridge.houses import HouseSlot, lease_key, turn_session_key

from tests.test_house_recycle import GROUP, H1, FakeRedis, _control, _house


# --- houses: recycle survives a Redis blip ------------------------------------

def test_recycle_retries_a_failed_ack_and_still_exits(monkeypatch):
    monkeypatch.setattr(houses, "RECYCLE_POLL_S", 0.001)

    async def go():
        redis = FakeRedis()
        slot = await _house(redis)
        real_xack, fails = redis.xack, [1]

        async def flaky_xack(*a):
            if fails:
                fails.pop()
                raise ConnectionError("redis blip")
            return await real_xack(*a)
        redis.xack = flaky_xack
        exits = []
        assert await slot.recycle(_control(slot), "9-0", redis, group=GROUP,
                                  exit_process=lambda: exits.append(1))
        assert exits == [1] and redis.acks == [(H1, GROUP, "9-0")]
        assert lease_key("h1", "codex") not in redis.kv
    asyncio.run(go())


# --- houses: finished turn removes only its own session route -------------------

def test_refresh_keeps_a_session_route_claimed_by_another_house(monkeypatch):
    async def go():
        redis = FakeRedis()
        slot = await _house(redis)
        sess = turn_session_key("codex", "s1")
        slot._turns[("s1", "r1")] = -1e9  # long finished, outside grace
        redis.kv[sess] = "agent:commands:house:h2"  # newer turn elsewhere
        await slot.refresh()
        assert redis.kv[sess] == "agent:commands:house:h2"
        # Its own route is still removed.
        slot._turns[("s1", "r2")] = -1e9
        redis.kv[sess] = slot.stream
        await slot.refresh()
        assert sess not in redis.kv
    asyncio.run(go())


# --- preview supervisor ---------------------------------------------------------

class CasRedis:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.on_get = None

    async def get(self, key):
        value = self.data.get(key)
        if self.on_get:
            self.on_get(self)
        return value

    async def delete(self, key):
        self.data.pop(key, None)

    async def eval(self, script, numkeys, key, expected):
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0


def _supervisor(redis, tmp_path):
    from agent_bridge import house_preview_supervisor as sup
    s = sup.PreviewSupervisor.__new__(sup.PreviewSupervisor)
    s._redis, s.house, s.target = redis, "h1", None
    return s


@pytest.mark.parametrize("old", [None, "not json"])
def test_preview_poll_never_deletes_a_choice_written_after_its_read(tmp_path, old):
    async def go():
        key = "choice"
        redis = CasRedis({key: old} if old else {})
        new = json.dumps({"bot": "nova", "path": "/x"})
        redis.on_get = lambda r: r.data.__setitem__(key, new)
        s = _supervisor(redis, tmp_path)
        assert await s._valid_choice(key) is None
        assert redis.data[key] == new
    asyncio.run(go())


def test_preview_launch_exception_restores_fallback_and_backs_off(tmp_path, monkeypatch):
    from agent_bridge import house_preview_supervisor as sup

    async def go():
        s = _supervisor(CasRedis(), tmp_path)
        s._proc, s._failures, s._retry_at = None, 0, 0.0
        s.occ, s.campus_url = sup.Occupancy((), 0), ""
        s.fallback = SimpleNamespace(serving=False, page="", start=AsyncMock(),
                                     stop=AsyncMock())
        monkeypatch.setattr(sup, "LOG_PATH", tmp_path / "log")
        monkeypatch.setattr(sup.PreviewSupervisor, "_step", AsyncMock(return_value=True))
        monkeypatch.setattr(sup.asyncio, "create_subprocess_exec",
                            AsyncMock(side_effect=FileNotFoundError("gone")))
        monkeypatch.setattr(sup, "render_page", lambda *a: "page")
        await s._serve(tmp_path)
        assert s._failures == 1 and s._retry_at > 0
        assert s.fallback.start.await_count >= 2  # restarted after the stop
    asyncio.run(go())


# --- preview port match ----------------------------------------------------------

def test_is_vite_requires_the_exact_port():
    from agent_bridge.house_preview import _is_vite
    vite = ["node", "/x/vite/bin/vite.js"]
    assert _is_vite([*vite, "--port", "5173"])
    assert _is_vite([*vite, "--port=5173"])
    assert not _is_vite([*vite, "--port", "15173"])
    assert not _is_vite([*vite, "--port", "3000", "--base", "/5173/"])


# --- NotebookEdit attribution ----------------------------------------------------

def test_notebook_edit_path_is_captured(tmp_path):
    from agent_bridge.changed_files import ToolChangedFileCapture
    nb = tmp_path / "a.ipynb"
    nb.write_text("{}")
    cap = ToolChangedFileCapture(cwd=tmp_path)
    cap.start(request_id="r1", tool_use_id="t1", tool_name="NotebookEdit",
              arguments={"notebook_path": str(nb)})
    assert cap._pending  # a capture was recorded


def test_app_side_notebook_edit_fallback_uses_notebook_path():
    from llm_bawt.service import tool_changed_files as tcf
    store = tcf.ToolChangedFilesCoordinator.__new__(tcf.ToolChangedFilesCoordinator)
    seen = {}

    def capture(files, bot):
        seen["files"] = files
        raise RuntimeError("stop before storage")
    store._apply_identity = capture
    try:
        store.persist({"turn_id": "t", "tool_name": "NotebookEdit",
                       "arguments": {"notebook_path": "/w/a.ipynb"}})
    except Exception:
        pass  # storage is not under test; the path decode is
    assert seen["files"][0].path == "/w/a.ipynb"


# --- subscriber -----------------------------------------------------------------

def test_latest_activity_splices_utf16_offsets():
    from agent_bridge.subscriber import RedisSubscriber
    events = [  # newest first, as xrevrange returns
        {"_type": "text_delta", "turn_id": "t", "text_offset": 6, "delta": "next"},
        {"_type": "tool_event", "event": "tool_start", "turn_id": "t", "text_offset": 6},
        {"_type": "text_delta", "turn_id": "t", "text_offset": 2, "delta": "x \U0001F600"},
        {"_type": "text_delta", "turn_id": "t", "text_offset": 0, "delta": "\U0001F600"},
        {"_type": "turn_start", "turn_id": "t", "content": "hi"},
    ]
    sub = RedisSubscriber.__new__(RedisSubscriber)
    batch = [(f"{i}-0", {"payload": json.dumps(e)}) for i, e in enumerate(events)]
    sub._pub_redis = SimpleNamespace(xrevrange=AsyncMock(side_effect=[batch, []]))
    got = asyncio.run(sub.latest_activity("b", "u"))
    assert got["full_text"] == "\U0001F600x \U0001F600next"
    assert got["last_bubble"] == "next"


def test_close_disconnects_the_owned_publisher_pool():
    from agent_bridge.subscriber import RedisSubscriber
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._redis = SimpleNamespace(aclose=AsyncMock())
    pool = SimpleNamespace(disconnect=AsyncMock())
    sub._pub_redis = SimpleNamespace(aclose=AsyncMock(), connection_pool=pool)
    asyncio.run(sub.close())
    pool.disconnect.assert_awaited_once()


# --- Codex ------------------------------------------------------------------------

def test_codex_toml_migration_drops_multiline_args(tmp_path):
    from codex_bridge import local_plugins as lp
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp_servers.crawl4ai]\ncommand = "npx"\nargs = [\n  "mcp-remote",\n'
        '  "http://x/[y]",\n]\nstartup_timeout_sec = 30\n\n[other]\nk = 1\n')
    lp._upsert_mcp_server_config(config_path=cfg, name="crawl4ai",
                                 url="http://c/mcp", logger=MagicMock())
    data = tomllib.loads(cfg.read_text())
    assert data["mcp_servers"]["crawl4ai"] == {"url": "http://c/mcp",
                                               "startup_timeout_sec": 30}
    assert data["other"] == {"k": 1}


def test_codex_final_command_output_wins_over_buffered_update():
    from codex_bridge.event_ops import CodexEventMixin
    ops = CodexEventMixin.__new__(CodexEventMixin)
    published = []
    ops._publish_event = lambda *a, **kw: published.append(kw)
    item = {"type": "command_execution", "id": "i1",
            "aggregated_output": "first line\nfinal line\n", "exit_code": 0}
    buffers = {"i1": {"aggregated_output": "first line\n"}}
    ops._emit_tool_end(item, "r", "s", 0, buffers, [])
    assert published[-1]["tool_result"] == "first line\nfinal line\n"


# --- local model ------------------------------------------------------------------

def test_vllm_chunks_preserve_leading_whitespace():
    from local_model_bridge.vllm_client import VLLMClient
    text = "    indented code\nnext line\n"
    assert "".join(VLLMClient._stream_chunks(None, text)) == text
    assert "".join(VLLMClient._stream_chunks(None, "\n\n  ")) == "\n\n  "


def test_llama_cpp_stream_error_propagates():
    from local_model_bridge.llama_cpp_client import LlamaCppClient

    def stream():
        yield {"choices": [{"delta": {"content": "partial"}}]}
        raise RuntimeError("native generation failed")
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.console = MagicMock()
    got = []
    with pytest.raises(RuntimeError):
        for chunk in client._iterate_llama_cpp_chunks(stream()):
            got.append(chunk)
    assert got == ["partial"]


def test_video_delete_refuses_pending_and_removes_failed_input(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from local_model_bridge import video_server as vs
    monkeypatch.setattr(vs.jobs, "directory", tmp_path)
    monkeypatch.setattr(vs.jobs, "jobs", {"p": {"status": "pending", "progress": 0},
                                          "f": {"status": "failed", "progress": 0}})
    with pytest.raises(HTTPException) as err:
        vs.remove("p")
    assert err.value.status_code == 409 and "p" in vs.jobs.jobs
    (tmp_path / "f.input.json").write_text("{}")
    assert vs.remove("f") == {"deleted": True}
    assert not (tmp_path / "f.input.json").exists()


def test_local_generation_deadline_reports_error_after_worker_stops():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from agent_bridge.events import AgentEventKind
    from tests.test_other_bridge_abort_targeting import HANG_GUARD_S, make_bridge, send

    async def run():
        bridge = make_bridge("local")
        bridge._request_timeout = 0.05
        redis = SimpleNamespace(xack=AsyncMock())
        stopped = threading.Event()
        release = threading.Event()

        class Client:
            def stream_raw(self, messages):
                try:
                    yield "first"
                    release.wait(HANG_GUARD_S)  # a long native token
                    yield "late"
                finally:
                    stopped.set()

        bridge._loader = SimpleNamespace(get_client=lambda *a, **kw: Client())
        with ThreadPoolExecutor(max_workers=1) as executor:
            bridge._executor = executor
            task = asyncio.create_task(send(bridge, "r", redis))
            async with asyncio.timeout(3):
                while not any(c.kwargs.get("kind") == AgentEventKind.ERROR
                              for c in bridge._publish_event.call_args_list):
                    await asyncio.sleep(0.01)
                    if bridge._session_queue.is_busy("bot:user") and not release.is_set():
                        # Deadline passed: the session stays fenced until the
                        # worker returns, then ERROR goes out.
                        await asyncio.sleep(0.1)
                        release.set()
            await task
        assert stopped.is_set()
        kinds = [c.kwargs.get("kind") for c in bridge._publish_event.call_args_list]
        assert AgentEventKind.ERROR in kinds and AgentEventKind.ASSISTANT_DONE not in kinds
        assert "timeout" in bridge._publish_event.call_args_list[-1].kwargs["text"]
    asyncio.run(run())
