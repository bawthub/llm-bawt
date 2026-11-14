"""TASK-2260 regressions: integrations, media, tools, core utilities and CLI."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.tools.executor import ToolExecutor, parse_date_param
from llm_bawt.tools.parser import ToolCall


def _executor(**kwargs) -> ToolExecutor:
    kwargs.setdefault("user_id", "nick")
    kwargs.setdefault("bot_id", "fixture-bot")
    return ToolExecutor(**kwargs)


# --- HA MCP bounded bridge (item 0) ----------------------------------------

@pytest.mark.parametrize("with_loop", [False, True])
def test_ha_run_bounded_times_out_instead_of_returning_none(with_loop):
    from llm_bawt.integrations.ha_mcp.client import run_bounded

    async def slow():
        await asyncio.sleep(5)
        return "done"

    def call():
        return run_bounded(slow(), 0.05)

    if not with_loop:
        with pytest.raises(TimeoutError):
            call()
        return

    async def inside_loop():
        call()

    with pytest.raises(TimeoutError):
        asyncio.run(inside_loop())


# --- web fetch budget (item 4) ---------------------------------------------

@pytest.mark.parametrize("title_len,body_len", [(200, 1000), (10, 1000), (0, 5)])
def test_web_fetch_output_never_exceeds_budget(title_len, body_len):
    from llm_bawt.integrations.web_fetch.client import _bound_output

    header = "Title: " + "t" * title_len + "\n---\n"
    out = _bound_output(header, "m" * body_len, 100)
    assert len(out) <= 100
    if len(header) + body_len <= 100:
        assert out == header + "m" * body_len


# --- image normalization (items 5, 6) --------------------------------------

def _normalize(img, fmt, **save):
    from llm_bawt.media.asset_kinds import _normalize_to_original

    buf = io.BytesIO()
    img.save(buf, fmt, **save)
    _, normalized, width, height = _normalize_to_original(buf.getvalue())
    return normalized, width, height


def test_exif_orientation_applied_before_strip():
    from PIL import Image

    img = Image.new("RGB", (6, 4), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90
    _, width, height = _normalize(img, "JPEG", exif=exif.tobytes())
    assert (width, height) == (4, 6)


def test_palette_transparency_is_preserved():
    from PIL import Image

    img = Image.new("P", (4, 4), 0)
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    img.info["transparency"] = 0
    normalized, _, _ = _normalize(img, "PNG", transparency=0)
    assert normalized.mode == "RGBA"
    assert normalized.getpixel((0, 0))[3] == 0


# --- media dedup race (item 7) ---------------------------------------------

def test_conflicting_insert_returns_the_winning_row():
    """A concurrent insert of the same bytes returns the existing row, not IntegrityError."""
    from llm_bawt.media import assets

    executed = []
    winner = {"id": "ma_first", "sha256": "a" * 64}

    class Result:
        def mappings(self):
            return self

        def first(self):
            return None  # ON CONFLICT DO NOTHING: no row returned

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            executed.append(str(sql))
            return Result()

        def commit(self):
            pass

    store = assets.MediaAssetStore.__new__(assets.MediaAssetStore)
    store.engine = SimpleNamespace(connect=lambda: Conn())
    lookups = iter([None, winner])  # pre-check misses, post-conflict lookup hits
    store.get_by_sha256 = lambda sha: next(lookups)
    row = store.insert(sha256="a" * 64, mime_type="image/webp", original_mime_type="image/png",
                       size_bytes=1, width=1, height=1, source="chat_upload", owner_user_id="nick")
    assert row == winner
    assert "ON CONFLICT (sha256) WHERE storage_key IS NULL DO NOTHING" in executed[0]


# --- xAI 429 (item 8) ------------------------------------------------------

def test_grok_poll_429_raises_for_transient_retry():
    from llm_bawt.media.clients import grok_media

    client = grok_media.GrokMediaClient.__new__(grok_media.GrokMediaClient)
    client._client = httpx.AsyncClient(
        base_url="https://x.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down")),
    )
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.poll_status("job"))


# --- num_outputs (item 9) --------------------------------------------------

def test_multi_output_rejected_before_provider_call():
    from llm_bawt.media.generation_service import MediaGenerationService

    svc = MediaGenerationService.__new__(MediaGenerationService)
    svc.registry = SimpleNamespace(create_client=lambda *a: pytest.fail("provider called"))
    with pytest.raises(ValueError):
        asyncio.run(svc.generate_image(prompt="x", num_outputs=3))


# --- S3 body read (item 10) ------------------------------------------------

def test_s3_body_read_failure_maps_to_unavailable():
    from botocore.exceptions import ReadTimeoutError

    from llm_bawt.media.object_store import BlobBackendUnavailable, _read_body

    closed = []

    class Body:
        def read(self):
            raise ReadTimeoutError(endpoint_url="http://s3")

        def close(self):
            closed.append(True)

    with pytest.raises(BlobBackendUnavailable):
        _read_body({"Body": Body()}, "k")
    assert closed == [True]


# --- uploads without metadata DB (item 11) ---------------------------------

def test_upload_refused_when_metadata_db_unavailable(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.media.store import MediaStore
    from llm_bawt.service.routes import uploads

    store = MediaStore(root=tmp_path, db=None)
    monkeypatch.setattr(uploads, "_store", lambda: store)
    app = FastAPI()
    app.include_router(uploads.router)
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    resp = TestClient(app).post(
        "/v1/uploads", headers={"X-Entity-Id": "user-1"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 503
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


# --- tool executor (items 12, 14, 15, 18, 19, 20) --------------------------

@pytest.mark.parametrize("limit,expected_ok", [(1, 1), (3, 3), (0, 5)])
def test_tool_call_limit_allows_exactly_the_configured_count(limit, expected_ok):
    ex = _executor(config=SimpleNamespace(MAX_TOOL_CALLS_PER_TURN=limit))
    results = [ex.execute(ToolCall(name="time", arguments={}, raw_text="")) for _ in range(5)]
    ok = [r for r in results if "Too many tool calls" not in r]
    assert len(ok) == expected_ok


def test_explicit_historical_dates_are_kept():
    since = parse_date_param("2020-01-01")
    until = parse_date_param("2020-02-01")
    assert since == datetime(2020, 1, 1).timestamp()
    assert until == datetime(2020, 2, 1).timestamp()


def test_search_honours_n_results_alias_from_react():
    seen = {}

    class Search:
        def search(self, query, max_results=5, **kw):
            seen["max"] = max_results
            return []

    ex = _executor(search_client=Search())
    ex.execute(ToolCall(name="search", arguments={"query": "tea", "n_results": 2}, raw_text=""))
    assert seen["max"] == 2


def test_react_list_models_keeps_list_action():
    from llm_bawt.tools.definitions import normalize_legacy_tool_call
    from llm_bawt.tools.formats.react import _normalize_tool_name

    for alias in ("list_models", "show_models", "available_models"):
        name, args = normalize_legacy_tool_call(_normalize_tool_name(alias), {})
        assert (name, args.get("action")) == ("model", "list")


class FakeProfiles:
    def __init__(self):
        self.rows = {}
        self.updates = []

    def set_attribute(self, entity_type, entity_id, category, key, value, **kw):
        self.rows[(str(entity_type), entity_id, category, key)] = value

    def get_attribute(self, entity_type, entity_id, category, key):
        return self.rows.get((str(entity_type), entity_id, category, key))

    def delete_attribute(self, entity_type, entity_id, category, key):
        return self.rows.pop((str(entity_type), entity_id, category, key), None) is not None

    def update_attribute_by_id(self, attribute_id, **kw):
        self.updates.append(kw)
        return None


def test_self_delete_normalizes_communication_category():
    profiles = FakeProfiles()
    ex = _executor(profile_manager=profiles)
    ex.execute(ToolCall(name="self", arguments={"action": "set", "category": "communication",
                                                "key": "tone", "value": "dry"}, raw_text=""))
    assert len(profiles.rows) == 1
    ex.execute(ToolCall(name="self", arguments={"action": "delete", "category": "communication",
                                                "key": "tone"}, raw_text=""))
    assert profiles.rows == {}


def test_profile_update_by_id_is_bound_to_current_user():
    from llm_bawt.profiles import EntityType

    profiles = FakeProfiles()
    ex = _executor(profile_manager=profiles)
    out = ex.execute(ToolCall(name="profile", arguments={"action": "update", "attribute_id": 7,
                                                         "value": "x"}, raw_text=""))
    assert profiles.updates[0]["entity_type"] == EntityType.USER
    assert profiles.updates[0]["entity_id"] == "nick"
    assert "not found" in out


def test_home_set_never_auto_picks_among_multiple_matches():
    calls = []

    class Home:
        def set(self, entity, state, brightness=None):
            calls.append(entity)
            return "Error: multiple matches for light.kitchen"

        def query(self, pattern, domain=None):
            return "light.kitchen_1\nlight.kitchen_2"

    ex = _executor(home_client=Home())
    out = ex.execute(ToolCall(name="home", arguments={"action": "set", "entity": "light.kitchen",
                                                      "state": "off"}, raw_text=""))
    assert calls == ["light.kitchen"]
    assert "nothing was changed" in out


# --- tool loop returns every result (item 13) ------------------------------

def test_over_budget_batch_answers_every_tool_call_id():
    from llm_bawt.tools.loop import ToolLoop

    loop = ToolLoop(user_id="nick", config=SimpleNamespace(MAX_TOOL_CALLS_PER_TURN=1), tool_format="native")
    handler = SimpleNamespace(format_result=lambda name, result, tool_call_id=None, error=None: {
        "role": "tool", "tool_call_id": tool_call_id, "content": error or result})
    calls = [SimpleNamespace(name="time", arguments={}, raw_text="", tool_call_id=f"c{i}") for i in range(4)]
    messages, _ = loop._execute_tools(calls, handler)
    assert [getattr(m, "tool_call_id", None) for m in messages] == ["c0", "c1", "c2", "c3"]


# --- native schema 'any' (item 17) -----------------------------------------

def test_any_param_schema_accepts_scalars_lists_and_objects():
    from llm_bawt.tools.formats.native_openai import NativeOpenAIFormatHandler

    tool = SimpleNamespace(name="profile", description="d", parameters=[
        SimpleNamespace(name="value", type="any", description="v", default=None, required=True)])
    prop = NativeOpenAIFormatHandler().get_tools_schema([tool])[0]["function"]["parameters"]["properties"]["value"]
    assert set(prop["type"]) == {"string", "number", "boolean", "array", "object"}


# --- tool-result classifier (item 22) --------------------------------------

def test_timestamped_tool_results_are_evidence():
    from llm_bawt.utils.history import _is_tool_result_system

    msg = SimpleNamespace(role="system", content="[Tool Results @ 2026-11-14 13:00]\nok")
    assert _is_tool_result_system(msg)
    assert not _is_tool_result_system(SimpleNamespace(role="system", content="You are a bot"))


# --- model switch scoping (items 23, 24) -----------------------------------

@pytest.fixture
def lifecycle(monkeypatch):
    from llm_bawt.core import model_lifecycle as ml

    monkeypatch.setattr(ml.ModelLifecycleManager, "_instance", None)
    mgr = ml.ModelLifecycleManager(config=None)
    monkeypatch.setattr(mgr, "_normalize_alias", lambda a: a)
    mgr._current_model_alias = "A"
    yield mgr
    ml.ModelLifecycleManager._instance = None


def test_selecting_current_model_cancels_pending_switch(lifecycle):
    # Unscoped (admin) request keeps the global same-model shortcut.
    lifecycle.switch_model("B")
    ok, msg = lifecycle.switch_model("A")
    assert ok and "Already using" in msg
    assert lifecycle.clear_pending_switch() is None


def test_scoped_reselect_replaces_owner_pending_switch(lifecycle):
    owner = ("bot", "nick")
    lifecycle.switch_model("B", owner=owner)
    lifecycle.switch_model("A", owner=owner)
    assert lifecycle.clear_pending_switch(owner=owner) == "A"


def test_scoped_switch_ignores_other_owners_global_model(lifecycle):
    # Owner A runs cached model A; owner B registered B last (global alias).
    lifecycle._current_model_alias = "B"
    owner_a = ("bot-a", "nick")
    ok, msg = lifecycle.switch_model("B", owner=owner_a)
    assert ok and "Already using" not in msg
    assert lifecycle.clear_pending_switch(owner=owner_a) == "B"


def test_pending_switch_only_reaches_its_owner(lifecycle):
    lifecycle.switch_model("B", owner=("bot-a", "nick"))
    assert lifecycle.clear_pending_switch(owner=("bot-c", "nick")) is None
    assert lifecycle.clear_pending_switch(owner=("BOT-A", "nick")) == "B"


# --- MCP status probe (item 25) --------------------------------------------

@pytest.mark.parametrize("body,expected", [
    ({"jsonrpc": "2.0", "id": "s", "error": {"code": -32601, "message": "nope"}}, "error"),
    ({"jsonrpc": "2.0", "id": "s", "result": {"isError": True}}, "error"),
    ({"jsonrpc": "2.0", "id": "s", "result": {"protocolVersion": "x"}}, "up"),
])
def test_mcp_probe_checks_application_result(monkeypatch, body, expected):
    from llm_bawt.core import status

    sent = {}

    def post(url, json=None, **kw):
        sent.update(json)
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "post", post)
    info = status._collect_mcp_info(SimpleNamespace(MCP_SERVER_URL="http://mcp"), "bot")
    assert info.status == expected
    assert sent["method"] == "initialize"


# --- DB URL encoding (item 26) ---------------------------------------------

def test_postgres_url_roundtrips_reserved_password_characters():
    from sqlalchemy.engine import make_url

    from llm_bawt.utils.db import build_postgres_url

    pw = "a b+c/@:%"
    url = build_postgres_url(SimpleNamespace(POSTGRES_HOST="h", POSTGRES_PORT=5432,
                                             POSTGRES_USER="u@x", POSTGRES_PASSWORD=pw,
                                             POSTGRES_DATABASE="d"))
    parsed = make_url(url)
    assert parsed.password == pw and parsed.username == "u@x"


# --- dotenv append (item 27) -----------------------------------------------

def test_dotenv_append_keeps_existing_line_separate(tmp_path):
    from llm_bawt.utils.env import set_env_value

    env = tmp_path / ".env"
    env.write_text("EXISTING=value", encoding="utf-8")
    assert set_env_value(env, "NEW", "other")
    assert env.read_text(encoding="utf-8").splitlines() == ["EXISTING=value", "NEW=other"]


# --- VRAM (items 31, 32) ---------------------------------------------------

def test_vram_free_uses_device_wide_mem_get_info(monkeypatch):
    import sys

    from llm_bawt.utils import vram

    gib = 1024 ** 3
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0,
        get_device_properties=lambda d: SimpleNamespace(total_memory=24 * gib, name="GPU"),
        memory_reserved=lambda d: 0, mem_get_info=lambda d: (2 * gib, 24 * gib)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    info = vram._detect_vram_torch()
    assert info.free_bytes == 2 * gib


def test_native_context_cap_applies_on_fallback(monkeypatch):
    from llm_bawt.utils import vram

    monkeypatch.setattr(vram, "detect_vram", lambda: None)
    result = vram.auto_size_context_window(
        {"type": "gguf", "native_context_limit": 4096}, global_n_ctx=32768)
    assert result.context_window == 4096


# --- CLI (items 28-30, 34-38) ----------------------------------------------

def test_settings_yaml_keeps_nested_values_under_their_key():
    import yaml

    from llm_bawt.cli.bot_editor import _build_settings_yaml

    settings = {"summarization_job": {"model": None, "trigger_tokens": 6000},
                "temperature": 0.8, "tags": ["a", "b"], "note": "line1\nline2"}
    assert yaml.safe_load(_build_settings_yaml(settings))["settings"] == settings


def test_secret_fields_are_masked_by_name():
    from llm_bawt.utils.config import is_secret_config_field

    for name in ("NEWSAPI_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY", "REDDIT_CLIENT_SECRET",
                 "TALK_PROVISIONER_TOKEN", "POSTGRES_PASSWORD"):
        assert is_secret_config_field(name)
    assert not is_secret_config_field("MAX_TOKENS")


def test_multiline_input_survives_eof(monkeypatch):
    from llm_bawt.utils.input_handler import MultilineInputHandler

    lines = iter(["first", "second"])

    def fake_input(*a):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    handler = MultilineInputHandler.__new__(MultilineInputHandler)
    handler.console = SimpleNamespace(print=lambda *a, **k: None)
    assert handler._get_multiline_input() == "first\nsecond"


@pytest.mark.parametrize("value", ["2026-11-14T05:00:00Z", "2026-11-14T05:00:00+02:00", "2026-11-14T05:00:00"])
def test_job_timestamps_are_timezone_aware(value):
    from llm_bawt.cli.display_cmd import _aware_utc

    assert _aware_utc(value).tzinfo is not None


@pytest.mark.parametrize("args", ["{}", {}])
def test_empty_tool_args_render_without_stopiteration(args):
    from llm_bawt.cli.tool_render import _format_tool_call, _summarize_tool_args

    assert _format_tool_call("NoArgs", args) == "NoArgs"
    _summarize_tool_args(args)


def test_setup_rejects_invalid_values_before_persisting(tmp_path, monkeypatch):
    from llm_bawt.cli import config_setup
    from llm_bawt.utils.config import Config

    config = Config()
    env = tmp_path / ".env"
    port = SimpleNamespace(key="POSTGRES_PORT", env_only=False)
    assert not config_setup._save_value(config, port, "abc", env, "LLM_BAWT_")
    assert not env.exists()
    flag = SimpleNamespace(key="USE_SERVICE", env_only=False)
    monkeypatch.delenv("LLM_BAWT_USE_SERVICE", raising=False)
    assert config_setup._save_value(config, flag, "on", env, "LLM_BAWT_")
    assert config.USE_SERVICE is True


# --- CLI bot editor writes (items 29, 30) ----------------------------------

class FakeServiceClient:
    def __init__(self, upsert_result=None):
        self.upserts = []
        self.upsert_result = upsert_result

    def is_available(self):
        return True

    def get_settings(self, scope_type="bot", scope_id=None):
        return {"settings": []}

    def upsert_bot_profile(self, slug, payload):
        self.upserts.append(payload)
        return self.upsert_result

    def set_setting(self, *a):
        return {"ok": True}

    def delete_setting(self, *a, **k):
        return {"ok": True}


@pytest.fixture
def editor(monkeypatch):
    import llm_bawt.bots as bots
    from llm_bawt.cli import bot_editor

    bot = SimpleNamespace(
        name="Fixture", color="#fff", description="d", system_prompt="p", requires_memory=False,
        voice_optimized=False, tts_mode=False, include_summaries=False, include_in_global_search=False,
        uses_tools=False, uses_search=False, uses_home_assistant=False, default_model=None,
        default_voice="voice-1", bot_type="chat", agent_backend=None, agent_backend_config=None)
    monkeypatch.setattr(bots, "get_bot", lambda slug: bot)
    monkeypatch.setattr(bots, "get_raw_bot_data", lambda slug: {})
    monkeypatch.setattr(bots, "get_bot_settings_template", lambda: {"temperature": 0.7})
    monkeypatch.setattr(bots, "invalidate_bots_cache", lambda: None, raising=False)
    monkeypatch.setattr(bot_editor, "_notify_service_reload", lambda config: None)
    monkeypatch.setattr(bot_editor.sys.stdin, "isatty", lambda: False, raising=False)
    edits = {}

    def open_editor(text, suffix):
        import yaml

        data = yaml.safe_load(text)
        data.update(edits)
        return data

    monkeypatch.setattr(bot_editor, "_open_yaml_editor", open_editor)
    return bot_editor, edits


def test_unchanged_bot_edit_writes_nothing(editor, monkeypatch):
    bot_editor, _ = editor
    client = FakeServiceClient(upsert_result={"ok": True})
    monkeypatch.setattr(bot_editor, "_get_service_client", lambda config: client)
    assert bot_editor.edit_bot_yaml(SimpleNamespace(), "fixture") is True
    assert client.upserts == []


def test_name_edit_keeps_default_voice_and_reports_failed_save(editor, monkeypatch):
    bot_editor, edits = editor
    edits["name"] = "Renamed"
    client = FakeServiceClient(upsert_result=None)  # service write failed
    monkeypatch.setattr(bot_editor, "_get_service_client", lambda config: client)
    assert bot_editor.edit_bot_yaml(SimpleNamespace(), "fixture") is False
    assert client.upserts[0]["default_voice"] == "voice-1"
