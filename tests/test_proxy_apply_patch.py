"""TASK-1976: apply_patch on GPT routes translated to native Claude Code tools."""
import asyncio
import json
import tempfile
from types import SimpleNamespace as NS

import pytest

from claude_code_bridge.proxy import apply_patch as ap
from claude_code_bridge.proxy.request_context import ProxyRequestContext, custom_header_env
from claude_code_bridge.proxy.stream import TranslatorState, responses_to_anthropic_sse
from claude_code_bridge.proxy.translate import anthropic_to_responses

EDIT = {"type": "object", "properties": {"file_path": {"type": "string"},
        "old_string": {"type": "string"}, "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"}},
        "required": ["file_path", "old_string", "new_string"]}
WRITE = {"type": "object", "properties": {"file_path": {"type": "string"},
         "content": {"type": "string"}}, "required": ["file_path", "content"]}
READ = {"type": "object", "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"]}

W = tempfile.mkdtemp(prefix="apply-patch-")
FILES = {"a.py": "def f():\n    return 1\n\nx = 1\ny = 2\n", "old.txt": "bye\n", "a": "a\nb\n"}


@pytest.fixture(autouse=True)
def _workdir():
    import os
    for name, text in FILES.items():
        with open(os.path.join(W, name), "w", newline="") as fh:
            fh.write(text)
    yield


def apply_like_cli(text: str, edits: list[dict]) -> str:
    """Claude Code Edit semantics: old_string must occur exactly once."""
    for e in edits:
        assert text.count(e["old_string"]) == 1, e
        text = text.replace(e["old_string"], e["new_string"])
    return text


PATCH = f"""*** Begin Patch
*** Update File: {W}/a.py
@@ def f():
 def f():
-    return 1
+    return 2
@@
 x = 1
-y = 2
+y = 3
*** Add File: {W}/new.txt
+hello
+world
*** Delete File: {W}/old.txt
*** End Patch"""


def _body(messages=None, tools=(("Edit", EDIT), ("Write", WRITE), ("Read", READ))):
    return {"model": "m", "max_tokens": 10,
            "messages": messages or [{"role": "user", "content": "x"}],
            "tools": [{"name": n, "description": n, "input_schema": s} for n, s in tools]}


# ── parsing / conversion ──────────────────────────────────────────────────────

def test_patch_becomes_edit_write_and_rm_in_order():
    uses = ap.patch_to_tool_uses("call_1", PATCH, None)
    assert [(i, n) for i, n, _ in uses] == [
        ("call_1__ap0of4", "Edit"), ("call_1__ap1of4", "Edit"),
        ("call_1__ap2of4", "Write"), ("call_1__ap3of4", "Bash")]
    assert uses[0][2] == {"file_path": f"{W}/a.py", "old_string": "def f():\n    return 1\n",
                          "new_string": "def f():\n    return 2\n"}
    assert uses[1][2]["old_string"] == "x = 1\ny = 2\n"
    assert uses[2][2] == {"file_path": f"{W}/new.txt", "content": "hello\nworld\n"}
    assert uses[3][2]["command"] == f"rm -- {W}/old.txt"


def test_lenient_fences_crlf_and_missing_markers():
    text = f"```\n*** Update File: {W}/a\r\n@@\r\n a\r\n-b\r\n+c\r\n```"
    [(_, name, inp)] = ap.patch_to_tool_uses("c", text, None)
    assert name == "Edit" and inp["old_string"] == "a\nb\n" and inp["new_string"] == "a\nc\n"


def test_relative_paths_resolve_against_cli_cwd_or_fail():
    text = "*** Begin Patch\n*** Add File: src/x.py\n+a\n*** End Patch"
    [(_, _, inp)] = ap.patch_to_tool_uses("c", text, "/home/u/repo")
    assert inp["file_path"] == "/home/u/repo/src/x.py"
    [(tid, name, inp)] = ap.patch_to_tool_uses("c", text, None)
    assert tid == "c__aperr" and name == "Bash" and "absolute path" in inp["command"]


@pytest.mark.parametrize("text, needle", [
    ("*** Begin Patch\n*** Update File: /a\n*** Move to: /b\n@@\n-x\n+y\n*** End Patch", "Move to"),
    ("*** Begin Patch\n*** End Patch", "no file sections"),
    ("*** Begin Patch\n*** Update File: /a\n@@\n x\n*** End Patch", "no + or -"),
    ("*** Begin Patch\n*** Update File: {W}/a\n@@\n+only added\n*** End Patch", "no context"),
    ("*** Begin Patch\n*** Update File: /nonexistent/x\n@@\n-a\n+b\n*** End Patch", "Add File"),
    ("*** Begin Patch\n*** Update File: {W}/a\n@@\n-zzz\n+b\n*** End Patch", "re-read"),
    ("*** Begin Patch\n*** Update File: {W}/a\n@@ nope\n-a\n+b\n*** End Patch", "not found"),
    ("*** Begin Patch\n*** Add File: /a\nnot plus\n*** End Patch", "must start with"),
    ("random text", "outside a file section"),
])
def test_malformed_patch_is_a_recoverable_error_block(text, needle):
    [(tid, name, inp)] = ap.patch_to_tool_uses("c", text.replace("{W}", W), None)
    assert (tid, name) == ("c__aperr", "Bash")
    assert needle in inp["command"] and inp["command"].endswith("exit 1")


def test_error_block_really_prints_to_stderr_and_fails():
    import subprocess

    [(_, _, inp)] = ap.patch_to_tool_uses("c", "*** Add File: /a\nit's bad $(touch /tmp/x)\n", None)
    proc = subprocess.run(["bash", "-c", inp["command"]], capture_output=True, text=True)
    assert proc.returncode == 1 and proc.stdout == ""
    assert proc.stderr.startswith("apply_patch error: Add File lines must start with '+'")
    assert "$(touch /tmp/x)" in proc.stderr  # quoted, never executed


@pytest.mark.parametrize("content, patch, expected", [
    # Codex review: deleting a line removes the line, not just its text.
    ("keep\nfoo\nkeep2\n", "@@\n keep\n-foo\n keep2", "keep\nkeep2\n"),
    ("foo\nbar\n", "@@\n-foo", "bar\n"),
    # A prefix of a longer line never matches; the whole line does.
    ("foobar\nfoo\n", "@@\n-foo\n+bar", "foobar\nbar\n"),
    ("xfoo\nfoo\nzfoo\n", "@@\n-foo\n+bar", "xfoo\nbar\nzfoo\n"),
    # Last changed line of a larger hunk cannot prefix-match the next line.
    ("a\nb\nbc\n", "@@\n a\n-b\n+B", "a\nB\nbc\n"),
    # EOF without a trailing newline keeps having none.
    ("a\nlast", "@@\n a\n-last\n+LAST", "a\nLAST"),
    # CRLF files keep CRLF on new lines.
    ("a\r\nb\r\n", "@@\n a\n-b\n+B\n+C", "a\r\nB\r\nC\r\n"),
    # Duplicate lines: sections apply in order after the previous one / @@ anchor.
    ("def f():\n    x\ndef g():\n    x\n", "@@ def g():\n-    x\n+    y", "def f():\n    x\ndef g():\n    y\n"),
    ("x\ny\nx\ny\n", "@@\n-x\n+X\n@@\n-x\n+Z", "X\ny\nZ\ny\n"),
    # Trailing-whitespace drift in context still locates; file text is kept.
    ("a  \nb\n", "@@\n a\n-b\n+c", "a  \nc\n"),
])
def test_edits_apply_with_line_semantics(content, patch, expected):
    import os
    path = os.path.join(W, "t.txt")
    with open(path, "w", newline="") as fh:
        fh.write(content)
    uses = ap.patch_to_tool_uses("c", f"*** Update File: {path}\n{patch}\n", None)
    assert all(name == "Edit" for _, name, _ in uses), uses
    assert apply_like_cli(content, [inp for _, _, inp in uses]) == expected


def test_regenerated_patch_round_trips_to_the_same_blocks():
    uses = ap.patch_to_tool_uses("c", PATCH, None)
    blocks = [{"type": "tool_use", "id": i, "name": n, "input": inp} for i, n, inp in uses]
    again = ap.patch_to_tool_uses("c", ap.regenerate_patch(blocks), None)
    assert again == uses


def test_fold_results_success_and_partial_failure():
    uses = ap.patch_to_tool_uses("c", PATCH, None)
    blocks = [{"type": "tool_use", "id": i, "name": n, "input": inp} for i, n, inp in uses]
    ok = {b["id"]: {"content": "done"} for b in blocks}
    assert ap.fold_results(blocks, ok) == (
        f"Success. Updated the following files:\nM {W}/a.py\nA {W}/new.txt\nD {W}/old.txt")
    bad = dict(ok)
    bad[blocks[1]["id"]] = {"is_error": True,
                            "content": "<tool_use_error>String to replace not found</tool_use_error>"}
    text = ap.fold_results(blocks, bad)
    assert f"FAILED M {W}/a.py: <tool_use_error>String to replace not found" in text
    assert f"ok A {W}/new.txt" in text and "corrected patch" in text


def test_find_cwd_takes_the_latest_reported_directory():
    body = {"system": [{"type": "text", "text": " - Primary working directory: /home/u/dev\n"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text":
                          " - Primary working directory: /home/u/dev/repo (was /home/u/dev)\n"}]}]}
    assert ap.find_cwd(body) == "/home/u/dev/repo"
    assert ap.find_cwd({"messages": []}) is None


# ── request translation ──────────────────────────────────────────────────────

def test_tools_replaced_only_when_flag_on():
    off = anthropic_to_responses(_body(), "gpt")
    assert sorted(t["name"] for t in off["tools"]) == ["Edit", "Read", "Write"]
    on = anthropic_to_responses(_body(), "gpt", apply_patch="freeform")
    tools = {t["name"]: t for t in on["tools"]}
    assert sorted(tools) == ["Read", "apply_patch"]
    assert tools["apply_patch"]["type"] == "custom"
    assert tools["apply_patch"]["format"]["syntax"] == "lark"
    fn = anthropic_to_responses(_body(), "gpt", apply_patch="function")
    assert {t["name"]: t for t in fn["tools"]}["apply_patch"]["type"] == "function"
    # No edit tool advertised (e.g. read-only subagent) → no patch tool either.
    ro = anthropic_to_responses(_body(tools=(("Read", READ),)), "gpt", apply_patch="freeform")
    assert [t["name"] for t in ro["tools"]] == ["Read"]


def _history():
    uses = ap.patch_to_tool_uses("call_9", PATCH, None)
    return [
        {"role": "user", "content": "edit it"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Patching."},
            *[{"type": "tool_use", "id": i, "name": n, "input": inp} for i, n, inp in uses],
        ]},
        {"role": "user", "content": [
            *[{"type": "tool_result", "tool_use_id": i, "content": "ok"} for i, _, _ in uses],
            {"type": "text", "text": "thanks"},
        ]},
    ]


def test_history_folds_back_to_one_custom_call_and_output():
    body = _body(_history())
    payload = anthropic_to_responses(body, "gpt", apply_patch="freeform")
    kinds = [(i.get("type") or i.get("role")) for i in payload["input"]]
    assert kinds == ["user", "assistant", "custom_tool_call", "custom_tool_call_output", "user"]
    call = payload["input"][2]
    assert call["call_id"] == "call_9" and call["name"] == "apply_patch"
    assert ap.patch_to_tool_uses("call_9", call["input"], None) == ap.patch_to_tool_uses(
        "call_9", PATCH, None)
    assert payload["input"][3]["call_id"] == "call_9"
    assert payload["input"][3]["output"].startswith("Success.")
    # Byte-stable across repeated translation (prompt cache).
    assert anthropic_to_responses(body, "gpt", apply_patch="freeform") == payload
    # The SDK's body is not mutated.
    assert body["messages"][1]["content"][1]["name"] == "Edit"


def test_history_function_mode_and_legacy_native_edit():
    legacy = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Edit",
            "input": {"file_path": f"{W}/a", "old_string": "a\nb", "new_string": "a\nc"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": "updated"}]},
    ]
    payload = anthropic_to_responses(_body(legacy), "gpt", apply_patch="function")
    call, out = payload["input"][1], payload["input"][2]
    assert call["type"] == "function_call" and call["name"] == "apply_patch"
    assert json.loads(call["arguments"])["input"] == (
        f"*** Begin Patch\n*** Update File: {W}/a\n@@\n a\n-b\n+c\n*** End Patch")
    assert out == {"type": "function_call_output", "call_id": "toolu_1",
                   "output": f"Success. Updated the following files:\nM {W}/a"}
    # Flag off: legacy history untouched.
    plain = anthropic_to_responses(_body(legacy), "gpt")
    assert plain["input"][1]["name"] == "Edit"


def test_chatgpt_app_keeps_patch_tool_top_level():
    from claude_code_bridge.proxy.adapters.chatgpt_app import TOOL_NAMESPACE, ChatGPTAppAdapter

    payload = anthropic_to_responses(_body(_history()), "gpt", apply_patch="freeform")
    adapter = ChatGPTAppAdapter.__new__(ChatGPTAppAdapter)
    body = adapter.prepare_request(payload, None)
    assert [t["type"] for t in body["tools"]] == ["namespace", "custom"]
    assert [t["name"] for t in body["tools"][0]["tools"]] == ["Read"]
    assert TOOL_NAMESPACE == body["tools"][0]["name"]
    custom = [i for i in body["input"] if i.get("type") == "custom_tool_call"]
    assert custom and "namespace" not in custom[0]


# ── stream translation ───────────────────────────────────────────────────────

def _frames(chunks):
    out = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _events(patch, kind="custom_tool_call"):
    item = NS(type=kind, id="item", call_id="call_7", name="apply_patch")
    delta_t = ("response.custom_tool_call_input.delta" if kind == "custom_tool_call"
               else "response.function_call_arguments.delta")
    payload = patch if kind == "custom_tool_call" else json.dumps({"input": patch})
    done = (NS(type="response.custom_tool_call_input.done", item_id="item", input=payload)
            if kind == "custom_tool_call" else
            NS(type="response.function_call_arguments.done", item_id="item", arguments=payload))
    return [NS(type="response.output_item.added", item=item),
            *[NS(type=delta_t, item_id="item", delta=payload[i:i + 40])
              for i in range(0, len(payload), 40)],
            done,
            NS(type="response.output_item.done", item=item),
            NS(type="response.completed", response=NS(status="completed", usage=None))]


def _run(events, *, mode="freeform", state=None, cwd=None, fail_after=None):
    ctx = ProxyRequestContext(request_id="r", provider="chatgpt_app", apply_patch=mode,
                              patch_cwd=cwd)

    async def upstream():
        for n, event in enumerate(events):
            if fail_after is not None and n == fail_after:
                raise ConnectionError("stall")
            yield event

    async def run():
        return [c async for c in responses_to_anthropic_sse(
            upstream(), anthropic_model="fixture", request_context=ctx, state=state)]
    return asyncio.run(run())


@pytest.mark.parametrize("kind, mode", [("custom_tool_call", "freeform"),
                                        ("function_call", "function")])
def test_stream_emits_native_blocks_after_done(kind, mode):
    frames = _frames(_run(_events(PATCH, kind), mode=mode))
    starts = [f for f in frames if f["type"] == "content_block_start"]
    assert [(s["content_block"]["id"], s["content_block"]["name"]) for s in starts] == [
        ("call_7__ap0of4", "Edit"), ("call_7__ap1of4", "Edit"),
        ("call_7__ap2of4", "Write"), ("call_7__ap3of4", "Bash")]
    inputs = [json.loads(f["delta"]["partial_json"]) for f in frames
              if f["type"] == "content_block_delta"]
    assert inputs == [inp for _, _, inp in ap.patch_to_tool_uses("call_7", PATCH, None)]
    # Well-formed start→delta→stop per block, monotone indexes.
    seq = [(f["type"], f.get("index")) for f in frames if f["type"].startswith("content_block")]
    assert seq == [(t, i) for i in range(4) for t in
                   ("content_block_start", "content_block_delta", "content_block_stop")]
    assert frames[-2]["delta"]["stop_reason"] == "tool_use"


def test_stall_inside_patch_is_precommit():
    events = _events(PATCH)
    state = TranslatorState()
    with pytest.raises(ConnectionError):
        _run(events, state=state, fail_after=len(events) - 3)  # before .done
    assert state.tool_committed is False and state.open_block is None


def test_stream_unaffected_when_flag_off():
    item = NS(type="function_call", id="item", call_id="call_1", name="apply_patch")
    events = [NS(type="response.output_item.added", item=item),
              NS(type="response.function_call_arguments.done", item_id="item", arguments="{}"),
              NS(type="response.output_item.done", item=item),
              NS(type="response.completed", response=NS(status="completed", usage=None))]
    frames = _frames(_run(events, mode=None))
    start = next(f for f in frames if f["type"] == "content_block_start")
    assert start["content_block"]["name"] == "apply_patch"


def test_malformed_stream_patch_reaches_cli_as_error_bash():
    frames = _frames(_run(_events("*** Begin Patch\n*** Add File: rel.txt\n+x\n*** End Patch")))
    start = next(f for f in frames if f["type"] == "content_block_start")
    assert start["content_block"]["id"] == "call_7__aperr"
    delta = next(f for f in frames if f["type"] == "content_block_delta")
    assert "absolute path" in json.loads(delta["delta"]["partial_json"])["command"]


# ── plumbing ─────────────────────────────────────────────────────────────────

def test_mode_parses_and_rides_the_custom_header():
    assert ap.parse_mode(True) == "freeform" and ap.parse_mode("function") == "function"
    assert ap.parse_mode("1") == "freeform"
    assert ap.parse_mode(None) is None and ap.parse_mode(False) is None and ap.parse_mode("x") is None
    ctx = ProxyRequestContext(request_id="r", provider="chatgpt_app", apply_patch="function")
    headers = dict(line.split(": ", 1) for line in custom_header_env(ctx).splitlines())
    assert headers[ap.APPLY_PATCH_HEADER] == "function"
    assert ap.APPLY_PATCH_HEADER not in custom_header_env(
        ProxyRequestContext(request_id="r", provider="chatgpt_app"))


def test_catalog_serving_config_validates_apply_patch():
    from pydantic import ValidationError

    from llm_bawt.service.routes.model_catalog import EndpointWrite

    for good in (True, False, "freeform", "function"):
        assert EndpointWrite(serving_config={"responses_apply_patch": good}).serving_config == {
            "responses_apply_patch": good}
    for bad in ("true", 1, "patch"):
        with pytest.raises(ValidationError, match="responses_apply_patch"):
            EndpointWrite(serving_config={"responses_apply_patch": bad})


def test_init_bot_threads_apply_patch_only_when_set(monkeypatch):
    from tests.test_model_consolidation import _run_init_bot

    bot = NS(agent_backend="claude-code", agent_backend_config={}, default_model="sol")
    model = {"type": "claude-code", "model_id": "chatgpt_app/gpt-6.1-sol"}
    for value, expected in ((True, "freeform"), ("function", "function")):
        on = _run_init_bot(monkeypatch, bot, {"sol": {**model, "responses_apply_patch": value}})
        assert on.client._bot_config["responses_apply_patch"] == expected
    for value in (False, "true", None):
        off = _run_init_bot(monkeypatch, bot, {"sol": {**model, "responses_apply_patch": value}})
        assert "responses_apply_patch" not in off.client._bot_config


def test_command_and_send_request_round_trip():
    from claude_code_bridge.send_request import SendRequest
    from claude_code_bridge.send_stream import ClaudeStreamMixin
    from tests.test_openrouter_provider_routing import _FIELDS, _published_command

    assert _published_command(responses_apply_patch="freeform")["responses_apply_patch"] == "freeform"
    assert "responses_apply_patch" not in _published_command()
    req = SendRequest.from_fields({**_FIELDS, "responses_apply_patch": "freeform"})
    assert req.apply_patch == "freeform"
    assert SendRequest.from_fields(_FIELDS).apply_patch is None

    class Mixin(ClaudeStreamMixin):
        _proxy_base_url = "http://127.0.0.1:12345"

    env = Mixin()._build_sdk_env(
        use_proxy=True, model=req.model, subagent_model=None, force_refresh=False,
        bot_id="vex", session_key="vex:nick", thread_session_id="thread-1",
        request_id="request-1", apply_patch=req.apply_patch,
    )
    headers = dict(line.split(": ", 1) for line in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines())
    assert headers[ap.APPLY_PATCH_HEADER] == "freeform"
