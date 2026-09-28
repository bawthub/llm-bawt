"""TASK-957: replay reordered offsets against the startup persistence sink."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

from llm_bawt.service import api


def _live_sink():
    # The production sink is currently nested in the app lifespan. Extract and
    # execute the function *itself*, not a hand-written approximation, without
    # booting the app or starting its background workers.
    source = ast.parse(textwrap.dedent(inspect.getsource(api.lifespan)))
    handler = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                   and node.name == "_tool_event_sink")
    code = compile(ast.Module(body=[handler], type_ignores=[]), api.__file__, "exec")
    snapshots: list[str] = []
    store = SimpleNamespace(update_partial_response=lambda **kw: snapshots.append(kw["response_text"]))
    scope = {"service": SimpleNamespace(_turn_log_store=store),
             "_partial_text": {}, "_partial_text_chunks": {},
             "_partial_reasoning": {}, "PartialText": api.PartialText}
    exec(code, scope)
    return scope["_tool_event_sink"], snapshots


def test_out_of_order_chunks_restore_full_text_without_truncation():
    sink, snapshots = _live_sink()
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 3, "delta": "def"})
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 0, "delta": "abc"})
    assert snapshots[-1] == "abcdef"
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 3, "delta": "def"})
    assert snapshots[-1] == "abcdef"


def test_gap_does_not_fabricate_text_or_discard_later_chunk():
    sink, snapshots = _live_sink()
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 6, "delta": "ghi"})
    assert snapshots[-1] == ""
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 0, "delta": "abc"})
    assert snapshots[-1] == "abc"
    sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": 3, "delta": "def"})
    assert snapshots[-1] == "abcdefghi"


def test_incident_offset_order_reconstructs_all_1400_characters():
    sink, snapshots = _live_sink()
    offsets = [520, 250, 621, 333, 0, 420, 84, 169,
               704, 796, 878, 965, 1048, 1130, 1301, 1216]
    text = "".join(chr(65 + i % 26) for i in range(1400))
    boundaries = sorted(set(offsets + [len(text)]))
    for offset in offsets:
        next_offset = boundaries[boundaries.index(offset) + 1]
        sink({"_type": "text_delta", "turn_id": "caid-turn", "text_offset": offset,
              "delta": text[offset:next_offset]})
    assert snapshots[-1] == text


def test_utf16_offsets_count_non_bmp_chars():
    sink, snapshots = _live_sink()
    sink({"_type": "text_delta", "turn_id": "emoji-turn", "text_offset": 2, "delta": "ok"})
    sink({"_type": "text_delta", "turn_id": "emoji-turn", "text_offset": 0, "delta": "😀"})
    assert snapshots[-1] == "😀ok"
