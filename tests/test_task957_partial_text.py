"""TASK-957: replay reordered offsets against the startup persistence sink."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

from llm_bawt.service import api
from llm_bawt.service.upstream_status import LiveTurnPhases


def _live_sink():
    # The production sink is currently nested in the app lifespan. Extract and
    # execute the function *itself*, not a hand-written approximation, without
    # booting the app or starting its background workers.
    source = ast.parse(textwrap.dedent(inspect.getsource(api.lifespan)))
    handler = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                   and node.name == "_tool_event_sink")
    code = compile(ast.Module(body=[handler], type_ignores=[]), api.__file__, "exec")
    snapshots: list[str] = []
    writes: list[dict] = []

    def update_partial_response(**kw):
        writes.append(kw)
        snapshots.append(kw["response_text"])

    store = SimpleNamespace(update_partial_response=update_partial_response)
    scope = {"service": SimpleNamespace(_turn_log_store=store,
                                        _live_turn_phases=LiveTurnPhases()),
             "_partial_text": {}, "_partial_text_chunks": {},
             "_partial_reasoning": {}, "PartialText": api.PartialText}
    exec(code, scope)
    sink = scope["_tool_event_sink"]
    sink.writes = writes
    return sink, snapshots


def _reasoning(sink) -> str:
    return [w for w in sink.writes if w.get("reasoning") is not None][-1]["reasoning"]


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


# TASK-1273: reasoning_delta carries the same UTF-16 reasoning_offset, and the
# unified publish reorders it the same way. Appending in arrival order stored a
# permutation of the deltas in turn_logs.reasoning, which the live drawer seeds
# from on resume.

def test_out_of_order_reasoning_is_placed_by_offset():
    sink, _ = _live_sink()
    sink({"_type": "reasoning_delta", "turn_id": "byte-turn", "reasoning_offset": 6,
          "delta": "test "})
    sink({"_type": "reasoning_delta", "turn_id": "byte-turn", "reasoning_offset": 0,
          "delta": "I'm a "})
    sink({"_type": "reasoning_delta", "turn_id": "byte-turn", "reasoning_offset": 11,
          "delta": "plan"})
    assert _reasoning(sink) == "I'm a test plan"


def test_reasoning_gap_is_never_filled_with_later_text():
    sink, _ = _live_sink()
    sink({"_type": "reasoning_delta", "turn_id": "byte-turn", "reasoning_offset": 3,
          "delta": "def"})
    assert _reasoning(sink) == ""
    sink({"_type": "reasoning_delta", "turn_id": "byte-turn", "reasoning_offset": 0,
          "delta": "abc"})
    assert _reasoning(sink) == "abcdef"


def test_reasoning_permutation_reassembles_in_order_with_text_untouched():
    sink, _ = _live_sink()
    sink({"_type": "text_delta", "turn_id": "byte-turn", "text_offset": 0, "delta": "answer"})
    reasoning = "".join(f"chunk {i:03d}; " for i in range(120))
    chunks = [(o, reasoning[o:o + 11]) for o in range(0, len(reasoning), 11)]
    order = list(range(len(chunks)))
    # Deterministic local shuffle: adjacent pairs swapped, like the incident.
    for i in range(0, len(order) - 1, 2):
        order[i], order[i + 1] = order[i + 1], order[i]
    for i in order:
        offset, delta = chunks[i]
        sink({"_type": "reasoning_delta", "turn_id": "byte-turn",
              "reasoning_offset": offset, "delta": delta})
    assert _reasoning(sink) == reasoning
    assert sink.writes[-1]["response_text"] == "answer"


def test_reasoning_without_offset_still_appends():
    sink, _ = _live_sink()
    sink({"_type": "reasoning_delta", "turn_id": "old-turn", "delta": "ab"})
    sink({"_type": "reasoning_delta", "turn_id": "old-turn", "delta": "cd"})
    assert _reasoning(sink) == "abcd"
