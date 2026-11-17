"""Sequence child x-sensitive args are masked in approval listings and tool-call history (TASK-2353)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from llm_bawt.retrieval import tool_calls
from llm_bawt.retrieval.tool_call_scrub import arguments_view
from llm_bawt.service.routes.ops_approvals import _card

PARENT_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array"}}}
CHILD_SCHEMA = {"type": "object", "properties": {
    "service": {"type": "string"}, "pin": {"type": "string", "x-sensitive": True}}}
PLAIN_SCHEMA = {"type": "object", "properties": {"service": {"type": "string"}}}
STEPS = [{"operation": "fake-secret-child", "args": {"service": "app", "pin": "hunter2"}},
         {"operation": "fake-plain-child", "args": {"service": "voice"}}]


class _Conn:
    """Answers the schema lookups only; no live catalog or database."""

    schemas = {"fake-sequence": PARENT_SCHEMA, "fake-secret-child": CHILD_SCHEMA,
               "fake-plain-child": PLAIN_SCHEMA}

    def execute(self, _sql, params):
        value = self.schemas.get(params["slug"])
        return SimpleNamespace(scalar=lambda: json.dumps(value) if value else None)


def test_tool_call_scrub_masks_sequence_child_sensitive_arg():
    children = tool_calls._sequence_child_schemas(_Conn(), {"steps": STEPS})
    schema = {"properties": {"args": {**PARENT_SCHEMA, "properties": {
        "steps": {**PARENT_SCHEMA["properties"]["steps"], "prefixItems": children}}}}}
    raw = json.dumps({"operation": "fake-sequence", "args": {"steps": STEPS}})
    view = arguments_view(raw, field_chars=400, max_chars=2400, schema_json=json.dumps(schema))
    steps = view["value"]["args"]["steps"]
    assert steps[0]["args"] == {"service": "app", "pin": "***"}
    assert steps[1]["args"] == {"service": "voice"}
    assert "hunter2" not in json.dumps(view)


def test_unknown_child_operation_keeps_generic_scrubbing():
    steps = [{"operation": "missing-op", "args": {"service": "x"}}, "junk"]
    assert tool_calls._sequence_child_schemas(_Conn(), {"steps": steps}) == [{}, {}]
    assert tool_calls._sequence_child_schemas(_Conn(), {"steps": "nope"}) == []


def test_approval_card_lists_bound_child_display_args():
    bound = [{"snapshot": {"operation": {"slug": s["operation"]}, "input_args": s["args"]},
              "target": {"id": "0" * 64, "name": "x"},
              "display_args": {k: ("***" if k == "pin" else v) for k, v in s["args"].items()}}
             for s in STEPS]
    snapshot = {"operation": {"slug": "fake-sequence", "title": "Seq", "risk_level": "critical"},
                "resolved_args": {"steps": STEPS}, "schema": PARENT_SCHEMA,
                "sequence": {"steps": bound}}
    row = SimpleNamespace(id="r1", bot_id="vex", created_at=datetime.now(timezone.utc),
                          severity="high", operations_snapshot_json=json.dumps(snapshot))
    card = _card(row)
    assert card is not None and card.operation == "fake-sequence"
    assert card.args["steps"][0] == {"operation": "fake-secret-child",
                                     "args": {"service": "app", "pin": "***"}}
    assert card.args["steps"][1]["args"] == {"service": "voice"}
    assert "hunter2" not in card.model_dump_json()
