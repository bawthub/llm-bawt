"""Regression tests for strip_tool_protocol_leakage (TASK-2283)."""

from __future__ import annotations

from llm_bawt.shared.output_sanitizer import strip_tool_protocol_leakage as strip


def test_indented_code_survives():
    text = "Here:\n\n```python\ndef f():\n    if x:\n        return 1\n```"
    assert strip(text) == text


def test_nested_list_and_first_line_indent_survive():
    text = "    indented first line\n- a\n  - b\n    - c"
    assert strip("\n\n" + text + "  \n") == text


def test_nested_tool_uses_array_fully_removed():
    assert strip('{"tool_uses":[{"a":[1]},{"b":2}]} done') == "done"


def test_tool_uses_with_brackets_in_strings_removed():
    text = 'before {"tool_uses": [{"q": "]} tricky"}]} after'
    assert strip(text) == "before  after"


def test_multiple_tool_uses_objects_removed():
    text = '{"tool_uses":[{"a":1}]} mid {"tool_uses":[{"b":[2]}]} end'
    assert strip(text) == "mid  end"


def test_malformed_tool_uses_falls_back():
    assert strip('{"tool_uses":[{"a":1}]} oops}') == "oops}"
    assert strip('text {"tool_uses": broken') == 'text {"tool_uses": broken'


def test_existing_cleanup_still_applies():
    text = "Hi <tool_call>{}</tool_call>there\n\n\n\nbye"
    assert strip(text) == "Hi there\n\nbye"
