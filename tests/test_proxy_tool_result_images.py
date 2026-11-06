"""Tool-result images retain their originating call through Responses translation."""

from copy import deepcopy

import pytest

from claude_code_bridge.proxy.translate import anthropic_to_responses


def _image(source: str, label: str = "A") -> dict:
    if source == "url":
        image_source = {"type": "url", "url": f"https://example.invalid/{label}.png"}
    else:
        image_source = {"type": "base64", "media_type": "image/png", "data": label}
    return {"type": "image", "source": image_source}


def _body(first: list[dict], second: list[dict]) -> dict:
    return {"messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": call, "name": "Read", "input": {}}
            for call in ("before", "after")
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "before", "content": first},
            {"type": "tool_result", "tool_use_id": "after", "content": second},
            {"type": "text", "text": "Compare the evidence."},
        ]},
    ]}


@pytest.mark.parametrize("source", ["url", "base64"])
def test_mixed_image_and_text_results_keep_the_image_owner(source: str) -> None:
    """Moving one image to another result must change the upstream request."""
    before = [{"type": "text", "text": "Before evidence"}, _image(source)]
    after = [{"type": "text", "text": "After evidence"}]
    body = _body(before, after)
    alternate = deepcopy(body)
    image = alternate["messages"][1]["content"][0]["content"].pop()
    alternate["messages"][1]["content"][1]["content"].append(image)

    original = anthropic_to_responses(body, "gpt-test")
    moved = anthropic_to_responses(alternate, "gpt-test")

    assert original != moved
    assert original["input"][:4] == moved["input"][:4]
    image_message = original["input"][4]
    assert image_message["content"][0]["text"] == "Image(s) returned by tool_call_id=before:"
    assert moved["input"][4]["content"][0]["text"] == "Image(s) returned by tool_call_id=after:"
    expected_url = "https://example.invalid/A.png" if source == "url" else "data:image/png;base64,A"
    assert image_message["content"][1] == {
        "type": "input_image", "image_url": expected_url, "detail": "auto",
    }
    assert original["input"][-1]["content"] == [
        {"type": "input_text", "text": "Compare the evidence."},
    ]


def test_each_image_result_gets_its_own_labeled_message_after_all_outputs() -> None:
    body = _body(
        [{"type": "text", "text": "Before"}, _image("url", "A")],
        [_image("url", "B")],
    )
    items = anthropic_to_responses(body, "gpt-test")["input"]

    assert [item["type"] for item in items[:4]] == [
        "function_call", "function_call", "function_call_output", "function_call_output",
    ]
    assert items[2]["output"] == "Before"
    assert items[3]["output"] == "[image returned by tool — shown in the following message]"
    assert [item["content"][0]["text"] for item in items[4:6]] == [
        "Image(s) returned by tool_call_id=before:",
        "Image(s) returned by tool_call_id=after:",
    ]
    assert [item["content"][1]["image_url"] for item in items[4:6]] == [
        "https://example.invalid/A.png", "https://example.invalid/B.png",
    ]


def test_two_images_in_one_result_do_not_collide_with_split_ownership() -> None:
    """The original preflight replay also retains per-result boundaries."""
    body = _body(
        [{"type": "text", "text": "Before"}, _image("url", "A"), _image("url", "B")],
        [{"type": "text", "text": "After"}],
    )
    alternate = deepcopy(body)
    image = alternate["messages"][1]["content"][0]["content"].pop()
    alternate["messages"][1]["content"][1]["content"].append(image)

    original = anthropic_to_responses(body, "gpt-test")
    moved = anthropic_to_responses(alternate, "gpt-test")

    assert original != moved
    assert len(original["input"][4]["content"]) == 3
    assert len(moved["input"][4]["content"]) == 2
    assert moved["input"][5]["content"][0]["text"] == "Image(s) returned by tool_call_id=after:"


def test_text_only_results_do_not_gain_image_messages() -> None:
    body = _body(
        [{"type": "text", "text": "Before"}],
        [{"type": "text", "text": "After"}],
    )
    items = anthropic_to_responses(body, "gpt-test")["input"]

    assert len(items) == 5
    assert items[2]["output"] == "Before"
    assert items[3]["output"] == "After"
    assert items[4]["content"] == [{"type": "input_text", "text": "Compare the evidence."}]
