"""A non-executable, model-visible result for one withheld incomplete tool.

This is not a retry: the upstream sample and original tool are discarded. The
CLI returns its ordinary unknown-tool error for the reserved carrier, letting
the model generate a new call. Completed/parallel tool batches stay fail-closed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

CARRIER_NAME = "ToolArgumentsIncomplete"


@dataclass
class ToolArgumentGenerationError(RuntimeError):
    tool_name: str
    held_chars: int
    cause: str
    incomplete: bool = True

    def __str__(self) -> str:
        return (f"Upstream tool argument generation failed: {self.cause}; "
                f"tool={self.tool_name} held_arg_chars={self.held_chars}; tool did not run")


def carrier_allowed(body: dict) -> bool:
    """One substitution per model conversation turn; never shadow a real tool."""
    if any(tool.get("name") == CARRIER_NAME for tool in body.get("tools", [])):
        return False
    for message in reversed(body.get("messages", [])):
        content = message.get("content", [])
        blocks = content if isinstance(content, list) else []
        if any(block.get("type") == "tool_use" and block.get("name") == CARRIER_NAME
               for block in blocks if isinstance(block, dict)):
            return False
        if message.get("role") == "user" and not any(
            block.get("type") == "tool_result" for block in blocks if isinstance(block, dict)
        ):
            break
    return True


def recovery_frames(error: ToolArgumentGenerationError, *, index: int, call_id: str,
                    output_tokens: int = 0) -> Iterator[bytes]:
    """Replace only the withheld suffix, preserving already-forwarded indexes."""
    message = (f"{error}. None of the 1 withheld tool calls ran. "
               "The next unavailable tool is a recovery notice, not an executable call. "
               "After its error result, generate a NEW complete call to the original tool. "
               "Do not assume any side effects or repeat already completed calls.")
    events = [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": message}},
        {"type": "content_block_stop", "index": index},
        {"type": "content_block_start", "index": index + 1, "content_block": {
            "type": "tool_use", "id": call_id, "name": CARRIER_NAME, "input": {},
        }},
        {"type": "content_block_delta", "index": index + 1,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": index + 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    ]
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
