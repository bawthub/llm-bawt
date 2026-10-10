"""Streaming content-block state and event publication for Claude send turns."""

from __future__ import annotations

from agent_bridge.events import AgentEventKind

from .send_boundaries import separator_before_new_block


class StreamBlockState:
    """Track block boundaries across SDK frames for one send request."""

    def __init__(self, bridge, request_id: str, session_key: str, text_parts: list[str]):
        self.bridge = bridge
        self.request_id = request_id
        self.session_key = session_key
        self.text_parts = text_parts
        self.reasoning_tail = ""
        self.current_tool_name: str | None = None
        self.current_tool_input = ""

    def on_event(self, event: dict, seq: int) -> tuple[int, bool]:
        """Publish deltas and return the sequence and model-side-effect flag."""
        event_type = event.get("type", "")
        model_side_effects = False
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    model_side_effects = True
                    seq += 1
                    self.text_parts.append(text)
                    self.bridge._publish_event(
                        self.request_id, self.session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DELTA,
                        text=text,
                    )
            elif delta.get("type") == "thinking_delta":
                # Reasoning belongs in its own lane, never the final answer.
                thinking = delta.get("thinking", "")
                if thinking:
                    model_side_effects = True
                    seq += 1
                    self.reasoning_tail = thinking
                    self.bridge._publish_event(
                        self.request_id, self.session_key, seq,
                        kind=AgentEventKind.REASONING_DELTA,
                        text=thinking,
                    )
            elif delta.get("type") == "signature_delta":
                # Opaque reasoning signature — no display value.
                pass
            elif delta.get("type") == "input_json_delta":
                self.current_tool_input += delta.get("partial_json", "")

        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "thinking":
                separator = separator_before_new_block(self.reasoning_tail)
                if separator:
                    seq += 1
                    self.reasoning_tail += separator
                    self.bridge._publish_event(
                        self.request_id, self.session_key, seq,
                        kind=AgentEventKind.REASONING_DELTA,
                        text=separator,
                    )
            elif block.get("type") == "tool_use":
                model_side_effects = True
                self.current_tool_name = block.get("name", "unknown")
                self.current_tool_input = ""
            elif block.get("type") == "text":
                # Each new text block needs a real streamed separator so the
                # final message and tool text offsets agree with the UI.
                tail = self.text_parts[-1] if self.text_parts else ""
                if tail and not tail.endswith("\n"):
                    seq += 1
                    self.text_parts.append("\n\n")
                    self.bridge._publish_event(
                        self.request_id, self.session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DELTA,
                        text="\n\n",
                    )

        elif event_type == "content_block_stop":
            if self.current_tool_name:
                self.current_tool_name = None
                self.current_tool_input = ""

        return seq, model_side_effects
