"""Generic Responses API client.

Uses the OpenAI SDK's Responses API (/v1/responses) which works with any
OpenAI-compatible provider that supports it (OpenAI, xAI, etc.) via base_url.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, List

from openai import OpenAI, OpenAIError

from .base import LLMClient
from .usage_hook import report_usage
from ..models.message import Message
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ResponsesClient(LLMClient):
    """Client using the OpenAI Responses API (/v1/responses).

    Generic client that works with any provider supporting the Responses API
    format via base_url (OpenAI, xAI/Grok, etc.).
    """

    SUPPORTS_STREAMING = True

    def __init__(
        self,
        model: str,
        config: Config,
        base_url: str | None = None,
        api_key: str | None = None,
        model_definition: dict | None = None,
    ):
        super().__init__(model, config, model_definition=model_definition)
        self.base_url = base_url
        if api_key is None:
            # DB-first (CredentialStore via the `openai-api` adapter); env is a
            # deprecated fallback (TASK-825). Subclasses (GrokClient) pass their
            # own resolved key explicitly.
            from ..service.providers.api_key import resolve_api_key

            api_key = resolve_api_key(config, "openai-api", env_vars=("OPENAI_API_KEY",))
        self.api_key = api_key

        if not self.api_key and not base_url:
            raise ValueError(
                "OpenAI API key not found. Connect the OpenAI API provider in "
                "the UI (Settings → Providers), or set OPENAI_API_KEY as a "
                "legacy fallback."
            )

        effective_key = self.api_key or "not-needed"
        if base_url:
            self.client = OpenAI(api_key=effective_key, base_url=base_url)
            logger.debug("Responses client using endpoint: %s", base_url)
        else:
            self.client = OpenAI(api_key=effective_key)

    def supports_native_tools(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Message -> Responses API input conversion
    # ------------------------------------------------------------------

    def _prepare_input(self, messages: List[Message], *, log_input: bool = True) -> tuple[list[dict], str | None]:
        """Convert Message list to Responses API input format.

        Returns (input_items, instructions) where instructions is the
        merged system message content (passed via the ``instructions`` param).
        """
        instructions_parts: list[str] = []
        input_items: list[dict] = []
        system_message = self.config.SYSTEM_MESSAGE

        for msg in messages:
            api = msg.to_api_format()
            role = api["role"]

            if role == "system":
                instructions_parts.append(api["content"])
                continue

            if role == "assistant":
                tool_calls = api.get("tool_calls")
                if tool_calls:
                    # Emit text content (if any) as a separate assistant item
                    if api.get("content"):
                        input_items.append({"role": "assistant", "content": api["content"]})
                    # Each tool call becomes a function_call input item
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                            "call_id": tc.get("id", ""),
                        })
                    continue
                input_items.append({"role": "assistant", "content": api.get("content", "")})
                continue

            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": api.get("tool_call_id", ""),
                    "output": api.get("content", ""),
                })
                continue

            # user (or any other role)
            content = api.get("content", "")
            # Convert Chat Completions multimodal content array to Responses API
            # input format:  text → input_text, image_url → input_image
            if isinstance(content, list):
                converted_parts: list[dict] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        converted_parts.append({"type": "input_text", "text": part.get("text", "")})
                    elif ptype == "image_url":
                        img_obj = part.get("image_url") or {}
                        url = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                        detail = img_obj.get("detail", "high") if isinstance(img_obj, dict) else "high"
                        if url:
                            converted_parts.append({"type": "input_image", "image_url": url, "detail": detail})
                if converted_parts:
                    input_items.append({"role": role, "content": converted_parts})
                else:
                    input_items.append({"role": role, "content": ""})
            else:
                input_items.append({"role": role, "content": content})

        instructions = "\n\n".join(instructions_parts) if instructions_parts else (system_message or None)

        if log_input and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Responses API input (%d items): %s", len(input_items), json.dumps(input_items, default=str)[:2000])

        return input_items, instructions

    # ------------------------------------------------------------------
    # Tool schema conversion (Chat Completions -> Responses API)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools_schema(tools_schema: list[dict] | None) -> list[dict] | None:
        """Flatten Chat Completions tool format to Responses API format.

        Chat Completions: {"type": "function", "function": {"name": ..., ...}}
        Responses API:    {"type": "function", "name": ..., ...}
        """
        if not tools_schema:
            return None

        converted = []
        for tool in tools_schema:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                item: dict[str, Any] = {"type": "function", "name": func["name"]}
                if "description" in func:
                    item["description"] = func["description"]
                if "parameters" in func:
                    item["parameters"] = func["parameters"]
                converted.append(item)
            else:
                converted.append(tool)
        return converted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_base_payload(self, messages: List[Message], **kwargs: Any) -> dict[str, Any]:
        """Build the common payload fields for Responses API calls."""
        input_items, instructions = self._prepare_input(messages, log_input=kwargs.get("log_input", True))

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
            "max_output_tokens": self._as_int(
                kwargs.get("max_tokens"), self.effective_max_tokens,
            ),
        }

        if instructions:
            payload["instructions"] = instructions

        payload["temperature"] = self._as_float(
            kwargs.get("temperature", self.config.TEMPERATURE),
            self.config.TEMPERATURE,
        )

        effort = kwargs.get("reasoning_effort")
        if effort:
            payload["reasoning"] = {"effort": effort}
        routing_key = kwargs.get("cache_routing_key")
        if routing_key:
            payload.update(self._cache_routing_fields(routing_key))

        return payload

    def job_request_options(self, job: str, *, reasoning_effort: str | None = None) -> dict[str, Any]:
        """Stable per-job cache routing key plus optional reasoning effort."""
        options: dict[str, Any] = {"cache_routing_key": f"llm-bawt:{job}"}
        if reasoning_effort:
            options["reasoning_effort"] = reasoning_effort
        return options

    def _cache_routing_fields(self, key: str) -> dict[str, Any]:
        """Payload fields that pin same-prefix requests to one prompt cache."""
        return {"prompt_cache_key": key}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_with_tools(
        self,
        messages: List[Message],
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Iterator[str | dict]:
        """Stream response with native tool support via Responses API.

        Yields:
            - str: Content text deltas
            - dict: {"tool_calls": [...], "content": ...} when tools are invoked
        """
        payload = self._build_base_payload(messages, **kwargs)
        payload["stream"] = True

        tools = self._convert_tools_schema(tools_schema)
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        content_buffer = ""
        # Track function calls: item_id -> {call_id, name, arguments}
        func_calls: dict[str, dict] = {}

        try:
            stream = self.client.responses.create(**payload)

            for event in stream:
                etype = getattr(event, "type", "")

                if logger.isEnabledFor(logging.DEBUG) and (
                    etype.startswith("response.function_call") or etype in (
                        "response.output_item.added", "response.output_item.done",
                    )
                ):
                    logger.debug("Responses event: %s attrs=%s", etype, {
                        k: v for k, v in vars(event).items() if not k.startswith("_")
                    })

                if etype == "response.output_text.delta":
                    text = event.delta
                    if text:
                        content_buffer += text
                        yield text

                elif etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        item_id = getattr(item, "id", "") or ""
                        func_calls[item_id] = {
                            "call_id": getattr(item, "call_id", "") or "",
                            "name": getattr(item, "name", "") or "",
                            "arguments": "",
                        }

                elif etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        item_id = getattr(item, "id", "") or ""
                        if item_id not in func_calls:
                            func_calls[item_id] = {"call_id": "", "name": "", "arguments": ""}
                        fc = func_calls[item_id]
                        fc["name"] = getattr(item, "name", "") or fc["name"]
                        fc["call_id"] = getattr(item, "call_id", "") or fc["call_id"]
                        fc["arguments"] = getattr(item, "arguments", "") or fc["arguments"]

                elif etype == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", "") or ""
                    if item_id not in func_calls:
                        func_calls[item_id] = {"call_id": "", "name": "", "arguments": ""}
                    func_calls[item_id]["arguments"] += event.delta

                elif etype == "response.function_call_arguments.done":
                    item_id = getattr(event, "item_id", "") or ""
                    if item_id not in func_calls:
                        func_calls[item_id] = {"call_id": "", "name": "", "arguments": ""}
                    fc = func_calls[item_id]
                    fc["name"] = getattr(event, "name", "") or fc["name"]
                    fc["call_id"] = getattr(event, "call_id", "") or fc["call_id"]
                    fc["arguments"] = getattr(event, "arguments", "") or fc["arguments"]

            # Emit accumulated tool calls (same contract as OpenAIClient)
            if func_calls:
                tool_calls_list = [
                    {
                        "id": fc["call_id"] or item_id,
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["arguments"],
                        },
                    }
                    for item_id, fc in func_calls.items()
                ]
                yield {"tool_calls": tool_calls_list, "content": content_buffer}

        except Exception as e:
            logger.error("Responses API streaming error for model '%s': %s", self.model, e)
            raise

    def stream_raw(self, messages: List[Message], stop: list[str] | str | None = None, **kwargs) -> Iterator[str]:
        """Stream raw text chunks via Responses API (no tool handling)."""
        for item in self.stream_with_tools(messages, **kwargs):
            if isinstance(item, str):
                yield item

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def query_with_tools(
        self,
        messages: List[Message],
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
        """Non-streaming query with tool support via Responses API."""
        payload = self._build_base_payload(messages, **kwargs)

        tools = self._convert_tools_schema(tools_schema)
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        response = self.client.responses.create(**payload)

        content = ""
        tool_calls_payload: list[dict[str, Any]] = []

        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", "") == "output_text":
                        content += getattr(part, "text", "")
            elif item_type == "function_call":
                tool_calls_payload.append({
                    "id": getattr(item, "call_id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "{}"),
                    },
                })

        return {"content": content, "tool_calls": tool_calls_payload}, None

    def query_bounded(
        self, messages: List[Message], *, timeout_seconds: float,
        max_tokens: int, temperature: float = 0.0,
        reasoning_effort: str | None = None, cache_routing_key: str | None = None,
    ) -> str:
        payload = self._build_base_payload(
            messages, max_tokens=max_tokens, temperature=temperature, log_input=False,
            reasoning_effort=reasoning_effort, cache_routing_key=cache_routing_key,
        )
        bounded = self.client.with_options(timeout=timeout_seconds, max_retries=0)
        response = bounded.responses.create(**payload)
        report_usage(self, response)
        return self._response_text(response).strip()

    @staticmethod
    def _response_text(response) -> str:
        return "".join(
            getattr(part, "text", "")
            for item in response.output if getattr(item, "type", "") == "message"
            for part in getattr(item, "content", []) if getattr(part, "type", "") == "output_text"
        )

    def query(
        self,
        messages: List[Message],
        plaintext_output: bool = False,
        stream: bool = True,
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> str:
        """Query via Responses API."""
        should_stream = stream and not self.config.NO_STREAM

        if should_stream:
            def text_iter():
                for item in self.stream_with_tools(messages, **kwargs):
                    if isinstance(item, str):
                        yield item

            return self._handle_streaming_output(
                stream_iterator=text_iter(),
                plaintext_output=plaintext_output,
            )

        # Non-streaming
        payload = self._build_base_payload(messages, **kwargs)
        try:
            response = self.client.responses.create(**payload)
        except OpenAIError as e:
            logger.error("Responses API error for model '%s': %s", self.model, e)
            return f"ERROR: Responses API Error - {e}"

        report_usage(self, response)
        response_text = self._response_text(response)

        if not plaintext_output:
            parts = response_text.split("\n\n", 1)
            self._print_assistant_message(parts[0], second_part=parts[1] if len(parts) > 1 else None)

        return response_text.strip()

    def get_styling(self) -> tuple[str | None, str]:
        return None, "green"
