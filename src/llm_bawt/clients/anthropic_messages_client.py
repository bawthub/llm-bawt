"""Plain-text client for Anthropic Messages endpoints (TASK-1726).

Serves app-side background calls (summaries, extraction, utility completions)
on any access path that speaks the Anthropic Messages protocol with an API key
or no auth: Anthropic API, OpenRouter, Moonshot, Z.AI, local Ollama/llama-server.
Subscription OAuth paths never reach this client (the background harness
rejects them in ``ProtocolCompatibility``).

Text only: no streaming, no tools. ``query_with_tools`` keeps the base
behavior (ignore tools, return text).
"""

from __future__ import annotations

import base64
import logging
from typing import Any, List

from anthropic import Anthropic, BadRequestError

from .base import LLMClient
from .usage_hook import report_usage
from ..models.message import Message
from ..utils.config import Config

logger = logging.getLogger(__name__)

#: Local Messages servers ignore the key, but the SDK requires one.
_NO_AUTH_KEY = "not-needed"


class AnthropicMessagesClient(LLMClient):
    SUPPORTS_STREAMING = False

    def __init__(
        self,
        model: str,
        config: Config,
        *,
        base_url: str,
        api_key: str | None,
        model_definition: dict | None = None,
    ):
        super().__init__(model, config, model_definition=model_definition)
        if not base_url:
            raise ValueError("Anthropic Messages endpoint has no base_url")
        self.base_url = base_url
        self.client = Anthropic(api_key=api_key or _NO_AUTH_KEY, base_url=base_url)
        self._rejected_fields: set[str] = set()

    # -- message conversion ------------------------------------------------
    @staticmethod
    def _image_block(part: dict) -> dict | None:
        image = part.get("image_url") or {}
        url = image.get("url") if isinstance(image, dict) else str(image)
        if not url:
            return None
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(",", 1)
            base64.b64decode(data, validate=True)  # reject a malformed payload here, not upstream
            return {"type": "image", "source": {
                "type": "base64", "media_type": header[5:].split(";", 1)[0], "data": data,
            }}
        return {"type": "image", "source": {"type": "url", "url": url}}

    @classmethod
    def _content(cls, content: Any) -> Any:
        if not isinstance(content, list):
            return content or ""
        blocks: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                if part.get("text"):
                    blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                block = cls._image_block(part)
                if block:
                    blocks.append(block)
        return blocks

    def _prepare(self, messages: List[Message]) -> tuple[str | None, list[dict]]:
        """Split system text out (Messages takes it as a parameter) and merge
        consecutive same-role turns, which the protocol rejects."""
        system_parts: list[str] = []
        turns: list[dict] = []
        for msg in messages:
            api = msg.to_api_format()
            role = api["role"]
            if role == "system":
                if api.get("content"):
                    system_parts.append(str(api["content"]))
                continue
            role = "assistant" if role == "assistant" else "user"
            content = self._content(api.get("content"))
            if turns and turns[-1]["role"] == role:
                prev = turns[-1]["content"]
                as_blocks = lambda c: c if isinstance(c, list) else [{"type": "text", "text": c}]  # noqa: E731
                turns[-1]["content"] = as_blocks(prev) + as_blocks(content)
            else:
                turns.append({"role": role, "content": content})
        return ("\n\n".join(system_parts) or None), turns

    @staticmethod
    def _text(response: Any) -> str:
        return "".join(
            getattr(block, "text", "") for block in (response.content or [])
            if getattr(block, "type", None) == "text"
        ).strip()

    def _payload(self, messages: List[Message], *, max_tokens: int, optional: dict[str, Any]) -> dict:
        system, turns = self._prepare(messages)
        payload: dict[str, Any] = {"model": self.model, "max_tokens": int(max_tokens), "messages": turns}
        if system:
            payload["system"] = system
        payload.update(optional)
        return payload

    def job_request_options(self, job: str, *, reasoning_effort: str | None = None) -> dict[str, Any]:
        """Reasoning effort ``none`` → Messages ``thinking: disabled`` (TASK-1726).

        Other efforts have no Messages shape that OpenAI-style effort maps to
        honored across access paths, so they keep the provider default.
        """
        if reasoning_effort == "none":
            return {"thinking": {"type": "disabled"}}
        return {}

    # -- LLMClient ---------------------------------------------------------
    def get_styling(self) -> tuple[str | None, str]:
        return None, "green"

    def query(self, messages: List[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_tokens") or self.effective_max_tokens
        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = self.config.TEMPERATURE
        optional: dict[str, Any] = {"temperature": float(temperature)}
        if kwargs.get("thinking"):
            optional["thinking"] = kwargs["thinking"]
        return self._create(self.client, messages, max_tokens=max_tokens, optional=optional)

    def query_bounded(
        self, messages: List[Message], *, timeout_seconds: float,
        max_tokens: int, temperature: float = 0.0, thinking: dict | None = None,
    ) -> str:
        bounded = self.client.with_options(timeout=timeout_seconds, max_retries=0)
        optional: dict[str, Any] = {"temperature": float(temperature)}
        if thinking:
            optional["thinking"] = thinking
        return self._create(bounded, messages, max_tokens=max_tokens, optional=optional)

    @staticmethod
    def _rejected_field(error: BadRequestError, sent: dict[str, Any]) -> str | None:
        message = str(error).lower()
        return next((field for field in sent if field in message), None)

    def _create(self, sdk: Any, messages: List[Message], *, max_tokens: int, optional: dict[str, Any]) -> str:
        """Send with the optional fields (``temperature``, ``thinking``) this
        endpoint hasn't rejected. A 400 naming one of them (current Claude
        models reject both) is retried without it and remembered for the
        client's lifetime, like ``OpenAIClient``; no model-name literals."""
        sent = {k: v for k, v in optional.items() if k not in self._rejected_fields}
        while True:
            try:
                response = sdk.messages.create(
                    **self._payload(messages, max_tokens=max_tokens, optional=sent))
                break
            except BadRequestError as error:
                field = self._rejected_field(error, sent)
                if field is None:
                    raise
                logger.info("%s rejects %s; retrying without it", self.model, field)
                self._rejected_fields.add(field)
                sent.pop(field)
        report_usage(self, response)
        return self._text(response)
