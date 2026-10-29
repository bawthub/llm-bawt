"""OpenAI-compatible API client.

Supports OpenAI API and any OpenAI-compatible endpoint (llama.cpp server, vLLM, Ollama, etc.)
"""

from __future__ import annotations

import json
import httpx
from openai import OpenAI, OpenAIError
from typing import List, Iterator, Any
from rich.json import JSON
from rich.rule import Rule
from ..clients.base import LLMClient
from .usage_hook import report_usage
from ..utils.config import Config
from ..models.message import Message
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Client for OpenAI API and OpenAI-compatible endpoints.
    
    Supports:
    - OpenAI API (default)
    - OpenAI-compatible endpoints (llama.cpp server, vLLM, Ollama, etc.)
      via base_url parameter
    """
    SUPPORTS_STREAMING = True

    def __init__(self, model: str, config: Config, base_url: str | None = None, api_key: str | None = None, model_definition: dict | None = None):
        """Initialize OpenAI client.
        
        Args:
            model: Model name/ID
            config: Application config
            base_url: Optional custom API endpoint for OpenAI-compatible servers
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
            model_definition: Optional model definition dict from models.yaml
        """
        super().__init__(model, config, model_definition=model_definition)
        self.base_url = base_url
        self.api_key = api_key or self._get_api_key()

        # For custom endpoints, API key is optional (many local servers don't need it)
        if not self.api_key and not base_url:
            raise ValueError(
                "OpenAI API key not found. Connect the OpenAI API provider in "
                "the UI (Settings → Providers), or set OPENAI_API_KEY as a "
                "legacy fallback."
            )
        
        # Use dummy key for local servers if none provided
        effective_key = self.api_key or "not-needed"
        
        if base_url:
            self.client = OpenAI(api_key=effective_key, base_url=base_url)
            logger.debug(f"OpenAI client using custom endpoint: {base_url}")
        else:
            self.client = OpenAI(api_key=effective_key)

    def _get_api_key(self) -> str | None:
        """DB-first (CredentialStore via the `openai-api` adapter), env as legacy fallback."""
        from ..service.providers.api_key import resolve_api_key

        return resolve_api_key(self.config, "openai-api", env_vars=("OPENAI_API_KEY",))

    def supports_native_tools(self) -> bool:
        return True

    def _model_supports_temperature_top_p(self) -> bool:
        """Check if the model supports temperature and top_p parameters."""
        unsupported_patterns = (
            "-chat-latest",
            "-search-preview",
            "-audio-preview",
        )
        unsupported_prefixes = ("o1", "o3", "o4")
        
        for pattern in unsupported_patterns:
            if pattern in self.model:
                return False
        if self.model.startswith(unsupported_prefixes):
            return False
        return True

    def _model_requires_max_completion_tokens(self) -> bool:
        """Check if the model requires max_completion_tokens instead of max_tokens."""
        legacy_models = (
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
        )
        if self.model in legacy_models or self.model == "gpt-4":
            return False
        if self.model.startswith(("gpt-3.5-", "gpt-4-turbo-", "gpt-4-32k-")):
            return False
        return True

    _TOKEN_PARAM_CANDIDATES = ("max_tokens", "max_completion_tokens", "max_output_tokens")

    def _initial_token_param_key(self) -> str:
        if self._model_requires_max_completion_tokens():
            return "max_completion_tokens"
        return "max_tokens"

    def _set_token_param(self, payload: dict, key: str, value: int) -> dict:
        for k in self._TOKEN_PARAM_CANDIDATES:
            payload.pop(k, None)
        payload[key] = value
        return payload

    def _print_verbose_params(self, payload: dict) -> None:
        token_key = next((k for k in self._TOKEN_PARAM_CANDIDATES if k in payload), None)
        token_info = f"{token_key}={payload[token_key]}" if token_key else "max_tokens=N/A"
        temp_info = f"temp={payload.get('temperature', 'N/A')}" if 'temperature' in payload else "temp=N/A"
        top_p_info = f"top_p={payload.get('top_p', 'N/A')}" if 'top_p' in payload else "top_p=N/A"
        self.console.print(f"[dim]Params:[/dim] [italic]{token_info}, {temp_info}, {top_p_info}, stream={payload['stream']}[/italic]")

    def _chat_create_with_fallback(self, payload: dict):
        """Call chat.completions.create with fallback across token param names."""
        current_key = next((k for k in self._TOKEN_PARAM_CANDIDATES if k in payload), self._initial_token_param_key())
        current_val = payload.get(current_key, self.effective_max_tokens)
        keys_to_try = [current_key] + [k for k in self._TOKEN_PARAM_CANDIDATES if k != current_key]

        last_err: Exception | None = None
        for key in keys_to_try:
            try_payload = dict(payload)
            self._set_token_param(try_payload, key, int(current_val))
            attempted_without_temp_top_p = False
            try:
                return self.client.chat.completions.create(**try_payload)
            except httpx.HTTPStatusError as e:
                self._log_http_error(e, try_payload)
                body = ""
                if getattr(e, "response", None) is not None:
                    try:
                        body = e.response.text.lower()
                    except Exception:
                        body = ""
                if "unsupported parameter" in body and any(
                    k in body for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
                ):
                    last_err = e
                    continue
                last_err = e
                break
            except OpenAIError as e:
                self._log_openai_error(e, try_payload)
                msg = str(e).lower()
                if ("unsupported parameter" in msg and any(k in msg for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"))) or (
                    "invalid_request_error" in msg and "token" in msg
                ):
                    last_err = e
                    continue
                if ("temperature" in msg or "top_p" in msg) and ("unsupported" in msg or "does not support" in msg):
                    if not attempted_without_temp_top_p:
                        try_payload_no_temp = dict(try_payload)
                        try_payload_no_temp.pop('temperature', None)
                        try_payload_no_temp.pop('top_p', None)
                        attempted_without_temp_top_p = True
                        try:
                            return self.client.chat.completions.create(**try_payload_no_temp)
                        except OpenAIError as e2:
                            m2 = str(e2).lower()
                            if ("unsupported parameter" in m2 and any(k in m2 for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"))):
                                last_err = e2
                                continue
                            last_err = e2
                            break
                        except Exception as e2:
                            last_err = e2
                            break
                    last_err = e
                    break
                raise
            except Exception as e:
                self._log_openai_error(e, try_payload)
                last_err = e
                break
        if last_err:
            raise last_err

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

    def _log_http_error(self, err: httpx.HTTPStatusError, payload: dict) -> None:
        response = getattr(err, "response", None)
        status = response.status_code if response is not None else "unknown"
        body = ""
        if response is not None:
            try:
                body = response.text
            except Exception:
                body = "<unreadable response body>"
        if body and len(body) > 2000:
            body = body[:2000] + "...(truncated)"
        logger.error(
            "OpenAI HTTP %s for %s. Response body: %s",
            status,
            getattr(response, "url", "unknown"),
            body or "<empty>",
        )

    def _log_openai_error(self, err: Exception, payload: dict) -> None:
        response = getattr(err, "response", None) or getattr(err, "http_response", None)
        status = getattr(response, "status_code", "unknown") if response is not None else "unknown"
        body = ""
        if response is not None:
            try:
                body = response.text
            except Exception:
                body = "<unreadable response body>"
        if body and len(body) > 2000:
            body = body[:2000] + "...(truncated)"
        if body:
            logger.error(
                "OpenAI error (status=%s). Response body: %s",
                status,
                body,
            )

    def _is_stream_disconnect_error(self, err: Exception) -> bool:
        """Detect transport-level stream disconnects from OpenAI-compatible backends."""
        seen: set[int] = set()
        current: Exception | None = err
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            msg = str(current).lower()
            if (
                "incomplete chunked read" in msg
                or "peer closed connection without sending complete message body" in msg
                or current.__class__.__name__ == "RemoteProtocolError"
            ):
                return True
            next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
            current = next_exc if isinstance(next_exc, Exception) else None
        return False

    def stream_raw(self, messages: List[Message], stop: list[str] | str | None = None, **kwargs) -> Iterator[str]:
        """Stream raw text chunks from OpenAI API without console formatting."""
        api_messages = self._prepare_api_messages(messages)
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        max_tokens = self._as_int(kwargs.get("max_tokens"), self.effective_max_tokens)
        token_key = self._initial_token_param_key()
        self._set_token_param(payload, token_key, max_tokens)

        if self._model_supports_temperature_top_p():
            payload["temperature"] = self._as_float(kwargs.get("temperature"), self.config.TEMPERATURE)
            payload["top_p"] = self._as_float(kwargs.get("top_p"), self.config.TOP_P)

        try:
            stream = self._chat_create_with_fallback(payload)
            yield from self._iterate_openai_chunks(stream)
        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}")
            raise

    def stream_with_tools(
        self,
        messages: List[Message],
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Iterator[str | dict]:
        """Stream response with native tool support.

        Yields:
            - str: Content chunks as they arrive
            - dict: Tool calls dict {"tool_calls": [...]} when stream ends with tool calls

        If the model responds with text only, yields content chunks.
        If the model calls tools, yields nothing (tool calls returned at end as dict).
        """
        api_messages = self._prepare_api_messages(messages)
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        if tools_schema:
            payload["tools"] = tools_schema
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        max_tokens = self._as_int(kwargs.get("max_tokens"), self.effective_max_tokens)
        token_key = self._initial_token_param_key()
        self._set_token_param(payload, token_key, max_tokens)

        if self._model_supports_temperature_top_p():
            payload["temperature"] = self._as_float(kwargs.get("temperature"), self.config.TEMPERATURE)
            payload["top_p"] = self._as_float(kwargs.get("top_p"), self.config.TOP_P)

        yielded_any_content = False
        try:
            stream = self._chat_create_with_fallback(payload)

            # Accumulate tool calls from stream
            tool_calls_acc: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
            content_buffer = ""

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Handle content
                if delta.content:
                    content_buffer += delta.content
                    yielded_any_content = True
                    yield delta.content

                # Handle tool calls (accumulated across chunks)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

            # If we accumulated tool calls, yield them as final item
            if tool_calls_acc:
                sorted_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
                yield {"tool_calls": sorted_calls, "content": content_buffer}

        except Exception as e:
            if self._is_stream_disconnect_error(e):
                logger.warning("Streaming with tools disconnected for model '%s': %s", self.model, e)
                # If nothing was emitted yet, fall back to non-streaming once.
                if not yielded_any_content:
                    try:
                        result, _ = self.query_with_tools(
                            messages=messages,
                            tools_schema=tools_schema,
                            tool_choice=tool_choice,
                            stop=stop,
                        )
                        content = result.get("content", "")
                        if content:
                            yield content
                        tool_calls = result.get("tool_calls") or []
                        if tool_calls:
                            yield {"tool_calls": tool_calls, "content": content}
                        return
                    except Exception as fallback_err:
                        logger.error(
                            "Non-streaming fallback after stream disconnect failed for model '%s': %s",
                            self.model,
                            fallback_err,
                        )
                return
            logger.error(f"Error during OpenAI streaming with tools: {e}")
            raise

    def _build_query_payload(
        self, messages: List[Message], *, stream: bool,
        stop: list[str] | str | None = None, **kwargs,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": self._prepare_api_messages(messages),
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        response_format = kwargs.get("response_format")
        if response_format is not None:
            payload["response_format"] = response_format
        max_tokens = self._as_int(kwargs.get("max_tokens"), self.effective_max_tokens)
        token_key = self._initial_token_param_key()
        self._set_token_param(payload, token_key, max_tokens)

        if self._model_supports_temperature_top_p():
            payload["temperature"] = self._as_float(kwargs.get("temperature"), self.config.TEMPERATURE)
            payload["top_p"] = self._as_float(kwargs.get("top_p"), self.config.TOP_P)

        return payload

    def query_bounded(
        self, messages: List[Message], *, timeout_seconds: float,
        max_tokens: int, temperature: float = 0.0,
    ) -> str:
        payload = self._build_query_payload(
            messages, stream=False, max_tokens=max_tokens, temperature=temperature,
        )
        # Recall never retries token/temperature compatibility errors: each
        # request is one transport-bounded call, without mutating a pooled client.
        bounded = self.client.with_options(timeout=timeout_seconds, max_retries=0)
        completion = bounded.chat.completions.create(**payload)
        report_usage(self, completion)
        return (completion.choices[0].message.content or "").strip()

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, stop: list[str] | str | None = None, **kwargs) -> str:
        """Query OpenAI API with full message history."""
        should_stream = stream and not self.config.NO_STREAM
        payload = self._build_query_payload(messages, stream=should_stream, stop=stop, **kwargs)
        api_messages = payload["messages"]
        if self.config.VERBOSE:
            self.console.print(Rule("Querying OpenAI API", style="green"))
            self._print_verbose_params(payload)
            self.console.print(Rule("Request Payload", style="dim blue"))
            try:
                payload_str = json.dumps(payload, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                import pprint
                self.console.print(pprint.pformat(payload))
            self.console.print(Rule(style="green"))
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"OpenAI Request Payload: {json.dumps(payload)}")

        if should_stream:
            response = self._stream_response(api_messages, plaintext_output, payload)
        else:
            response = self._get_full_response(api_messages, plaintext_output, payload)

        if self.config.VERBOSE:
            self.console.print(Rule(style="green"))

        return response

    def query_with_tools(
        self,
        messages: List[Message],
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
        """Query OpenAI with native tool calling schema."""
        api_messages = self._prepare_api_messages(messages)
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        response_format = kwargs.get("response_format")
        if response_format is not None:
            payload["response_format"] = response_format
        if tools_schema:
            payload["tools"] = tools_schema
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        max_tokens = self._as_int(kwargs.get("max_tokens"), self.effective_max_tokens)
        token_key = self._initial_token_param_key()
        self._set_token_param(payload, token_key, max_tokens)

        if self._model_supports_temperature_top_p():
            payload["temperature"] = self._as_float(kwargs.get("temperature"), self.config.TEMPERATURE)
            payload["top_p"] = self._as_float(kwargs.get("top_p"), self.config.TOP_P)

        completion = self._chat_create_with_fallback(payload)
        message = completion.choices[0].message
        content = message.content or ""
        tool_calls_payload: list[dict[str, Any]] = []
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            func = getattr(call, "function", None)
            if func is None and isinstance(call, dict):
                func = call.get("function")
            name = getattr(func, "name", None) if func is not None else None
            arguments = getattr(func, "arguments", None) if func is not None else None
            if isinstance(func, dict):
                name = func.get("name")
                arguments = func.get("arguments")
            tool_calls_payload.append(
                {
                    "id": getattr(call, "id", None) if not isinstance(call, dict) else call.get("id"),
                    "type": getattr(call, "type", None) if not isinstance(call, dict) else call.get("type"),
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

        return {"content": content, "tool_calls": tool_calls_payload}, None

    def _prepare_api_messages(self, messages: List[Message]) -> list[dict]:
        prepared = []
        system_contents = []
        system_message = self.config.SYSTEM_MESSAGE
        
        for msg in messages:
            api_msg = msg.to_api_format()
            if api_msg['role'] == 'system':
                system_contents.append(api_msg['content'])
            else:
                prepared.append(api_msg)

        if system_contents:
            merged_system = "\n\n".join(system_contents)
            prepared.insert(0, {"role": "system", "content": merged_system})
        elif system_message:
            prepared.insert(0, {"role": "system", "content": system_message})

        return prepared

    def _stream_response(self, api_messages: List[dict], plaintext_output: bool, payload: dict) -> str:
        """Stream the response using the base class handler."""
        try:
            stream = self._chat_create_with_fallback(payload)
            return self._handle_streaming_output(
                stream_iterator=self._iterate_openai_chunks(stream),
                plaintext_output=plaintext_output,
            )
        except OpenAIError as e:
            logger.error(f"Error during OpenAI API request: {e}")
            return f"ERROR: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI streaming: {e}")
            return f"ERROR: Unexpected error - {e}"

    def _iterate_openai_chunks(self, stream: Iterator) -> Iterator[str]:
        """Iterates through OpenAI stream chunks and yields content."""
        try:
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except OpenAIError as e:
            logger.error(f"\nError during OpenAI stream processing: {e}")
            yield f"\nERROR: OpenAI API Error - {e}"
        except Exception as e:
            if self._is_stream_disconnect_error(e):
                logger.warning("Streaming disconnected for model '%s': %s", self.model, e)
                return
            logger.error(f"\nUnexpected error during OpenAI stream iteration: {e}")
            yield f"\nERROR: Unexpected error - {e}"

    def _get_full_response(self, api_messages: List[dict], plaintext_output: bool, payload: dict) -> str:
        """Gets the full response without streaming."""
        try:
            payload['stream'] = False
            completion = self._chat_create_with_fallback(payload)
            report_usage(self, completion)
            response_text = completion.choices[0].message.content or ""
            if self.config.VERBOSE:
                usage = completion.usage
                if usage:
                    self.console.print(f"[dim]OpenAI Tokens: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}[/dim]")
            elif logger.isEnabledFor(logging.DEBUG):
                usage = completion.usage
                if usage:
                    logger.debug(f"OpenAI Tokens: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")

            if not plaintext_output:
                parts = response_text.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self._print_assistant_message(first_part, second_part=second_part)

            return response_text.strip()

        except OpenAIError as e:
            logger.error(f"Error making OpenAI API request: {e}")
            return f"ERROR: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI request: {e}")
            return f"ERROR: Unexpected error - {e}"

    def get_styling(self) -> tuple[str | None, str]:
        """Return OpenAI specific styling."""
        return None, "green"
