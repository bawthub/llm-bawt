"""Base LLM client class.

Simple abstract base for OpenAI-compatible API clients.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from abc import ABC, abstractmethod
from typing import Iterator, List, Any
from ..models.message import Message
from ..utils.config import Config
from ..utils.streaming import render_streaming_response


def api_error_text(error: Exception) -> str:
    """SDK error text that keeps the HTTP status for ``ERROR:`` reply strings.

    A non-JSON gateway reply's SDK message is just its body, and callers
    classify the string later (maintenance fallback), so the status must survive.
    """
    text = str(error)
    # TASK-2339: openai 3 keeps the body in the message only for an
    # ``{"error": {"message": ...}}`` reply; a plain-text or other JSON body
    # (gateways, our proxy's {"detail": ...}) leaves just "Error code: N".
    # Keep that reason in the text, as openai 2 did.
    body = getattr(error, "body", None)
    if body not in (None, "", {}, []):
        piece = body if isinstance(body, str) else repr(body)
        message = body.get("message") if isinstance(body, dict) else None
        if piece not in text and not (isinstance(message, str) and message and message in text):
            text = f"{text} - {piece}"
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not text.startswith("Error code:"):
        return f"Error code: {status} - {text}"
    return text


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    SUPPORTS_STREAMING = False

    def __init__(self, model: str, config: Config, model_definition: dict | None = None):
        self.model = model
        self.config = config
        self.model_definition: dict = model_definition or {}
        self.bot_name: str | None = None  # Set by LLMBawt after initialization
        self.model_alias: str | None = None  # Set by LLMBawt after initialization
        force_term = not self.config.PLAIN_OUTPUT
        self.console = Console(force_terminal=force_term)

    def begin_turn_usage(self) -> None:
        """Start one app-owned direct turn, never reset inside its tool loop."""
        from .direct_usage import DirectTurnUsage

        self._direct_turn_usage = DirectTurnUsage(
            self.model_definition, context_window=self.effective_context_window,
            max_output_tokens=self.effective_max_tokens,
        )

    def get_token_usage(self) -> dict | None:
        usage = getattr(self, "_direct_turn_usage", None)
        return usage.snapshot() if usage is not None else None

    @property
    def effective_max_tokens(self) -> int:
        """Model output CAPABILITY for API-request parameterization.

        TASK-609: delegates to the single catalog-driven authority
        (``config.get_model_max_tokens``). No env read, no type guessing —
        the catalog per-model ``max_tokens`` -> global ``max_output_tokens``.
        This is the output cap sent to the model, NOT the budget subtraction
        (that is ``config.resolve_context_budget``'s reserve).
        """
        return self.config.get_model_max_tokens(self.model_alias)

    @property
    def effective_context_window(self) -> int:
        """Context ceiling for this client's model, for engine sizing / display.

        TASK-609: delegates to the single catalog-driven authority
        (``config.get_model_context_window``). No type allow-list, no
        LLAMA_CPP_N_CTX fallback — catalog per-model window -> global
        ``model_context_window_default``.
        """
        return self.config.get_model_context_window(self.model_alias)

    def _format_panel_title(self, style: str = "green", is_service: bool = False) -> str:
        """Format the panel title as 'bot_name [model] (service)'."""
        display_name = self.bot_name or self.model
        # Use \[ to escape opening bracket so Rich doesn't interpret [model] as markup
        # Use dim style for model and service parts to emphasize bot name
        model_part = f" [dim]\\[{self.model_alias}][/dim]" if self.model_alias else ""
        service_part = " [dim](service)[/dim]" if is_service else ""
        return f"[bold {style}]{display_name}[/bold {style}]{model_part}{service_part}"

    @abstractmethod
    def query(self, messages: List[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        """Query the LLM with the given messages.

        Args:
            messages: List of Message objects for conversation history.
            plaintext_output: Whether to return plain text instead of formatted.
            **kwargs: Client-specific arguments (e.g., stream).

        Returns:
            The model's response as a string.
        """
        pass

    def query_bounded(
        self, messages: List[Message], *, timeout_seconds: float,
        max_tokens: int, temperature: float = 0.0, **job_options: Any,
    ) -> str:
        """One stateless text call with transport timeout and no retries/tools.

        Utility callers must fail closed for clients without this contract;
        silently delegating to query() could start an unbounded agent run.
        ``job_options`` are this client's own :meth:`job_request_options`
        (e.g. reasoning effort), so a bounded maintenance call honors the
        same DB settings as ``query`` (TASK-1795).
        """
        raise NotImplementedError("Client does not support bounded utility completion")

    def job_request_options(self, job: str, *, reasoning_effort: str | None = None) -> dict[str, Any]:
        """Provider-specific ``query`` kwargs for a background job (TASK-1729).

        Capability hook: a client whose API supports prompt-cache routing or a
        reasoning-effort knob returns the kwargs that express them; the default
        supports neither and returns nothing. Request shaping lives on the
        client hierarchy because ProviderAdapter only owns credentials.
        """
        return {}

    def supports_native_tools(self) -> bool:
        """Return True if the client supports native tool calling."""
        return False

    def query_with_tools(
        self,
        messages: List[Message],
        tools_schema: Any | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any | None]:
        """Query the LLM with tool schema (native tools if supported).

        Default implementation ignores tools_schema and returns text response.
        Returns a tuple of (response_payload, native_tool_calls_or_none).
        """
        response = self.query(messages, plaintext_output=True, stream=False, stop=stop, **kwargs)
        return response, None

    def _print_user_message(self, content: str):
        """Default implementation for printing user messages."""
        self.console.print()
        self.console.print(f"[bold blue]User:[/bold blue] {content}")

    def _print_assistant_message(self, content: str, panel_title: str | None = None, panel_border_style: str = "green", second_part: str | None = None, is_service: bool = False):
        """Prints assistant message. First part in panel, optional second part below."""
        title = panel_title or self._format_panel_title(style=panel_border_style, is_service=is_service)
        assistant_panel = Panel(
            Markdown(content.strip()),
            title=title,
            border_style=panel_border_style,
            padding=(1, 2),
        )
        self.console.print(Align(assistant_panel, align="left"))
        if second_part:
            self.console.print()
            self.console.print(Align(Markdown(second_part.strip()), align="left", pad=False))
            self.console.print()

    @abstractmethod
    def get_styling(self) -> tuple[str | None, str]:
        """Return the specific panel title and border style for this client."""
        pass

    def unload(self) -> None:
        """Unload the model and free resources.
        
        Override in subclasses that load models into memory (e.g., GGUF).
        Called by ModelLifecycleManager when switching models.
        """
        pass

    def stream_raw(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        """
        Stream raw text chunks from the LLM without console formatting.
        
        This method is used by the API service for SSE streaming.
        Default implementation falls back to non-streaming query.
        
        Args:
            messages: List of Message objects for conversation history.
            **kwargs: Client-specific arguments.
            
        Yields:
            Text chunks as they're generated.
        """
        # Default fallback: yield the entire response as one chunk
        response = self.query(messages, plaintext_output=True, stream=False, **kwargs)
        yield response

    def _handle_streaming_output(
        self,
        stream_iterator: Iterator[str],
        plaintext_output: bool,
        panel_title: str | None = None,
        panel_border_style: str = "green",
        is_service: bool = False,
    ) -> str:
        """Handles streaming output using shared streaming utilities."""
        title = panel_title or self._format_panel_title(style=panel_border_style, is_service=is_service)

        return render_streaming_response(
            stream_iterator=stream_iterator,
            console=self.console,
            panel_title=title,
            panel_border_style=panel_border_style,
            plaintext_output=plaintext_output,
        )


class StubClient(LLMClient):
    """Stub client for history-only operations that don't need actual LLM."""
    
    SUPPORTS_STREAMING = False
    
    def __init__(self, config: Config, bot_name: str = "Assistant"):
        super().__init__(model="stub", config=config)
        self.bot_name = bot_name
    
    def query(self, messages: List[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        raise NotImplementedError("StubClient does not support querying")
    
    def get_styling(self) -> tuple[str | None, str]:
        return self.bot_name, "cyan"
