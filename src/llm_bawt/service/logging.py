"""
Rich-formatted logging for the llm-service.

Provides intelligent logging with three verbosity levels:
- Normal: Minimal output, only errors
- Verbose (--verbose/-v): Human-readable flow with meaningful messages
- Debug (--debug): Low-level DEBUG messages with technical details

Verbose mode aims to be:
- Scannable: Use visual icons and colors for quick pattern recognition
- Meaningful: Show "Fetching memories..." not "MCP tool call -> tools/get_messages"  
- Deduped: Avoid redundant information (e.g., model loading shown once)
- Hierarchical: Group related operations, show timing for parent operations

Usage:
    from .logging import ServiceLogger, setup_service_logging

    # At startup
    setup_service_logging(verbose=args.verbose, debug=args.debug)

    # Create logger for a module
    log = ServiceLogger(__name__)

    # Log events
    log.api_request("POST", "/v1/chat/completions", request_data)
    log.model_loading("gpt4", "openai")
    log.info("Something happened")
"""

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Context variable to track request IDs across async operations
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Custom theme for service logs
SERVICE_THEME = Theme({
    "api.method": "bold cyan",
    "api.path": "green",
    "api.status.ok": "bold green",
    "api.status.error": "bold red",
    "model.name": "bold magenta",
    "model.type": "dim magenta",
    "timing": "dim cyan",
    "request_id": "dim yellow",
    "bot.name": "bold blue",
    "user.id": "dim blue",
    "memory": "yellow",
    "task": "cyan",
    "tool": "bold yellow",
    "success": "bold green",
    "muted": "dim",
})

# Visual icons for different operation types (verbose mode)
ICONS = {
    "request": "📨",
    "response": "📤",
    "model": "🤖",
    "memory": "💾",
    "tool": "🔧",
    "search": "🔍",
    "history": "📜",
    "user": "👤",
    "assistant": "🤖",
    "success": "✓",
    "error": "✗",
    "loading": "⏳",
    "done": "✓",
}

def _redact_image_data(obj: Any) -> Any:
    """Deep-copy a payload, replacing base64 image URLs with a placeholder."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image/"):
                mime = v.split(";")[0].split(":")[1] if ";" in v else "image"
                result[k] = f"[{mime} base64 redacted]"
            else:
                result[k] = _redact_image_data(v)
        return result
    if isinstance(obj, list):
        return [_redact_image_data(item) for item in obj]
    return obj


# Global state
_verbose = False
_debug = False
_console: Console | None = None
_model_loading_announced: set[str] = set()  # Track which models we've announced loading


def get_console() -> Console:
    """Get the shared console instance."""
    global _console
    if _console is None:
        # force_terminal=True ensures Rich renders markup even when backgrounded
        _console = Console(theme=SERVICE_THEME, stderr=True, force_terminal=True)
    return _console


def setup_service_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for the service.

    Args:
        verbose: Enable verbose logging (shows payloads, timing details)
        debug: Enable debug logging (low-level DEBUG messages, unformatted)
    """
    global _verbose, _debug

    _verbose = verbose or debug
    _debug = debug

    # Suppress Hugging Face / Transformers progress noise in non-debug mode
    if not debug:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Determine log levels
    console_level = logging.INFO if (verbose or debug) else logging.WARNING
    root_level = logging.DEBUG if debug else console_level

    console = get_console()

    # Configure root logger for the service
    log_prefix = os.getenv("LLM_BAWT_LOG_PREFIX", "").strip()
    log_time_format = "[%I:%M:%S %p]"
    log_dir = os.getenv("LLM_BAWT_LOG_DIR", ".logs")

    class _PrefixFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not log_prefix:
                return True
            # Format the message first if there are args, then prepend prefix
            if record.args:
                record.msg = record.msg % record.args
                record.args = ()
            # Use plain text prefix - escape [ for Rich markup
            record.msg = f"\\[{log_prefix}] {record.msg}"
            return True

    def _add_debug_file_handler() -> None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "llm-service.debug.log")
        try:
            # Truncate the log file on restart
            with open(log_path, "w", encoding="utf-8"):
                pass
        except OSError as exc:
            print(f"[llm-bawt] Failed to open debug log file: {log_path} ({exc})", file=sys.stderr)
            return
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, RotatingFileHandler) and handler.baseFilename == os.path.abspath(log_path):
                return

        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handler.addFilter(_PrefixFilter())
        root_logger.addHandler(handler)
    
    # Rich formatted output to stdout
    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        handlers=[RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            omit_repeated_times=False,
            log_time_format=log_time_format,
            show_level=False,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
            markup=True,
        )],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_PrefixFilter())
        handler.setLevel(console_level)
    if verbose or debug:
        _add_debug_file_handler()

    # Reduce noise from third-party libraries
    # These libraries produce a lot of INFO/DEBUG messages that clutter the output
    noisy_loggers = [
        "uvicorn",
        "uvicorn.access",  # HTTP request logs like "127.0.0.1 - POST /mcp HTTP/1.1 200"
        "uvicorn.error",
        "httpx",
        "httpcore",
        "asyncio",
        "urllib3",
        "urllib3.connectionpool",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "transformers",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "torch",
        "filelock",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "tqdm",
        "mcp",                        # FastMCP internal logs ("Processing request of type ...", "Terminating session: ...")
        "mcp.server",
        "mcp.server.lowlevel.server",
        "mcp.server.streamable_http",
        "watchfiles",                 # File watcher logs — writes to debug log create feedback loop
        "watchfiles.main",
    ]
    
    # In verbose mode, suppress HTTP access logs completely (we log our own summaries)
    # In debug mode, show everything
    for logger_name in noisy_loggers:
        if logger_name in ("uvicorn.access",) and not debug:
            # Always suppress access logs in non-debug (we have our own MCP logging)
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        else:
            logging.getLogger(logger_name).setLevel(logging.WARNING if not debug else logging.INFO)
    
    # Suppress tqdm progress bars in non-debug mode
    # This prevents the "Batches: 100%..." output from sentence-transformers
    if not debug:
        os.environ["TQDM_DISABLE"] = "1"


def generate_request_id() -> str:
    """Generate a short request ID for tracing."""
    return uuid4().hex[:8]


def set_request_id(request_id: str) -> None:
    """Set the current request ID (for async context)."""
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return _request_id.get()


@dataclass
class RequestContext:
    """Context for tracking a single request."""
    request_id: str
    method: str
    path: str
    start_time: float = field(default_factory=time.time)
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    stream: bool = False

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class ServiceLogger:
    """
    Rich-formatted logger for the llm-service.

    Provides structured logging methods for common service events.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._console = get_console()

    # -------------------------------------------------------------------------
    # Standard logging methods (delegate to underlying logger)
    # -------------------------------------------------------------------------

    def debug(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", {}).setdefault("markup", False)
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", {}).setdefault("markup", False)
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", {}).setdefault("markup", False)
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", {}).setdefault("markup", False)
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", {}).setdefault("markup", False)
        self._logger.exception(msg, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Structured service logging methods
    # -------------------------------------------------------------------------

    def startup(self, version: str, host: str, port: int, models: list[str], default_model: str | None) -> None:
        """Log service startup with configuration summary."""
        if _debug:
            self._logger.info(f"Service v{version} starting on {host}:{port}")
            self._logger.info(f"Models: {', '.join(models)}")
            self._logger.info(f"Default model: {default_model}")
            return

        table = Table(title="llm-bawt Service Started", show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")
        table.add_row("Version", f"[bold]{version}[/bold]")
        table.add_row("Endpoint", f"[bold green]http://{host}:{port}[/bold green]")
        table.add_row("Models", f"[model.name]{len(models)} available[/model.name]")
        if default_model:
            table.add_row("Default", f"[model.name]{default_model}[/model.name]")

        self._console.print()
        self._console.print(Panel(table, border_style="green"))
        self._console.print()

    def shutdown(self) -> None:
        """Log service shutdown."""
        if _debug:
            self._logger.info("Service shutting down")
            return

        self._console.print("[dim]Service shutting down...[/dim]")

    def api_request(self, ctx: RequestContext, payload: dict[str, Any] | None = None) -> None:
        """Log an incoming API request."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        method = f"[api.method]{ctx.method}[/api.method]"
        path = f"[api.path]{ctx.path}[/api.path]"

        parts = [req_id, method, path]

        if ctx.model:
            parts.append(f"model=[model.name]{ctx.model}[/model.name]")
        if ctx.bot_id:
            parts.append(f"bot=[bot.name]{ctx.bot_id}[/bot.name]")
        if ctx.stream:
            parts.append("[dim]streaming[/dim]")

        self._logger.info(" ".join(parts))

        # Verbose: show compact incoming summary
        if _verbose and payload:
            messages = payload.get("messages", [])
            if messages:
                # Get the user's prompt (last user message)
                user_prompt = ""
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = m.get("content", "")
                        if isinstance(content, list):
                            # Multimodal — extract text parts, summarize images
                            parts = []
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text":
                                        parts.append(block.get("text", ""))
                                    elif block.get("type") == "image_url":
                                        parts.append("[image]")
                            user_prompt = " ".join(parts)[:100]
                            if len(" ".join(parts)) > 100:
                                user_prompt += "..."
                        else:
                            user_prompt = str(content)[:100]
                            if len(str(content)) > 100:
                                user_prompt += "..."
                        break
                
                self._console.print(f"  [bold green]← Client[/bold green] | {len(messages)} msg(s)")
                if user_prompt:
                    self._console.print(f"  [green]prompt:[/green] {user_prompt}")

        # Debug: dump request payload (with image data redacted)
        if _debug and payload:
            try:
                sanitized = _redact_image_data(payload)
                self._logger.info("Request payload: %s", json.dumps(sanitized, ensure_ascii=False, default=str))
            except Exception:
                self._logger.info("Request payload: (unserializable)")

    def api_response(self, ctx: RequestContext, status: int = 200, tokens: int | None = None) -> None:
        """Log an API response."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        elapsed = f"[timing]{ctx.elapsed_ms:.0f}ms[/timing]"

        if status < 400:
            status_str = f"[api.status.ok]{status}[/api.status.ok]"
        else:
            status_str = f"[api.status.error]{status}[/api.status.error]"

        parts = [req_id, status_str, elapsed]

        if tokens:
            parts.append(f"[dim]{tokens} tokens[/dim]")

        self._logger.info(" ".join(parts))

    def api_error(self, ctx: RequestContext, error: str, status: int = 500) -> None:
        """Log an API error."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        status_str = f"[api.status.error]{status}[/api.status.error]"
        elapsed = f"[timing]{ctx.elapsed_ms:.0f}ms[/timing]"

        self._logger.error(f"{req_id} {status_str} {elapsed} {error}")

    def model_loading(self, model_alias: str, model_type: str, cached: bool = False) -> None:
        """Log model loading event - deduped to avoid repeated messages."""
        global _model_loading_announced
        
        cache_key = f"{model_alias}:{model_type}"
        if cache_key in _model_loading_announced:
            # Already announced this model loading, skip duplicate
            return
        _model_loading_announced.add(cache_key)
        
        if cached:
            self._logger.info(
                f"{ICONS['model']} [model.name]{model_alias}[/model.name] [muted](cached)[/muted]"
            )
        else:
            self._logger.info(
                f"{ICONS['loading']} [model.name]{model_alias}[/model.name] [muted]loading...[/muted]"
            )

    def model_loaded(self, model_alias: str, model_type: str, load_time_ms: float) -> None:
        """Log model loaded successfully - deduped to avoid repeated messages."""
        global _model_loading_announced
        
        cache_key = f"{model_alias}:{model_type}"
        
        # Only log if we previously announced loading for this model
        # This prevents duplicate "ready" messages
        if cache_key not in _model_loading_announced:
            return
        
        # Clear the loading announcement
        _model_loading_announced.discard(cache_key)
        
        self._logger.info(
            f"{ICONS['done']} [model.name]{model_alias}[/model.name] [success]ready[/success] [timing]{load_time_ms:.0f}ms[/timing]"
        )

    def model_error(self, model_alias: str, error: str) -> None:
        """Log model loading error."""
        self.error("%s %s failed: %s", ICONS['error'], model_alias, error)

    def mcp_operation(
        self, 
        operation: str, 
        bot_id: str | None = None,
        duration_ms: float | None = None, 
        count: int | None = None,
        details: str | None = None,
        success: bool = True,
        params: dict[str, Any] | None = None,
        result: Any | None = None,
    ) -> None:
        """Log MCP tool operations - single line, always with context."""
        
        def _sanitize(text: str, max_len: int = 40) -> str:
            """Sanitize text for single-line logging."""
            text = " ".join(text.split())  # Collapse all whitespace
            return text[:max_len] + "..." if len(text) > max_len else text
        
        if operation == "get_messages":
            icon = ICONS["history"]
            if count == 0 or not result:
                msg = f"{icon} History: empty"
            else:
                first = result[0] if isinstance(result, list) else result
                first_content = first.get("content", "") if isinstance(first, dict) else str(first)
                if count == 1:
                    msg = f"{icon} History ({count}): \"{_sanitize(first_content)}\""
                else:
                    last = result[-1] if isinstance(result, list) else result
                    last_content = last.get("content", "") if isinstance(last, dict) else str(last)
                    msg = f"{icon} History ({count}): \"{_sanitize(first_content, 30)}\" → \"{_sanitize(last_content, 30)}\""
                
        elif operation == "add_message":
            icon = ICONS["success"]
            role = params.get("role", "?") if params else "?"
            content = params.get("content", "") if params else ""
            msg = f"{icon} Saved [{role}]: \"{_sanitize(content, 50)}\""
            
        elif operation == "search_memories":
            icon = ICONS["memory"]
            query = params.get("query", "") if params else ""
            msg = f"{icon} Memory search \"{_sanitize(query, 30)}\": {count or 0} results"
            
        elif operation == "store_memory":
            icon = ICONS["memory"]
            content = params.get("content", "") if params else ""
            importance = params.get("importance", 0.5) if params else 0.5
            msg = f"{icon} Stored (imp={importance:.1f}): \"{_sanitize(content, 50)}\""
            
        elif operation == "delete_memory":
            icon = ICONS["memory"]
            memory_id = params.get("memory_id", "?") if params else "?"
            msg = f"{icon} Deleted memory: {memory_id}"
            
        elif operation == "clear_messages":
            icon = ICONS["history"]
            msg = f"{icon} Cleared history"
            
        else:
            icon = ICONS["tool"]
            msg = f"{icon} {operation}"
            if count is not None:
                msg += f" ({count})"
        
        if duration_ms is not None:
            msg += f" [{duration_ms:.0f}ms]"
            
        self._logger.info(msg) if success else self._logger.warning(msg + " [FAILED]")

    def _log_mcp_params(self, operation: str, params: dict[str, Any]) -> None:
        """Log MCP operation parameters in verbose mode - one line."""
        # For store_memory, show the content prominently
        if operation == "store_memory" and "content" in params:
            content = params["content"]
            # Show full content up to 200 chars
            if len(content) > 200:
                content = content[:200] + "..."
            tags = params.get("tags", [])
            importance = params.get("importance", 0.5)
            self._console.print(f"  [dim]content:[/dim] [cyan]{content}[/cyan]")
            self._console.print(f"  [dim]tags={tags} importance={importance:.1f}[/dim]")
            return
            
        param_strs = []
        for k, v in params.items():
            if isinstance(v, str) and len(v) > 80:
                param_strs.append(f"{k}={v[:80]}...")
            else:
                param_strs.append(f"{k}={v}")
        self._console.print(f"  [dim]params: {', '.join(param_strs)}[/dim]")

    def _log_mcp_result(self, operation: str, result: Any) -> None:
        """Log MCP operation result in verbose mode.
        
        For history (get_messages), show condensed view.
        For everything else (memory searches, etc.), show full data.
        """
        if result is None:
            return
        
        # History can be condensed - it's usually long and just context
        is_history = operation == "get_messages"
        
        if isinstance(result, list):
            if len(result) == 0:
                self._console.print("  [dim]result: (empty)[/dim]")
            elif is_history:
                # Very condensed for history: just first and last
                if len(result) == 1:
                    self._log_result_item(result[0], 0, condensed=True)
                elif len(result) <= 3:
                    for i, item in enumerate(result):
                        self._log_result_item(item, i, condensed=True)
                else:
                    self._log_result_item(result[0], 0, condensed=True)
                    self._console.print(f"  [dim]... ({len(result) - 2} more) ...[/dim]")
                    self._log_result_item(result[-1], len(result) - 1, condensed=True)
            else:
                # Full output for memory searches, etc.
                for i, item in enumerate(result):
                    self._log_result_item(item, i, condensed=is_history)
        elif isinstance(result, dict):
            self._log_result_item(result, None, condensed=False)
        else:
            self._console.print(f"  [dim]result:[/dim] {result}")

    def _log_result_item(self, item: Any, index: int | None, condensed: bool = False) -> None:
        """Log a single result item - one line per item.
        
        Args:
            item: The item to log
            index: Index in list (or None for single items)
            condensed: If True, truncate long content (for history)
        """
        if not isinstance(item, dict):
            self._console.print(f"    {item}")
            return
        
        prefix = f"    [dim][{index}][/dim] " if index is not None else "    "
        
        # Format based on item type (message vs memory)
        if "role" in item and "content" in item:
            # Message format - one line
            role = item.get("role", "?")
            content = item.get("content", "")
            if condensed and len(content) > 120:
                content = content[:120] + "..."
            role_color = "green" if role == "user" else "blue" if role == "assistant" else "yellow"
            self._console.print(f"{prefix}[{role_color}]{role}[/{role_color}]: {content}")
        elif "content" in item:
            # Memory format - one line with metadata at end
            content = item.get("content", "")
            tags = item.get("tags", [])
            importance = item.get("importance", "")
            relevance = item.get("relevance", "")
            
            # Build metadata suffix
            meta_parts = []
            if tags:
                meta_parts.append(f"tags={tags}")
            if importance:
                meta_parts.append(f"imp={importance}")
            if relevance:
                meta_parts.append(f"rel={relevance:.2f}" if isinstance(relevance, float) else f"rel={relevance}")
            meta_str = f" [dim]({', '.join(meta_parts)})[/dim]" if meta_parts else ""
            
            self._console.print(f"{prefix}{content}{meta_str}")
        else:
            # Generic - one line JSON
            import json
            try:
                self._console.print(f"{prefix}{json.dumps(item, default=str)}")
            except Exception:
                self._console.print(f"{prefix}{item}")

    def memory_operation(self, operation: str, bot_id: str, count: int | None = None, details: str | None = None) -> None:
        """Log memory operations (retrieval, storage, extraction)."""
        parts = [f"{ICONS['memory']} [memory]{operation}[/memory]"]
        if count is not None:
            parts.append(f"[muted]({count} items)[/muted]")
        if details:
            parts.append(f"[muted]{details}[/muted]")

        self._logger.info(" ".join(parts))

    def task_submitted(self, task_id: str, task_type: str, bot_id: str, payload: dict | None = None) -> None:
        """Log background task submission with context - single line."""
        self._logger.info(f"{ICONS['loading']} {task_type} queued")

    def task_completed(self, task_id: str, task_type: str, elapsed_ms: float, result: dict | None = None) -> None:
        """Log background task completion with meaningful results."""
        # Generic task completion
        self._logger.info(f"{ICONS['done']} {task_type} done [{elapsed_ms:.0f}ms]")

    def task_failed(self, task_id: str, task_type: str, error: str, elapsed_ms: float) -> None:
        """Log background task failure."""
        self.error("%s %s failed %.0fms: %s", ICONS['error'], task_type, elapsed_ms, error)

    def cache_hit(self, cache_type: str, key: str) -> None:
        """Log cache hit (verbose only)."""
        if _verbose:
            self._logger.debug(f"[dim]cache hit: {cache_type} [{key}][/dim]")

    def cache_miss(self, cache_type: str, key: str) -> None:
        """Log cache miss (verbose only)."""
        if _verbose:
            self._logger.debug(f"[dim]cache miss: {cache_type} [{key}][/dim]")

    def tool_result(self, tool_name: str, result: str, truncate_at: int = 500) -> None:
        """Log tool execution result (verbose only).
        
        In verbose mode, shows the actual result returned to the LLM.
        Useful for debugging what information the model received.
        
        Args:
            tool_name: Name of the tool that was executed
            result: The formatted result string returned to the LLM
            truncate_at: Max characters to show (0 = no truncation)
        """
        if not _verbose:
            return
        
        console = get_console()
        
        # Truncate if needed
        display_result = result
        truncated = False
        if truncate_at > 0 and len(result) > truncate_at:
            display_result = result[:truncate_at]
            truncated = True
        
        # Format nicely
        lines = display_result.strip().split('\n')
        if len(lines) == 1:
            # Single line - show inline
            console.print(f"  [tool]→ {tool_name}[/tool]: {display_result}{'...' if truncated else ''}")
        else:
            # Multi-line - show in a panel
            console.print(f"  [tool]→ {tool_name} result:[/tool]")
            for line in lines:
                console.print(f"    {line}")
            if truncated:
                console.print(f"    [dim]... ({len(result) - truncate_at} more chars)[/dim]")

    def llm_context(self, messages: list, label: str = "LLM Context") -> None:
        """
        Log a summary of messages being sent to the LLM (verbose only).
        
        Shows a compact summary: message counts by type, memory content, and the user prompt.
        With --verbose, also dumps the full message contents.
        """
        if not _verbose:
            return
        
        # Count messages by role
        role_counts = {"system": 0, "user": 0, "assistant": 0}
        memory_content = ""
        
        for msg in messages:
            # Handle both Message objects and dicts
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content or ""
            else:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
            
            role_counts[role] = role_counts.get(role, 0) + 1
            
            # Check for memory context (second system message typically)
            if role == "system" and "What You Remember" in content:
                memory_content = content
            
        # Calculate history turns (user+assistant pairs, excluding current prompt)
        history_turns = min(role_counts["user"] - 1, role_counts["assistant"])
        
        # Build compact summary line  
        parts = [f"{ICONS['request']} [bold cyan]→ LLM[/bold cyan]"]
        
        if history_turns > 0:
            parts.append(f"{history_turns} turns")
        
        memory_facts = memory_content.count("• ") if memory_content else 0
        if memory_facts > 0:
            parts.append(f"[memory]{memory_facts} memories[/memory]")
        
        parts.append(f"[muted]{len(messages)} msgs[/muted]")
        
        self._console.print(" | ".join(parts))
        
        # VERBOSE: Dump conversation history (skip system prompt, only show last few turns)
        self._console.print(f"[dim]{'─' * 60}[/dim]")
        
        # Find non-system messages (the actual conversation)
        conv_messages = [(i, msg) for i, msg in enumerate(messages) 
                        if (hasattr(msg, 'role') and msg.role != 'system') or 
                           (isinstance(msg, dict) and msg.get('role') != 'system')]
        
        # Show system prompt summary (just length)
        system_msgs = [msg for msg in messages 
                      if (hasattr(msg, 'role') and msg.role == 'system') or
                         (isinstance(msg, dict) and msg.get('role') == 'system')]
        if system_msgs:
            sys_content = system_msgs[0].content if hasattr(system_msgs[0], 'content') else system_msgs[0].get('content', '')
            self._console.print(f"[dim]── SYSTEM ({len(sys_content)} chars) ──[/dim]")
        
        # Show last N conversation turns (user + assistant pairs)
        max_turns_to_show = 3
        if len(conv_messages) > max_turns_to_show * 2:
            skipped = len(conv_messages) - max_turns_to_show * 2
            self._console.print(f"[dim]... ({skipped} earlier messages) ...[/dim]")
            conv_messages = conv_messages[-(max_turns_to_show * 2):]
        
        for i, msg in conv_messages:
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content or ""
            else:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
            
            # Color by role
            if role == "user":
                role_color = "green"
            else:
                role_color = "blue"
            
            # Truncate long messages
            display_content = content[:300] + "..." if len(content) > 300 else content
            self._console.print(f"[{role_color}]── {role.upper()} ──[/{role_color}]")
            self._console.print(display_content)
        self._console.print(f"[dim]{'─' * 60}[/dim]")

    def llm_response(self, response: str, tokens: int | None = None, elapsed_ms: float | None = None) -> None:
        """Log a summary of the LLM response (verbose only)."""
        if not _verbose:
            return
        
        parts = [f"{ICONS['response']} [bold blue]← LLM[/bold blue]"]
        
        # Show tokens or chars
        if tokens:
            parts.append(f"~{tokens} tok")
        else:
            # Estimate tokens (~4 chars per token)
            tokens = len(response) // 4
            parts.append(f"~{tokens} tok")
        
        # Show timing and tokens/sec if available
        if elapsed_ms and elapsed_ms > 0 and tokens:
            secs = elapsed_ms / 1000
            tok_per_sec = tokens / secs
            # Color code: green = fast (>30), yellow = ok (10-30), red = slow (<10)
            if tok_per_sec >= 30:
                speed_color = "green"
            elif tok_per_sec >= 10:
                speed_color = "yellow"
            else:
                speed_color = "red"
            parts.append(f"[{speed_color}]{tok_per_sec:.1f} tok/s[/{speed_color}]")
            parts.append(f"[muted]{elapsed_ms:.0f}ms[/muted]")
        
        self._console.print(" | ".join(parts))
        
        # Show full response
        self._console.print("  response: " + response, markup=False)

    def _log_payload(self, label: str, data: dict[str, Any], max_content_len: int = 500) -> None:
        """Log a payload (for verbose mode)."""
        # Deep copy and truncate long content
        def truncate(obj: Any, depth: int = 0) -> Any:
            if depth > 5:
                return "..."
            if isinstance(obj, dict):
                return {k: truncate(v, depth + 1) for k, v in obj.items()}
            if isinstance(obj, list):
                return [truncate(item, depth + 1) for item in obj[:10]]  # Max 10 items
            if isinstance(obj, str) and len(obj) > max_content_len:
                return obj[:max_content_len] + "..."
            return obj

        truncated = truncate(data)
        try:
            formatted = json.dumps(truncated, indent=2, default=str)
            self._console.print(f"[dim]{label}:[/dim]")
            self._console.print(formatted, highlight=True)
        except Exception:
            self._logger.debug(f"{label}: {truncated}")


# Module-level convenience function
def get_service_logger(name: str) -> ServiceLogger:
    """Get a ServiceLogger instance for the given module name."""
    return ServiceLogger(name)
