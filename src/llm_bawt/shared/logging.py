from __future__ import annotations

import asyncio
import functools
import logging
import sys
import os
import threading
import time
import uuid
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Iterator

from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree


class EventCategory(Enum):
    QUERY = auto()
    LLM = auto()
    MEMORY = auto()
    MCP = auto()
    MODEL = auto()
    CONFIG = auto()
    SYSTEM = auto()


class EventLevel(Enum):
    INFO = auto()
    DETAIL = auto()
    TRACE = auto()


@dataclass(frozen=True)
class Event:
    category: EventCategory
    level: EventLevel
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float | None = None
    parent_id: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class EventCollector:
    """Collects structured events during a single operation (typically one query)."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._operation_stack: list[str] = []
        self._start_times: dict[str, float] = {}

    def emit(
        self,
        category: EventCategory,
        message: str,
        level: EventLevel = EventLevel.INFO,
        **data: Any,
    ) -> str:
        parent_id = self._operation_stack[-1] if self._operation_stack else None
        event = Event(
            category=category,
            level=level,
            message=message,
            data=data,
            parent_id=parent_id,
        )
        self.events.append(event)
        return event.event_id

    @contextmanager
    def operation(
        self,
        category: EventCategory,
        name: str,
        level: EventLevel = EventLevel.DETAIL,
        **data: Any,
    ) -> Iterator[str]:
        event_id = self.emit(category, f"Starting: {name}", level=level, **data)
        self._operation_stack.append(event_id)
        self._start_times[event_id] = time.perf_counter()
        try:
            yield event_id
        finally:
            self._operation_stack.pop()
            duration_ms = (time.perf_counter() - self._start_times.pop(event_id)) * 1000
            self.events.append(
                Event(
                    category=category,
                    level=EventLevel.INFO,
                    message=f"Completed: {name}",
                    data=data,
                    parent_id=self._operation_stack[-1] if self._operation_stack else None,
                    duration_ms=duration_ms,
                )
            )

    def clear(self) -> None:
        self.events.clear()


class EventRenderer:
    """Renders a collector based on verbosity settings."""

    def __init__(self, console: Console, verbose: bool = False, debug: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self.debug = debug
        self._logger = logging.getLogger("llm_bawt")

    def render_query_flow(self, collector: EventCollector) -> None:
        if not (self.verbose or self.debug):
            return

        if self.verbose:
            self._render_verbose_flow(collector.events)

        if self.debug:
            self._render_debug_flow(collector.events)

    def _render_verbose_flow(self, events: list[Event]) -> None:
        if not events:
            return

        by_category: dict[EventCategory, list[Event]] = {}
        for event in events:
            if event.level == EventLevel.TRACE:
                continue
            by_category.setdefault(event.category, []).append(event)

        tree = Tree("Query flow")
        for category, category_events in by_category.items():
            branch = tree.add(category.name)
            for event in category_events:
                msg = event.message
                if event.duration_ms is not None:
                    msg += f" ({event.duration_ms:.1f}ms)"

                if event.data:
                    event_branch = branch.add(msg)
                    for key, value in event.data.items():
                        if key == "duration_ms":
                            continue
                        event_branch.add(f"{key}: {value}")
                else:
                    branch.add(msg)

        self.console.print(Panel(tree, title="Verbose", border_style="dim"))

    def _render_debug_flow(self, events: list[Event]) -> None:
        for event in events:
            extra = ""
            if event.data:
                extra = f" | {event.data}"
            if event.duration_ms is not None:
                extra += f" | {event.duration_ms:.2f}ms"
            self._logger.debug(f"[{event.category.name}] {event.message}{extra}")

    def render_error(self, message: str, exception: Exception | None = None) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {message}")
        if exception and self.debug:
            self._logger.exception(exception)


class LogConfig:
    """Centralized logging setup for CLI/server use.
    
    Logging levels:
    - Default: ERROR only, clean Rich output (spinners, status messages)
    - --verbose: INFO level, structured event rendering via Rich
    - --debug: DEBUG spam with full tracebacks and third-party noise filtered
    """

    NOISY_LOGGERS: tuple[str, ...] = (
        "httpx",
        "httpcore",
        "openai",
        "requests",
        "urllib3",
        "markdown_it",
        "sqlalchemy.engine",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "asyncio",
        "uvicorn",
        "uvicorn.error",
    )
    
    # These loggers are suppressed even more aggressively (ERROR only)
    # because they duplicate information we log ourselves
    VERY_NOISY_LOGGERS: tuple[str, ...] = (
        "uvicorn.access",  # HTTP access logs like "127.0.0.1 - POST /mcp HTTP/1.1 200"
    )

    @classmethod
    def configure(cls, *, verbose: bool = False, debug: bool = False) -> None:
        """Configure logging based on verbosity flags.
        
        Args:
            verbose: Enable INFO level logs with Rich rendering.
            debug: Enable DEBUG spam (overrides verbose).
        """
        from rich.logging import RichHandler

        from . import logfmt

        if not logfmt.is_tty():
            # Containers: the standard logfmt line (TASK-2225). --debug keeps
            # DEBUG in the debug file only; stdout follows LLM_BAWT_LOG_LEVEL.
            prefix = os.getenv("LLM_BAWT_LOG_PREFIX", "").strip()
            log_dir = os.getenv("LLM_BAWT_LOG_DIR", ".logs")
            name = f"{prefix}.debug.log" if prefix else "llm-bawt.debug.log"
            logfmt.configure_logging(
                prefix or "app",
                level=None if (verbose or debug) else logging.WARNING,
                debug_file=os.path.join(log_dir, name) if debug else None,
            )
            for logger_name in cls.NOISY_LOGGERS:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
            return

        # Get the root logger and clear any existing handlers
        root = logging.getLogger()
        root.handlers.clear()
        
        # Always silence noisy third-party loggers first
        for logger_name in cls.NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # Suppress very noisy loggers completely (we log our own summaries)
        for logger_name in cls.VERY_NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        log_prefix = os.getenv("LLM_BAWT_LOG_PREFIX", "").strip()
        log_dir = os.getenv("LLM_BAWT_LOG_DIR", ".logs")
        console_level = logging.INFO if (verbose or debug) else logging.WARNING
        root_level = logging.DEBUG if debug else console_level
        log_file_name = f"{log_prefix}.debug.log" if log_prefix else "llm-bawt.debug.log"
        class _PrefixFormatter(logging.Formatter):
            def __init__(self, *args, markup=False, **kwargs):
                super().__init__(*args, **kwargs)
                from rich.markup import escape
                self.prefix = escape(f"[{log_prefix}]") if markup else f"[{log_prefix}]"

            def format(self, record: logging.LogRecord) -> str:
                # Each handler owns its formatting; never mutate the shared record.
                local = copy(record)
                if log_prefix:
                    local.msg = f"{self.prefix} {record.getMessage()}"
                    local.args = ()
                return super().format(local)

        def _add_debug_file_handler() -> None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file_name)
            try:
                with open(log_path, "a", encoding="utf-8"):
                    pass
            except OSError as exc:
                print(f"[llm-bawt] Failed to open debug log file: {log_path} ({exc})", file=sys.stderr)
                return
            for handler in root.handlers:
                if isinstance(handler, RotatingFileHandler) and handler.baseFilename == os.path.abspath(log_path):
                    return

            handler = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(_PrefixFormatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            root.addHandler(handler)
        
        # Create a Rich console that renders even when backgrounded
        console = Console(stderr=True, force_terminal=True)

        if debug:
            # Debug mode: keep Rich formatting but keep DEBUG out of stdout
            handler = RichHandler(
                console=console,
                show_path=False,
                show_time=False,
                show_level=False,
                rich_tracebacks=True,
                markup=True,
            )
            handler.setFormatter(_PrefixFormatter(markup=True))
            handler.setLevel(console_level)
            root.addHandler(handler)
            root.setLevel(root_level)
            _add_debug_file_handler()
            logging.getLogger("llm_bawt").setLevel(logging.INFO)
            return

        if verbose:
            # Verbose mode: Show INFO from llm-bawt with Rich formatting
            handler = RichHandler(
                console=console,
                show_path=False,
                show_time=False,
                show_level=False,
                rich_tracebacks=True,
                markup=True,
            )
            handler.setFormatter(_PrefixFormatter(markup=True))
            root.addHandler(handler)
            root.setLevel(root_level)
            logging.getLogger("llm_bawt").setLevel(logging.INFO)
            return

        # Default: Errors only (WARNING+), rely on Rich for user-facing feedback
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=False,
            show_level=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(_PrefixFormatter(markup=True))
        root.addHandler(handler)
        root.setLevel(root_level)

    @classmethod
    def get_renderer(cls, console: Console, *, verbose: bool, debug: bool) -> EventRenderer:
        return EventRenderer(console=console, verbose=verbose, debug=debug)

    @classmethod
    def new_collector(cls) -> EventCollector:
        return EventCollector()


_collector_context = threading.local()


def _get_current_collector() -> EventCollector | None:
    return getattr(_collector_context, "collector", None)


def set_current_collector(collector: EventCollector | None) -> None:
    _collector_context.collector = collector


@contextmanager
def query_context(collector: EventCollector) -> Iterator[EventCollector]:
    old = _get_current_collector()
    set_current_collector(collector)
    try:
        yield collector
    finally:
        set_current_collector(old)


def log_operation(category: EventCategory, name: str | None = None):
    def decorator(func):
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = _get_current_collector()
            if collector is None:
                return func(*args, **kwargs)
            with collector.operation(category, op_name):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = _get_current_collector()
            if collector is None:
                return await func(*args, **kwargs)
            with collector.operation(category, op_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
