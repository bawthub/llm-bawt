from __future__ import annotations

import logging
import sys
import os
from copy import copy

from logging.handlers import RotatingFileHandler
from rich.console import Console


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
        # TASK-2386: botocore DEBUG prints signed request headers.
        "botocore",
        "boto3",
        "s3transfer",
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
