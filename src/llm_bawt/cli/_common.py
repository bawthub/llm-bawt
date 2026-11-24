"""Shared CLI primitives (TASK-554).

Extracted from ``cli/app.py`` so the split-out command modules
(``tool_render``, ``display_cmd``, ``admin_cmd``) and ``app`` itself can all
share one Rich console and one service-client singleton without importing
``app`` (which would create a cycle). ``app`` re-imports these names so
existing ``cli.app.console`` / ``cli.app.get_service_client`` references keep
working.
"""

import argparse

from rich.console import Console

from llm_bawt.utils.config import Config

console = Console()


# Cache for service client
_service_client = None


def get_service_client(config: Config | None = None):
    """Get or create the service client singleton."""
    global _service_client
    if _service_client is None:
        try:
            from llm_bawt.service import ServiceClient
            # Build service URL from config if not provided
            if config is None:
                config = Config()
            service_url = getattr(config, 'SERVICE_URL', None)
            if not service_url and hasattr(config, 'SERVICE_HOST') and hasattr(config, 'SERVICE_PORT'):
                host = "127.0.0.1" if config.SERVICE_HOST == "0.0.0.0" else config.SERVICE_HOST
                service_url = f"http://{host}:{config.SERVICE_PORT}"
            _service_client = ServiceClient(http_url=service_url)
        except ImportError:
            _service_client = False  # Mark as unavailable
    return _service_client if _service_client else None


def _is_service_mode(args: argparse.Namespace, config: Config) -> bool:
    """Determine if the CLI should operate in service mode.

    True if --service flag or USE_SERVICE config is enabled.
    """
    if getattr(args, "service", False):
        return True
    return bool(config.USE_SERVICE)
