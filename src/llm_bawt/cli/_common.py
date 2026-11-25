"""Shared CLI primitives (TASK-554).

Extracted from ``cli/app.py`` so the split-out command modules
(``tool_render``, ``display_cmd``) and ``app`` itself can all
share one Rich console and one service-client singleton without importing
``app`` (which would create a cycle). ``app`` re-imports these names so
existing ``cli.app.console`` / ``cli.app.get_service_client`` references keep
working.
"""

import argparse
import sys

from rich.console import Console
from rich.prompt import Prompt

from llm_bawt.utils.config import Config, set_config_value

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


def ensure_default_user(config: Config, args: argparse.Namespace) -> None:
    """Prompt for and persist DEFAULT_USER if missing and no --user is provided."""
    if getattr(args, "user", None):
        return
    default_user = getattr(config, "DEFAULT_USER", None)
    if default_user and str(default_user).strip():
        return
    if not sys.stdin.isatty():
        console.print("[bold red]DEFAULT_USER is required to continue.[/bold red]")
        console.print("[dim]Set LLM_BAWT_DEFAULT_USER or pass --user.[/dim]")
        sys.exit(1)
    console.print("[yellow]No default user configured.[/yellow]")
    user_id = Prompt.ask("Enter default user id", default="default").strip()
    if not user_id:
        console.print("[red]No user id provided. Aborting.[/red]")
        sys.exit(1)
    if not set_config_value("DEFAULT_USER", user_id, config):
        console.print("[red]Failed to save DEFAULT_USER to config.[/red]")
        sys.exit(1)
    config.DEFAULT_USER = user_id
    console.print(f"[green]Default user set to '{user_id}'.[/green]")
