"""CLI package for llm-bawt.

- app: chat CLI entry point, argument parsing and history/status commands
- display_cmd: service query streaming plus status/job rendering
"""

from .app import main, run_app, show_status

__all__ = ["main", "run_app", "show_status"]
