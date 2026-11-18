"""Rich output that can never fail a caller (TASK-2434).

Diagnostic output (log lines, verbose previews) often embeds untrusted text:
user and bot messages, tool results, memories. Rich parses square-bracket
markup, so a message containing a closing-tag-shaped token raises
``MarkupError`` from inside ``Console.print`` -- even on a quiet console,
because Rich renders before it discards. That aborted agent turns.

Two rules keep logging inert:

* ``SafeConsole`` skips rendering entirely when quiet, retries a failed print
  with markup off, and otherwise swallows the error. A diagnostic print never
  raises.
* ``SafeRichHandler`` defaults to ``markup=False``. A record opts in with
  ``extra={"markup": True}`` and must escape any untrusted fields it embeds;
  if its markup still fails to parse, the line is printed literally.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.text import Text

__all__ = ["SafeConsole", "SafeRichHandler", "escape"]


class SafeConsole(Console):
    """A ``Console`` whose ``print``/``log`` never raise into the caller."""

    def print(self, *objects: Any, **kwargs: Any) -> None:  # noqa: A003
        if self.quiet:
            return
        try:
            super().print(*objects, **kwargs)
        except Exception:
            try:
                super().print(*objects, **{**kwargs, "markup": False})
            except Exception:
                pass

    def log(self, *objects: Any, **kwargs: Any) -> None:
        if self.quiet:
            return
        try:
            super().log(*objects, **kwargs)
        except Exception:
            try:
                super().log(*objects, **{**kwargs, "markup": False})
            except Exception:
                pass


class SafeRichHandler(RichHandler):
    """``RichHandler`` with markup off by default and a literal fallback."""

    def __init__(self, *args: Any, markup: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, markup=markup, **kwargs)

    def render_message(self, record: logging.LogRecord, message: str):
        try:
            return super().render_message(record, message)
        except Exception:
            return Text(message)

    def emit(self, record: logging.LogRecord) -> None:
        # RichHandler renders outside its own try block; keep every failure
        # inside logging's standard handleError path, never the caller's.
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)
