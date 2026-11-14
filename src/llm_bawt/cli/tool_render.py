"""Tool-call rendering helpers for the CLI (TASK-554).

Extracted verbatim from ``cli/app.py``: formatting/summarizing tool arguments
and rendering compact, styled tool-activity lines. Self-contained — the only
external state is the ``Console`` passed into ``_render_tool_event``. ``app``
and ``display_cmd`` re-import these names.
"""

import json

from rich.console import Console
from rich.text import Text


def _format_tool_call(tool_name: str, tool_args: dict | str) -> str:
    """Format a tool call for display — show primary arg cleanly, avoid raw JSON dump."""
    if not tool_args:
        return tool_name

    # OpenAI API returns tool arguments as a JSON string; parse it
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            # Unparseable string — truncate and show as-is
            display = tool_args[:90] + "…" if len(tool_args) > 90 else tool_args
            return f"{tool_name}({display.replace('[', chr(92) + '[')})"
        if not isinstance(tool_args, dict) or not tool_args:
            return tool_name

    # Keys to try as the "primary" display arg, in preference order
    PRIMARY_KEYS = ("command", "file_path", "path", "pattern", "query", "url", "description", "prompt", "code", "text", "content", "subagent_type")
    primary_key = next((k for k in PRIMARY_KEYS if k in tool_args), None)
    if primary_key is None:
        primary_key = next(iter(tool_args))

    primary_val = str(tool_args[primary_key])
    # Truncate long values
    if len(primary_val) > 90:
        primary_val = primary_val[:87] + "…"
    # Escape Rich markup in user content
    primary_val = primary_val.replace("[", "\\[")

    extra = len(tool_args) - 1
    extra_str = f", +{extra}" if extra else ""
    return f"{tool_name}({primary_key}={primary_val}{extra_str})"


def _summarize_tool_args(tool_args: dict | str) -> tuple[str | None, str | None, int]:
    """Return the primary tool arg, display value, and remaining arg count."""
    if not tool_args:
        return None, None, 0

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            return None, tool_args, 0
        if not isinstance(tool_args, dict) or not tool_args:
            return None, None, 0

    primary_keys = ("command", "file_path", "path", "pattern", "query", "url", "description", "prompt", "code", "text", "content", "subagent_type")
    primary_key = next((key for key in primary_keys if key in tool_args), None)
    if primary_key is None:
        primary_key = next(iter(tool_args), None)
    if primary_key is None:
        return None, None, 0

    primary_value = str(tool_args.get(primary_key, ""))
    return primary_key, primary_value, max(0, len(tool_args) - 1)


def _truncate_middle(value: str, max_length: int = 92) -> str:
    """Trim long values while preserving both ends for paths and commands."""
    if len(value) <= max_length:
        return value
    keep_left = max_length // 2 - 2
    keep_right = max_length - keep_left - 3
    return f"{value[:keep_left]}...{value[-keep_right:]}"


def _tool_badge(tool_name: str) -> Text:
    """Render a compact pill for the tool name."""
    normalized = tool_name.lower()
    palette = {
        "bash": ("#6be6b4", "#123d35"),
        "read": ("#7ed7ff", "#153a56"),
        "agent": ("#7fdfff", "#133851"),
        "grep": ("#86e1ca", "#163b36"),
        "edit": ("#ffbf7a", "#4c3320"),
        "write": ("#ffbf7a", "#4c3320"),
    }
    fg, bg = palette.get(normalized, ("#c8d3dc", "#32424f"))
    badge = Text()
    badge.append(f" {tool_name} ", style=f"bold {fg} on {bg}")
    return badge


def _format_tool_preview(primary_key: str | None, primary_value: str | None) -> str:
    """Create a concise preview string optimized for one-line terminal output."""
    if not primary_value:
        return ""

    value = " ".join(primary_value.split())
    if value.startswith("/home/"):
        # Shorten any "/home/<user>/" prefix to "~/" for a tidy preview.
        parts = value.split("/", 3)
        if len(parts) == 4:  # ["", "home", "<user>", "<rest>"]
            value = "~/" + parts[3]

    if primary_key in {"file_path", "path"}:
        return _truncate_middle(value, 88)
    if primary_key == "command":
        return _truncate_middle(value, 96)
    if primary_key in {"description", "prompt"}:
        return _truncate_middle(value, 84)
    return _truncate_middle(value, 90)


def _render_tool_event(
    console: Console,
    *,
    tool_name: str,
    tool_args: dict | str | None = None,
    result_text: str | None = None,
    plaintext_output: bool = False,
) -> None:
    """Render tool activity in a compact, styled line."""
    if result_text is None:
        primary_key, primary_value, extra_count = _summarize_tool_args(tool_args or {})
        if plaintext_output:
            console.print(f"-> {tool_name}: {_format_tool_call(tool_name, tool_args or {})}")
            return

        preview = _format_tool_preview(primary_key, primary_value)
        body = Text()
        body.append_text(_tool_badge(tool_name))
        if preview:
            body.append("  ")
            body.append(preview, style="#c8d3dc")
        if extra_count:
            body.append("  ")
            body.append(f"+{extra_count}", style="#7f8a96")
            if extra_count != 1:
                body.append(" args", style="#7f8a96")
            else:
                body.append(" arg", style="#7f8a96")

        console.print(body)
        return

    display = _truncate_middle(" ".join(result_text.split()), 96)
    if plaintext_output:
        console.print(f"<- {tool_name}: {display}")
        return

    body = Text()
    body.append("✓", style="bold #82d59d")
    body.append(" ")
    body.append_text(_tool_badge(tool_name))
    body.append("  ")
    body.append(display, style="#d6dccf")

    console.print(body)

