"""Shared streaming output utilities for consistent rendering across modes.

This module provides common functions for streaming LLM output with:
- First paragraph in a Rich panel
- Remainder streamed below with markdown rendering
- Platform-specific handling for Windows vs Linux/macOS
"""

import platform
from typing import Iterator

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

# Detect Windows for streaming workarounds
IS_WINDOWS = platform.system() == "Windows"


def render_streaming_response(
    stream_iterator: Iterator[str],
    console: Console,
    panel_title: str,
    panel_border_style: str = "green",
    plaintext_output: bool = False,
) -> str:
    """Render a streaming response with panel + markdown formatting.
    
    The first paragraph (before \\n\\n) goes in a Rich panel.
    The remainder streams below and is rendered as markdown when complete.
    
    Args:
        stream_iterator: Iterator yielding string chunks
        console: Rich Console for output
        panel_title: Title for the panel (e.g., "[bold cyan]Nova[/bold cyan]")
        panel_border_style: Border style for the panel
        plaintext_output: If True, output plain text without formatting
        
    Returns:
        The complete response text
    """
    if plaintext_output:
        return _stream_plaintext(stream_iterator)
    elif IS_WINDOWS:
        return _stream_windows(stream_iterator, console, panel_title, panel_border_style)
    else:
        return _stream_rich_live(stream_iterator, console, panel_title, panel_border_style)


def render_complete_response(
    response: str,
    console: Console,
    panel_title: str,
    panel_border_style: str = "green",
    plaintext_output: bool = False,
) -> None:
    """Render a complete (non-streaming) response.
    
    Args:
        response: The complete response text
        console: Rich Console for output
        panel_title: Title for the panel
        panel_border_style: Border style for the panel
        plaintext_output: If True, output plain text without formatting
    """
    if plaintext_output:
        print(response)
        return
    
    split_marker = "\n\n"
    stripped = response.lstrip()
    if split_marker in stripped:
        first_part, remainder = stripped.split(split_marker, 1)
        _print_panel(console, first_part, panel_title, panel_border_style)
        if remainder.strip():
            console.print(Align(Markdown(remainder.strip()), align="left", pad=False))
    else:
        _print_panel(console, response, panel_title, panel_border_style)


def _print_panel(
    console: Console,
    content: str,
    panel_title: str,
    panel_border_style: str,
) -> None:
    """Print content in a Rich panel."""
    panel = Panel(
        Markdown(content.strip()),
        title=panel_title,
        border_style=panel_border_style,
        padding=(1, 2),
    )
    console.print(Align(panel, align="left"))


def _stream_plaintext(stream_iterator: Iterator[str]) -> str:
    """Stream output as plain text."""
    total_response = ""
    try:
        for chunk in stream_iterator:
            if chunk:
                total_response += chunk
                print(chunk, end='', flush=True)
        print()
    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        total_response += f"\nERROR: {e}"
    return total_response


def _stream_windows(
    stream_iterator: Iterator[str],
    console: Console,
    panel_title: str,
    panel_border_style: str,
) -> str:
    """Windows-specific streaming handler.
    
    Shows panel immediately after first paragraph, then streams the rest
    as plain text (avoiding Rich Live which causes display issues on Windows).
    """
    total_response = ""
    split_marker = "\n\n"
    first_part_buffer = ""
    first_part_printed = False
    remainder_buffer = ""
    cursor = "▌"
    
    try:
        for chunk in stream_iterator:
            if not chunk:
                continue
            total_response += chunk
            
            if not first_part_printed:
                first_part_buffer += chunk
                stripped = first_part_buffer.lstrip()
                if split_marker in stripped:
                    first_part, remainder = stripped.split(split_marker, 1)
                    _print_panel(console, first_part, panel_title, panel_border_style)
                    first_part_printed = True
                    remainder_buffer = remainder
                    if remainder:
                        print(remainder + cursor, end='', flush=True)
            else:
                remainder_buffer += chunk
                print(f"\b \b{chunk}{cursor}", end='', flush=True)
        
        # Finalize
        if not first_part_printed:
            _print_panel(console, first_part_buffer, panel_title, panel_border_style)
        else:
            print("\b \b", end='', flush=True)
            if remainder_buffer.strip():
                line_count = remainder_buffer.count('\n')
                if line_count > 0:
                    print(f"\r\033[{line_count}A\033[J", end='', flush=True)
                else:
                    print("\r\033[K", end='', flush=True)
                console.print(Align(Markdown(remainder_buffer.strip()), align="left", pad=False))
            else:
                print(flush=True)
                
    except KeyboardInterrupt:
        print()
        console.print("[bold yellow]Interrupted![/bold yellow]")
    except Exception as e:
        print()
        console.print(f"[bold red]Error during streaming:[/bold red] {e}")
        total_response += f"\nERROR: {e}"
    
    return total_response


def _stream_rich_live(
    stream_iterator: Iterator[str],
    console: Console,
    panel_title: str,
    panel_border_style: str,
) -> str:
    """Linux/macOS streaming handler using Rich Live display."""
    total_response = ""
    split_marker = "\n\n"
    first_part_buffer = ""
    first_part_printed = False
    cursor = "▌"
    live_display: Live | None = None
    visible_text = ""
    overflow_buffer = ""
    live_frozen = False
    
    console_size = getattr(console, "size", None)
    max_live_lines = max(5, (getattr(console_size, "height", 24) or 24) - 8)

    def _split_visible(text: str) -> tuple[str, str]:
        lines = text.splitlines(keepends=True)
        if len(lines) <= max_live_lines:
            return text, ""
        return "".join(lines[:max_live_lines]), "".join(lines[max_live_lines:])

    def _start_live(initial: str) -> None:
        nonlocal live_display, visible_text
        if live_display:
            return
        visible_text = initial
        live_display = Live(
            Align(Markdown(f"{visible_text}{cursor}" if visible_text else cursor), align="left", pad=False),
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible",
            transient=False,
            auto_refresh=False,
        )
        live_display.start(refresh=True)

    def _update_live(add_cursor: bool = True) -> None:
        if not live_display:
            return
        suffix = cursor if add_cursor else ""
        live_display.update(
            Align(Markdown(f"{visible_text}{suffix}"), align="left", pad=False),
            refresh=True,
        )

    try:
        for chunk in stream_iterator:
            if not chunk:
                continue
            total_response += chunk
            
            if not first_part_printed:
                first_part_buffer += chunk
                # Strip leading whitespace so a response starting with \n\n
                # doesn't produce an empty panel.
                stripped = first_part_buffer.lstrip()
                if split_marker in stripped:
                    first_part, remainder = stripped.split(split_marker, 1)
                    _print_panel(console, first_part, panel_title, panel_border_style)
                    first_part_printed = True
                    visible_text, overflow_buffer = _split_visible(remainder.lstrip('\n'))
                    _start_live(visible_text)
                    if overflow_buffer:
                        live_frozen = True
                        _update_live()
                continue

            if live_frozen:
                overflow_buffer += chunk
                continue

            candidate = visible_text + chunk
            visible_text, extra = _split_visible(candidate)
            if extra:
                overflow_buffer = extra
                live_frozen = True
            _update_live()

        if not first_part_printed and first_part_buffer:
            _print_panel(console, first_part_buffer, panel_title, panel_border_style)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted![/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error during streaming:[/bold red] {e}")
        total_response += f"\nERROR: {e}"
    finally:
        if live_display:
            _update_live(add_cursor=False)
            live_display.stop()
        if overflow_buffer.strip():
            console.print(Align(Markdown(overflow_buffer.strip()), align="left", pad=False))
    
    return total_response
