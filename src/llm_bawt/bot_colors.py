"""Terminal colors for bots.

A bot's ``color`` is its UI color, normally a ``#rrggbb`` hex chosen for the
web app; ``cli_color`` is an optional Rich color name for the terminal. The
CLI uses ``cli_color`` when set, otherwise a legible variant of ``color``.
"""
from __future__ import annotations

import colorsys
import re

from rich.color import Color, ColorParseError

FALLBACK_CLI_COLOR = "green"
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
#: HLS lightness range that reads well on a dark terminal background.
_MIN_LIGHTNESS = 0.62
_MAX_LIGHTNESS = 0.82


def is_valid_cli_color(value: str) -> bool:
    """True when Rich accepts ``value`` as a color."""
    try:
        Color.parse(value)
    except ColorParseError:
        return False
    return True


def normalize_cli_color(value: object) -> str | None:
    """Validate a ``cli_color`` write: blank clears; otherwise Rich must accept it."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not is_valid_cli_color(text):
        raise ValueError(f"cli_color {value!r} is not a color Rich understands")
    return text


def legible_hex(hex_color: str) -> str | None:
    """The same hue with lightness lifted into a readable range, or None if not a hex."""
    match = _HEX.match(hex_color.strip())
    if not match:
        return None
    raw = match.group(1)
    r, g, b = (int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lightness = min(max(lightness, _MIN_LIGHTNESS), _MAX_LIGHTNESS)
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return "#" + "".join(f"{round(c * 255):02x}" for c in (r, g, b))


def cli_color_for(bot: object | None) -> str:
    """The Rich color the CLI should use for ``bot``."""
    if bot is None:
        return FALLBACK_CLI_COLOR
    cli_color = (getattr(bot, "cli_color", None) or "").strip()
    if cli_color and is_valid_cli_color(cli_color):
        return cli_color
    color = (getattr(bot, "color", None) or "").strip()
    if color:
        derived = legible_hex(color)
        if derived:
            return derived
        if is_valid_cli_color(color):
            return color
    return FALLBACK_CLI_COLOR
