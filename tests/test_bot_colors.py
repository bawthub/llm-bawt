"""CLI color resolution for bots (TASK-2028)."""
from __future__ import annotations

from types import SimpleNamespace

import colorsys

import pytest

from llm_bawt.bot_colors import cli_color_for, legible_hex, normalize_cli_color


def _bot(color=None, cli_color=None):
    return SimpleNamespace(color=color, cli_color=cli_color)


def _lightness(hex_color):
    raw = hex_color.lstrip("#")
    return colorsys.rgb_to_hls(*(int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)))[1]


def test_cli_color_wins_when_set():
    assert cli_color_for(_bot("#1a1a2e", "magenta")) == "magenta"


def test_dark_hex_is_lifted_to_a_readable_lightness_with_the_same_hue():
    derived = cli_color_for(_bot("#1a1a2e"))
    assert derived.startswith("#") and _lightness(derived) >= 0.6
    hue = lambda h: colorsys.rgb_to_hls(*(int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)))[0]  # noqa: E731
    assert hue(derived) == pytest.approx(hue("#1a1a2e"), abs=0.01)


def test_readable_hex_is_kept():
    assert legible_hex("#a0d8ef") == "#a0d8ef"


@pytest.mark.parametrize(("color", "expected"), [
    ("cyan", "cyan"), ("rose", "green"), (None, "green"), ("", "green")])
def test_named_and_missing_colors(color, expected):
    assert cli_color_for(_bot(color)) == expected


def test_invalid_stored_cli_color_falls_back_to_color():
    assert cli_color_for(_bot("cyan", "onyx")) == "cyan"
    assert cli_color_for(None) == "green"


def test_normalize_rejects_unknown_names_and_clears_blank():
    assert normalize_cli_color("  ") is None
    assert normalize_cli_color(None) is None
    assert normalize_cli_color("Red") == "red"
    with pytest.raises(ValueError):
        normalize_cli_color("rose")
