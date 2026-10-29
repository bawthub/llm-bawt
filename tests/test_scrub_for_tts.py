"""Tests for scrub_for_tts — the single source of truth for TTS-safe text.

Every bot's chat stream is scrubbed at complete block boundaries before speech,
regardless of voice-optimized prompt style. Markdown can't be scrubbed per token;
these tests lock the "no markdown survives" audio contract.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from llm_bawt.bots import StreamingTTSScrubber, scrub_for_tts, should_scrub_for_tts


def test_tts_scrubbing_is_independent_of_prompt_style_and_voice_selection():
    # Chat voice is selected in the client, including a global fallback or
    # per-user override that the chat request never carries to llm-bawt.
    for optimized in (False, True):
        for voice in (None, "test-voice"):
            bot = SimpleNamespace(voice_optimized=optimized, default_voice=voice)
            assert should_scrub_for_tts(bot)
    assert should_scrub_for_tts(None)


def _assert_no_markdown(out: str) -> None:
    assert "##" not in out, out
    assert "**" not in out, out
    assert "~~" not in out, out
    assert "`" not in out, out
    assert "|" not in out, out
    assert "](" not in out, out
    assert not re.search(r"(?m)^\s*[-*+]\s", out), out       # no unordered bullets
    assert not re.search(r"(?m)^\s*\d+[.)]\s", out), out     # no ordered markers
    assert not re.search(r"(?m)^\s{0,3}#", out), out         # no ATX headers
    assert not re.search(r"(?m)^\s{0,3}>", out), out         # no blockquote markers


def test_headers_stripped():
    assert scrub_for_tts("# Title\n## Sub") == "Title\nSub"


def test_bold_italic_strike():
    out = scrub_for_tts("This is **bold**, __under__ and ~~gone~~ text")
    assert out == "This is bold, under and gone text"


def test_links_keep_text_drop_url():
    out = scrub_for_tts("See the [docs](https://example.com/x) now")
    assert out == "See the docs now"


def test_images_keep_alt():
    out = scrub_for_tts("Look ![a cat](http://img/c.png) here")
    assert out == "Look a cat here"


def test_inline_code_keeps_token():
    out = scrub_for_tts("Run `make deploy` please")
    assert out == "Run make deploy please"


def test_fenced_code_drops_fences_keeps_body():
    src = "Before\n```python\nx = 1\n```\nAfter"
    out = scrub_for_tts(src)
    assert "```" not in out
    assert "x = 1" in out
    assert "python" not in out


def test_unordered_and_ordered_lists():
    src = "- one\n- two\n\n1. first\n2. second"
    out = scrub_for_tts(src)
    assert out == "one\ntwo\n\nfirst\nsecond"


def test_blockquote_and_hr():
    src = "> a warning\n\n---\n\ndone"
    out = scrub_for_tts(src)
    _assert_no_markdown(out)
    assert "a warning" in out and "done" in out


def test_table_flattened():
    src = "| A | B |\n|---|---|\n| 1 | 2 |"
    out = scrub_for_tts(src)
    assert "|" not in out
    assert "---" not in out
    assert "A" in out and "B" in out and "1" in out and "2" in out


def test_emotes_still_removed():
    # Established voice_optimized behavior: *action* emotes are deleted.
    assert scrub_for_tts("Sure *nods* thing") == "Sure thing"
    assert scrub_for_tts("::waves:: hello") == "hello"


def test_kitchen_sink_no_markdown_survives():
    src = (
        "## Summary\n\n"
        "Found **three** issues in `config.py`:\n\n"
        "- Check the [guide](https://x.io/g)\n"
        "- Run `make test`\n\n"
        "1. back up\n2. migrate\n\n"
        "```bash\necho hi\n```\n\n"
        "> note this\n\n"
        "| K | V |\n|---|---|\n| a | b |\n\n"
        "---\n\nAll ~~not~~ set *smiles*."
    )
    _assert_no_markdown(scrub_for_tts(src))


def test_empty_and_plain_passthrough():
    assert scrub_for_tts("") == ""
    assert scrub_for_tts("Just a normal sentence.") == "Just a normal sentence."


def test_streaming_scrubber_flushes_final_paragraph():
    scrubber = StreamingTTSScrubber()

    assert scrubber.feed("First paragraph.\n") == ""
    assert scrubber.feed("\nFinal sentence.") == "First paragraph.\n"
    assert scrubber.flush() == "Final sentence."


def test_streaming_scrubber_single_paragraph_is_terminal_tail():
    scrubber = StreamingTTSScrubber()

    assert scrubber.feed("A complete response. Last sentence.") == ""
    assert scrubber.flush() == "A complete response. Last sentence."
