"""The streaming emote filter and ``strip_emotes`` must give the same text.

Voice-optimized replies are cleaned by ``strip_emotes`` on the non-stream path
and by ``StreamingEmoteFilter`` while streaming; both share one set of emote
rules, and these tests hold the streamed result equal for any chunking.
"""

from __future__ import annotations

import random

import pytest

from llm_bawt.bots import StreamingEmoteFilter, strip_emotes

CORPUS = [
    "",
    "Hello there.",
    "*smiles warmly* Hello there!",
    "Hi *waves* and *nods* bye",
    "2 * 3 = 6, and a * b * c",
    "use std::vector<int> here",
    "::waves:: hello ::bows deeply:: there",
    "Note: this :: that ::x::",
    "(laughs)\nOkay then.",
    "Okay.\n\n(sighs heavily)\n\n\nFine.",
    "Text (inline aside) stays",
    "  (waves)  \n  Hi  ",
    "(multi\nline aside)\nafter",
    "(x) trailing text\nnext",
    "**bold** and *emote* and ***both***",
    "unclosed *star at the end",
    "line one *a\nline two* b",
    "a  \t b\n\n\n\nc   ",
    "*only an emote*",
    "\n\n  leading space *x*\n",
    "end with colon:",
    "end with space colon :",
    "::\n::",
    "(\n)",
    "()",
    "*\n*",
    "* not an emote *",
    "close*then* open",
    "mixed *a*::b:: (c)\n(d)\ne",
]


def _stream(text: str, cuts: list[int]) -> str:
    f = StreamingEmoteFilter()
    out = []
    prev = 0
    for cut in [*cuts, len(text)]:
        out.append(f.process(text[prev:cut]))
        prev = cut
    out.append(f.flush())
    return "".join(out)


@pytest.mark.parametrize("text", CORPUS)
def test_every_two_way_split_matches_strip_emotes(text: str) -> None:
    expected = strip_emotes(text)
    for cut in range(len(text) + 1):
        assert _stream(text, [cut]) == expected, (text, cut)


@pytest.mark.parametrize("text", CORPUS)
def test_char_by_char_matches_strip_emotes(text: str) -> None:
    assert _stream(text, list(range(1, len(text)))) == strip_emotes(text)


def test_random_texts_and_chunkings_match_strip_emotes() -> None:
    rng = random.Random(460)
    alphabet = ["*", "*", ":", ":", "(", ")", " ", " ", "\t", "\n", "\n", "a", "b", "\r"]
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        cuts = sorted(rng.sample(range(len(text) + 1), rng.randint(0, len(text) + 1)))
        assert _stream(text, cuts) == strip_emotes(text), (text, cuts)


def test_settled_text_streams_before_the_end() -> None:
    f = StreamingEmoteFilter()
    assert f.process("Hello *wa") == "Hello"
    assert f.process("ves* there") == " there"
    assert f.flush() == ""


def test_code_like_text_is_not_held() -> None:
    f = StreamingEmoteFilter()
    assert f.process("use std::vector and 2 * 3") == "use std::vector and 2"
    assert f.process(" = 6\n") + f.flush() == " * 3 = 6"
