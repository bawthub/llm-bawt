"""Codex exec stream patch: event lines larger than the reader limit.

Regression for the 2026-10-23 codex-bridge crash: a Playwright screenshot
tool result arrived as one >1 MiB JSONL line, ``StreamReader.readline()``
raised ``ValueError: Separator is found, but chunk is longer than limit`` and
the whole turn was killed.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import textwrap

import pytest
from openai_codex_sdk import exec as exec_mod
from openai_codex_sdk.errors import CodexExecError

from codex_bridge import exec_patch
from codex_bridge.exec_patch import read_event_line


def _reader(data: bytes, *, limit: int) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=limit)
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def _drain(reader: asyncio.StreamReader, **kwargs) -> list[bytes]:
    lines = []
    while line := await read_event_line(reader, **kwargs):
        lines.append(line)
    return lines


@pytest.mark.asyncio
async def test_line_longer_than_the_reader_limit_is_read_whole():
    big = b"x" * 5000 + b"\n"
    lines = await _drain(_reader(b"a\n" + big + b"b\n", limit=64))
    assert lines == [b"a\n", big, b"b\n"]


@pytest.mark.asyncio
async def test_unterminated_tail_and_eof_match_readline():
    lines = await _drain(_reader(b"one\n" + b"y" * 300, limit=64))
    assert lines == [b"one\n", b"y" * 300]
    assert await read_event_line(_reader(b"", limit=64)) == b""


@pytest.mark.asyncio
async def test_line_over_the_hard_cap_fails_closed():
    with pytest.raises(CodexExecError, match="exceeded 1000 bytes"):
        await read_event_line(_reader(b"z" * 5000 + b"\n", limit=64), max_bytes=1000)


@pytest.mark.asyncio
async def test_patched_run_streams_a_multi_mib_event_line(tmp_path, monkeypatch):
    """End to end through the patched ``CodexExec.run`` with a real child."""
    big = '{"type":"item.completed","data":"' + "A" * (3 * 1024 * 1024) + '"}'
    fake = tmp_path / "codex"
    fake.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        sys.stdin.read()
        sys.stdout.write('{{"type":"thread.started"}}\\n')
        sys.stdout.write('{{"type":"item.completed","data":"' + "A" * {3 * 1024 * 1024} + '"}}\\n')
        sys.stdout.write('{{"type":"turn.completed"}}\\n')
    """))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    # Restore the SDK's original run() after this test.
    monkeypatch.setattr(exec_mod.CodexExec, "run", exec_mod.CodexExec.run)
    exec_patch.install()

    runner = exec_mod.CodexExec(executable_path=str(fake), env=dict(os.environ))
    lines = [line async for line in runner.run(exec_mod.CodexExecArgs(input="hi"))]

    assert lines == ['{"type":"thread.started"}', big, '{"type":"turn.completed"}']
