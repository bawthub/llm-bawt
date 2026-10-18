"""Parse Playwright function arguments before execution; never evaluate them.

A syntax error is a deterministic tool failure, so the existing PreToolUse hook
can return a model-visible correction without sending malformed code to MCP.
Unavailable/slow Node leaves the ordinary Playwright tool result in charge.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

_FIELDS = {
    "mcp__playwright__browser_evaluate": "function",
    "mcp__playwright__browser_run_code_unsafe": "code",
}
# The bridge image already runs Node; no project files, external imports, or
# execution of user code. vm.Script compiles an expression without evaluating it.
_PARSE_SCRIPT = """
const fs = require('node:fs');
const vm = require('node:vm');
try {
  new vm.Script('(\\n' + fs.readFileSync(0, 'utf8') + '\\n)', {filename: 'playwright-function'});
} catch (error) {
  if (!(error instanceof SyntaxError)) process.exit(2);
  process.stderr.write(error.message + '\\n' + (error.stack || '').split('\\n')[0]);
  process.exit(1);
}
"""
_MAX_SCRIPT_BYTES = 64 * 1024
_PARSE_TIMEOUT_S = 1.0


def _syntax_error(stderr: bytes, field: str) -> str:
    lines = stderr.decode("utf-8", errors="replace").splitlines()
    message = (lines[0] if lines else "Invalid JavaScript syntax")[:180]
    location = ""
    if len(lines) > 1:
        match = re.fullmatch(r"playwright-function:(\d+)", lines[1].strip())
        if match:
            location = f" at line {max(1, int(match.group(1)) - 1)}"
    return (
        f"[Playwright JavaScript syntax error in `{field}`{location}: {message}. "
        "The browser tool was not executed. Correct the JavaScript function "
        "(remove any stray words/tokens outside it) and submit a NEW tool call.]"
    )


async def _stop_parser(process: asyncio.subprocess.Process | None) -> None:
    """Kill a timed-out parser; never stall the SDK hook during pipe cleanup."""
    if process is None:
        return
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass  # Exit raced the timeout; still drain its pipes.
    try:
        await asyncio.wait_for(process.communicate(), timeout=0.25)
    except (OSError, RuntimeError, asyncio.TimeoutError):
        logger.warning("Playwright syntax parser cleanup did not finish", exc_info=True)


async def playwright_syntax_error(tool_name: str, tool_input: dict) -> str | None:
    """Return a model-visible denial only on a confirmed JavaScript SyntaxError."""
    field = _FIELDS.get(tool_name)
    if field is None:
        return None
    source = tool_input.get(field)
    if not isinstance(source, str):
        return None
    try:
        encoded = source.encode("utf-8")
    except UnicodeError:
        return None  # Preserve the normal policy gate; MCP owns invalid input.
    if len(encoded) > _MAX_SCRIPT_BYTES:
        return None  # Tool schema/MCP server still owns huge inputs.
    # `fn [sic]` parses as property access but the editorial marker raises
    # ReferenceError when executed. A marker inside a trailing line comment
    # is valid code, not executable: do not flag it as an error.
    marker = re.search(r"[})]\s+\[sic\]\s*;?\s*$", source)
    if marker and "//" not in source[source.rfind("\n", 0, marker.start()) + 1:marker.start()]:
        return (
            f"[Playwright JavaScript in `{field}` has a stray trailing `[sic]` marker. "
            "The browser tool was not executed. Remove the marker and submit "
            "a NEW tool call.]"
        )
    process = None
    try:
        async with asyncio.timeout(_PARSE_TIMEOUT_S):
            process = await asyncio.create_subprocess_exec(
                "node", "-e", _PARSE_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate(encoded)
    except asyncio.CancelledError:
        await _stop_parser(process)
        raise
    except (OSError, TimeoutError):
        await _stop_parser(process)
        logger.warning("Playwright syntax preflight unavailable; deferring to MCP tool", exc_info=True)
        return None
    if process.returncode == 1:
        return _syntax_error(stderr, field)
    if process.returncode != 0:
        logger.warning("Playwright syntax preflight exited %s; deferring to MCP tool", process.returncode)
    return None
