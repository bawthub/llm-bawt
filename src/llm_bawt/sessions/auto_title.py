"""TASK-256 (M2c): cheap thread auto-title from the first user message.

Pure string ops — NO LLM call. Runs server-side at the single live message
write chokepoint (``mcp_server/storage.py::add_message``) after a user-role
row persists. A thread whose ``title_source`` is still ``default`` (or unset)
gets ``title = <cleaned first message>`` and ``title_source='auto'``; a user
rename sets ``title_source='user'`` (sessions PATCH route) and is never
auto-touched again. Slash commands (``/new`` etc.) are ignored as title
sources — the first real message wins.

Failure policy: titling is cosmetic; nothing here may ever break a message
write. Every entry point swallows and logs.
"""

import logging
import re

logger = logging.getLogger(__name__)

MAX_TITLE_LEN = 60

# PostgreSQL btrim's default is only ASCII space, not Python str.strip().
# Keep the existing title eligibility rule for all Python whitespace.
PYTHON_STRIP_CHARS = (
    "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0 "
    "           "
    "    　"
)

# Positive memo: committed UPDATE/RETURNING evidence or an already-required
# fetched row proved settlement. Missing/wrong-owner rows remain retryable.
# Cold already-titled rows receive one value-preserving physical UPDATE, not
# another one on every subsequent write. This is cosmetic, never authority.
_settled_sessions: set[str] = set()

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*")
_WS_RE = re.compile(r"\s+")


def derive_thread_title(content: str, max_len: int = MAX_TITLE_LEN) -> str | None:
    """Distill a display title from message content, or None if unusable.

    - Slash-commands (``/new``, ``/help arg`` …) are not titles.
    - Code-fence markers and markdown heading/quote prefixes are stripped.
    - Whitespace collapses; truncation breaks on a word boundary with ``…``.
    """
    if not content:
        return None
    text = content.strip()
    if not text or text.startswith("/"):
        return None
    text = _FENCE_RE.sub(" ", text)
    text = text.replace("`", " ")
    # First non-empty line drives the title; drop md heading/quote markers.
    for line in text.splitlines():
        line = line.strip().lstrip("#>*- ").strip()
        if line:
            text = line
            break
    else:
        return None
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return None
    if len(text) <= max_len:
        return text
    cut = text[: max_len + 1]
    space = cut.rfind(" ")
    if space > max_len // 2:
        cut = cut[:space]
    else:
        cut = text[:max_len]
    return cut.rstrip(" ,.;:!?") + "…"


def session_title_is_settled(metadata) -> bool:
    """Recognize positive title evidence from a trusted stored object."""
    meta = metadata or {}
    if not isinstance(meta, dict):
        return False
    source = meta.get("title_source") or "default"
    title = meta.get("title")
    return source != "default" or bool(isinstance(title, str) and title.strip())


def remember_session_title(row: dict | None) -> None:
    """Reuse an already-fetched row; never fetch solely to prime the memo.

    This is cosmetic suppression only, not ownership or write authorization.
    Missing/untitled/malformed-title rows stay retryable. The existing memo
    assumes a settled title is not reset to default during this process.
    """
    if row and row.get("id") and session_title_is_settled(row.get("session_metadata")):
        _settled_sessions.add(row["id"])


def maybe_auto_title(
    manager, session_id: str | None, content: str, *, user_id: str | None = None,
) -> None:
    """Derive and conditionally persist a title without a session pre-read.

    The manager checks title eligibility and owner in one native UPDATE.
    Caller invokes only after a successful user-role message write. Never raises.
    """
    if not session_id or session_id in _settled_sessions:
        return
    try:
        title = derive_thread_title(content)
        if title is None:
            return  # slash command / empty — leave the thread for a real message
        ok = manager.try_set_session_auto_title(session_id, title, user_id=user_id)
        if ok:
            _settled_sessions.add(session_id)
            logger.info("session title settled for %s", session_id)
    except Exception as e:  # cosmetic feature — never break the write path
        logger.warning("auto-title failed for session %s: %s", session_id, e)
