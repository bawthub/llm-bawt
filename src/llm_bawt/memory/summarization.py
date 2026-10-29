"""History summarization for llm-bawt.

This module provides session detection and summarization functionality
to condense older conversation history, allowing more context to fit
in the LLM window.

Summarization is budget-driven: sessions are only summarized when their
messages would be dropped from context due to token budget overflow.
Sessions are detected by timestamp gaps (default 1 hour).
Summaries are non-destructive: raw messages remain in the messages table.
"""

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import httpx

from .speaker_attribution import speaker_of, summary_label

if TYPE_CHECKING:
    from .summarization import Session

logger = logging.getLogger(__name__)

# Summarization prompt template
SUMMARIZATION_PROMPT = """Summarize this conversation session for future continuation.

Requirements:
0. Be strictly faithful to source material. Do not sanitize, soften, editorialize, or introduce bias. Do not omit concrete facts due to style preferences.
1. Capture key topics, decisions, and unresolved threads.
2. Capture user intent/goals (what they were trying to accomplish).
3. Capture emotional tone/mood trajectory and communication style cues.
4. Keep factual continuity anchors (names, constraints, preferences stated in-session).
5. Preserve critical implementation details when present (files, commands, config values, errors, fixes, decisions).
6. Do not add facts not present in the conversation.

Output format (plain text, exactly these sections):
Summary:
Key Details:
Intent:
Tone:
Open Loops:

7. Start directly with concrete facts. Avoid boilerplate lead-ins like "The conversation begins with", "The conversation revolves around", or "The conversation is about".
8. Prefer dense wording over scene-setting. If a short noun phrase is clearer than a full sentence, use it.

Target detail level: concise but concrete (about 120-220 words total across all sections).
Keep this materially shorter than the source dialog while preserving concrete facts. Do not pad with filler.

Conversation:
{messages}
"""


BATCH_SUMMARIZATION_PROMPT = """You are summarizing multiple conversation sessions together.

Goal:
1. Produce one concise structured summary for EACH session.
2. Use cross-session awareness to enrich each summary with recurring patterns/themes.
3. Do not invent facts.
4. Be strictly faithful to source material. Do not sanitize, soften, editorialize, or introduce bias.

Return STRICT JSON only (no markdown, no prose):
{{
    "global_themes": ["theme 1", "theme 2"],
    "summaries": [
        {{
            "session_index": 1,
            "summary": "...",
            "key_details": "...",
            "intent": "...",
            "tone": "...",
            "open_loops": "...",
            "cross_session_links": "..."
        }}
    ]
}}

Constraints:
- Keep each session concise but specific (~90-180 words across fields).
- "cross_session_links" should mention recurring user goals/patterns if present; otherwise empty string.
- session_index must match input ordering exactly.
- Start directly with concrete facts. Avoid lead-ins like "The conversation begins with" or "The conversation revolves around".
- Prefer compact, information-dense wording over narrative filler.

Sessions:
{sessions_blob}
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0:
        return None

    # If no closing brace is present, attempt best-effort brace completion.
    if end <= start:
        snippet = candidate[start:]
        open_braces = snippet.count("{")
        close_braces = snippet.count("}")
        open_brackets = snippet.count("[")
        close_brackets = snippet.count("]")
        if open_brackets > close_brackets:
            snippet += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            snippet += "}" * (open_braces - close_braces)
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_summary_sections(text: str) -> dict[str, str]:
    """Extract structured fields from summary text."""
    sections = {
        "summary": "",
        "key_details": "",
        "intent": "",
        "tone": "",
        "open_loops": "",
    }
    if not text:
        return sections

    current_key: str | None = None
    buffers: dict[str, list[str]] = {k: [] for k in sections}
    key_map = {
        "summary:": "summary",
        "key details:": "key_details",
        "key_details:": "key_details",
        "intent:": "intent",
        "tone:": "tone",
        "open loops:": "open_loops",
        "open_loops:": "open_loops",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_key:
                buffers[current_key].append("")
            continue
        if line.lower() in {"[historical summary]", "[active session summary]"}:
            continue
        normalized = line.lower()
        matched = False
        for prefix, key in key_map.items():
            if normalized.startswith(prefix):
                current_key = key
                value = line[len(prefix):].strip()
                if value:
                    buffers[key].append(value)
                matched = True
                break
        if matched:
            continue
        if current_key:
            buffers[current_key].append(line)
        else:
            buffers["summary"].append(line)

    for key, lines in buffers.items():
        value = "\n".join(lines).strip()
        sections[key] = value

    return sections


def normalize_structured_summary_text(text: str) -> str:
    """Normalize summary text into consistent sectioned format."""
    data = extract_summary_sections(text)
    return _render_structured_summary_text(data)

def compress_structured_summary_text(
    text: str,
    source_session: "Session | None" = None,
) -> str:
    """Normalize summary sections without truncating stored content."""
    _ = source_session  # Signature retained for compatibility with existing callers.
    sections = extract_summary_sections(text)
    return _render_structured_summary_text(sections)


def _clean_summary_section_value(section: str, value: str) -> str:
    """Reduce boilerplate in structured summaries without dropping facts."""
    cleaned = " ".join((value or "").strip().split())
    if not cleaned:
        return ""

    if section == "summary":
        boilerplate_prefixes = (
            "the conversation begins with ",
            "the conversation starts with ",
            "the conversation revolves around ",
            "the conversation is about ",
            "the conversation is ",
            "this conversation begins with ",
            "this conversation starts with ",
            "this conversation is about ",
            "this conversation is ",
            "the exchange begins with ",
            "the exchange starts with ",
            "the session begins with ",
            "the session starts with ",
        )
        lowered = cleaned.lower()
        stripped_boilerplate = False
        for prefix in boilerplate_prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip()
                stripped_boilerplate = True
                break

        # If stripping left a leading article, keep it terse and natural.
        cleaned = re.sub(r"^(?:that|which)\s+", "", cleaned, flags=re.IGNORECASE)
        if stripped_boilerplate and cleaned[:1].islower():
            cleaned = cleaned[:1].upper() + cleaned[1:]

    return cleaned


def _render_structured_summary_text(
    sections: dict[str, str],
    *,
    include_empty: bool = False,
) -> str:
    """Render structured summary sections in a compact, stable format."""
    labels = (
        ("summary", "Summary"),
        ("key_details", "Key Details"),
        ("intent", "Intent"),
        ("tone", "Tone"),
        ("open_loops", "Open Loops"),
    )
    lines: list[str] = []
    for key, label in labels:
        value = _clean_summary_section_value(key, sections.get(key) or "")
        if key == "summary" or include_empty or value:
            lines.append(f"{label}: {value}".rstrip())
    return "\n".join(lines).strip()


def is_summary_low_quality(text: str, source_session: "Session | None" = None) -> bool:
    """Heuristic gate for vague/underspecified summaries.

    Reject summaries that are too short or dominated by generic wording.
    When *source_session* is provided, thresholds are scaled to the actual
    conversation weight so trivial sessions (e.g. "hi" / "testing") don't
    trigger expensive per-session retry cascades for content that genuinely
    has nothing more to say.
    """
    sections = extract_summary_sections(text)
    summary = (sections.get("summary") or "").strip()
    key_details = (sections.get("key_details") or "").strip()
    intent = (sections.get("intent") or "").strip()
    open_loops = (sections.get("open_loops") or "").strip()

    total_len = len(summary) + len(key_details) + len(intent) + len(open_loops)

    # For sessions with very little real content, accept shorter summaries.
    source_chars = 0
    if source_session is not None:
        source_chars = sum(len(m.get("content", "")) for m in source_session.messages)

    # Trivial sessions (< 200 chars of source content) → accept anything non-empty.
    if source_chars > 0 and source_chars < 200:
        return total_len < 20

    # Light sessions (< 600 chars) → lenient thresholds.
    if source_chars > 0 and source_chars < 600:
        return total_len < 80

    if total_len < 260:
        return True

    if len(key_details) < 120:
        return True

    generic_patterns = [
        r"\bgame discussion\b",
        r"\btesting\b",
        r"\bplayful\b",
        r"\bvent\/motivation\b",
        r"\bmisc\b",
        r"\bvarious topics\b",
    ]
    merged = f"{summary}\n{key_details}\n{intent}\n{open_loops}".lower()
    if any(re.search(pat, merged) for pat in generic_patterns):
        return True

    detail_signals = 0
    if re.search(r"\b(error|exception|command|config|file|path|api|model|session|timestamp|search)\b", merged):
        detail_signals += 1
    if re.search(r"\b\d{2,}\b", merged):
        detail_signals += 1
    if re.search(r"[`'\"]", merged):
        detail_signals += 1
    if detail_signals == 0:
        return True

    return False


@dataclass
class Session:
    """A detected conversation session.

    TASK-641: ``session_id`` is the durable DB thread the messages belong to
    when the window was cut along thread boundaries (the live path). ``None``
    only for legacy gap-detected windows (re-split tooling / NULL-session
    fallback).
    """
    start_timestamp: float
    end_timestamp: float
    messages: list[dict]
    message_ids: list[str]
    session_id: str | None = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def end_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.end_timestamp)


def resolve_source_session_ids(
    conn: Any, table_name: str, message_ids: list[str] | None
) -> list[str]:
    """Distinct durable-thread ids the given source messages belong to.

    TASK-284 step 13: summary provenance. Returns the sorted DISTINCT non-NULL
    ``session_id`` values of the source rows so a summary can record which DB
    thread(s) it compacted. A time-gap-detected summarization window can
    straddle thread rotations, hence a list. Best-effort: an empty list simply
    means the sources predate session linkage (nothing determinable).
    """
    if not message_ids:
        return []
    from sqlalchemy import text

    rows = conn.execute(
        text(
            f"""
            SELECT DISTINCT session_id FROM {table_name}
            WHERE id = ANY(:ids) AND session_id IS NOT NULL
            """
        ),
        {"ids": list(message_ids)},
    ).fetchall()
    return sorted(str(r[0]) for r in rows)


def detect_sessions(
    messages: list[dict],
    session_gap_seconds: int = 3600,
) -> list[Session]:
    """Detect conversation sessions by splitting on timestamp gaps.

    Args:
        messages: List of message dicts with 'id', 'role', 'content', 'timestamp'
        session_gap_seconds: Gap between messages to consider them in different sessions

    Returns:
        List of Session objects, sorted by start timestamp
    """
    if not messages:
        return []

    # Sort by timestamp
    sorted_msgs = sorted(messages, key=lambda m: m.get("timestamp", 0))

    sessions: list[Session] = []
    current_session_msgs: list[dict] = []
    current_session_ids: list[str] = []
    session_start: float | None = None
    last_timestamp: float | None = None

    for msg in sorted_msgs:
        timestamp = msg.get("timestamp", 0)
        msg_id = msg.get("id", "")

        # Skip summary rows - they shouldn't be counted in session detection
        if msg.get("role") == "summary":
            continue

        # Check if this starts a new session
        if last_timestamp is not None and (timestamp - last_timestamp) > session_gap_seconds:
            # Save the current session
            if current_session_msgs and session_start is not None:
                sessions.append(Session(
                    start_timestamp=session_start,
                    end_timestamp=last_timestamp,
                    messages=current_session_msgs,
                    message_ids=current_session_ids,
                ))
            # Start a new session
            current_session_msgs = []
            current_session_ids = []
            session_start = None

        # Add message to current session
        current_session_msgs.append(msg)
        current_session_ids.append(msg_id)
        if session_start is None:
            session_start = timestamp
        last_timestamp = timestamp

    # Don't forget the last session
    if current_session_msgs and session_start is not None and last_timestamp is not None:
        sessions.append(Session(
            start_timestamp=session_start,
            end_timestamp=last_timestamp,
            messages=current_session_msgs,
            message_ids=current_session_ids,
        ))

    return sessions


def group_messages_by_thread(
    messages: list[dict],
    session_gap_seconds: int = 3600,
) -> list[Session]:
    """Group messages into Sessions along durable-thread boundaries (TASK-641).

    The durable ``session_id`` stamped on every message row IS the
    conversation boundary — one Session per distinct thread, so a
    summarization window can never span threads and every summary gets a
    structural (not incidental) session_id stamp.

    Messages without a ``session_id`` (legacy rows predating thread linkage)
    fall back to timestamp-gap detection (``detect_sessions``) so they still
    summarize; in the live deployment that set is empty (TASK-630 verified
    0 orphans).

    Returns Sessions sorted by start timestamp, mirroring detect_sessions.
    """
    if not messages:
        return []

    sorted_msgs = sorted(messages, key=lambda m: (m.get("timestamp", 0), str(m.get("id", ""))))

    by_thread: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for msg in sorted_msgs:
        if msg.get("role") == "summary":
            continue
        sid = msg.get("session_id")
        if sid:
            by_thread.setdefault(str(sid), []).append(msg)
        else:
            orphans.append(msg)

    sessions: list[Session] = []
    for sid, msgs in by_thread.items():
        sessions.append(Session(
            start_timestamp=msgs[0].get("timestamp", 0),
            end_timestamp=msgs[-1].get("timestamp", 0),
            messages=msgs,
            message_ids=[m.get("id", "") for m in msgs],
            session_id=sid,
        ))

    if orphans:
        sessions.extend(detect_sessions(orphans, session_gap_seconds))

    return sorted(sessions, key=lambda s: s.start_timestamp)


def find_budget_overflow_sessions(
    all_messages: list[dict],
    session_gap_seconds: int = 3600,
    max_context_tokens: int = 0,
    protected_recent_turns: int = 3,
    min_messages_per_session: int = 2,
) -> list[Session]:
    """Find sessions whose messages would be dropped from context due to budget overflow.

    Instead of using a time-based cutoff, this computes which messages won't fit
    in the token budget and returns the sessions they belong to.

    Args:
        all_messages: All message dicts (user/assistant only, no summaries or recalled).
        session_gap_seconds: Gap between messages to split into sessions.
        max_context_tokens: Total token budget for context. 0 = no budget, nothing to summarize.
        protected_recent_turns: Number of recent user+assistant pairs always kept raw.
        min_messages_per_session: Minimum messages in a session for it to be summarizable.

    Returns:
        List of Session objects whose messages would overflow the budget, oldest first.
    """
    if max_context_tokens <= 0 or not all_messages:
        return []

    # Sort chronologically
    sorted_msgs = sorted(all_messages, key=lambda m: (m.get("timestamp", 0), str(m.get("id", ""))))

    # Separate protected recent messages from the rest
    n_protected = min(protected_recent_turns * 2, len(sorted_msgs))
    protected = sorted_msgs[-n_protected:] if n_protected > 0 else []
    candidates = sorted_msgs[:-n_protected] if n_protected > 0 else list(sorted_msgs)

    if not candidates:
        return []

    # Estimate token cost of protected messages
    protected_cost = sum(_estimate_msg_tokens(m) for m in protected)

    # Reserve budget for system message overhead (~500 tokens) and existing summaries
    system_overhead = 500
    remaining_budget = max(0, max_context_tokens - protected_cost - system_overhead)

    # Walk candidates newest-to-oldest, accumulating cost.
    # Messages that fit stay raw; the rest overflow and need summarization.
    fits_budget = 0
    for msg in reversed(candidates):
        cost = _estimate_msg_tokens(msg)
        if fits_budget + cost <= remaining_budget:
            fits_budget += cost
        else:
            break
    else:
        # Everything fits — nothing to summarize
        return []

    # Find the overflow boundary index in candidates
    overflow_end_idx = len(candidates)
    accumulated = 0
    for i in range(len(candidates) - 1, -1, -1):
        cost = _estimate_msg_tokens(candidates[i])
        if accumulated + cost <= remaining_budget:
            accumulated += cost
        else:
            overflow_end_idx = i + 1
            break

    overflow_messages = candidates[:overflow_end_idx]
    if not overflow_messages:
        return []

    # TASK-641: cut windows along durable-thread boundaries, not time gaps —
    # a window can never span threads, so every summary stamps one session_id.
    # (Gap detection survives only as the fallback for NULL-session legacy
    # rows inside group_messages_by_thread, and for the re-split tooling.)
    sessions = group_messages_by_thread(overflow_messages, session_gap_seconds)

    # Filter to sessions with enough messages to be worth summarizing
    return [s for s in sessions if s.message_count >= min_messages_per_session]


def _estimate_msg_tokens(msg: dict) -> int:
    """Estimate token cost of a single message dict (4 chars/token + overhead)."""
    content = msg.get("content", "") or ""
    return len(content) // 4 + 4


def estimate_session_token_savings(session: Session) -> int:
    """Estimate token savings from replacing a session with a summary.

    This is a heuristic for prioritization, not an exact tokenizer measurement.
    """
    raw_tokens = estimate_tokens(format_session_for_summarization(session))
    # Typical summary target is short but non-trivial; clamp to a practical range.
    expected_summary_tokens = min(200, max(48, raw_tokens // 5))
    return max(raw_tokens - expected_summary_tokens, 0)


def prioritize_summarizable_sessions(sessions: list[Session]) -> list[Session]:
    """Prioritize sessions by token savings impact (largest first)."""
    return sorted(
        sessions,
        key=lambda s: (-estimate_session_token_savings(s), s.start_timestamp),
    )


def format_session_for_summarization(session: Session) -> str:
    """Format a session's messages as text for the summarization prompt."""
    lines = []
    for msg in session.messages:
        # Label by durable authorship, not role: bot deliveries are role=user
        # but must never read as the user's words (TASK-1724).
        speaker = speaker_of(msg)
        if speaker.kind == "system":
            continue  # Skip system messages in summarization
        content = msg.get("content", "").strip()
        lines.append(f"{summary_label(speaker)}: {content}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4


def split_session_into_chunks(
    session: Session,
    max_tokens_per_chunk: int = 4000,  # Leave room for prompt template and response
) -> list[Session]:
    """Split a large session into smaller chunks that fit within token limits.
    
    Each chunk maintains chronological order and is treated as a sub-session
    that will be summarized independently.
    
    Args:
        session: The session to split
        max_tokens_per_chunk: Maximum tokens per chunk (conservative estimate)
        
    Returns:
        List of Session objects (sub-sessions), or [session] if no split needed
    """
    if max_tokens_per_chunk <= 0:
        return [session]

    # Format to estimate size
    full_text = format_session_for_summarization(session)
    total_tokens = estimate_tokens(full_text)
    
    # If it fits, return as-is
    if total_tokens <= max_tokens_per_chunk:
        return [session]
    
    # Split into chunks
    chunks = []
    current_msgs = []
    current_ids = []
    current_tokens = 0
    
    for msg, msg_id in zip(session.messages, session.message_ids):
        # Estimate this message's tokens
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_text = f"{role}: {content}"
        msg_tokens = estimate_tokens(msg_text)
        
        # If adding this message would exceed limit, start new chunk
        if current_tokens + msg_tokens > max_tokens_per_chunk and current_msgs:
            # Create chunk from current messages
            chunks.append(Session(
                start_timestamp=current_msgs[0].get("timestamp", 0),
                end_timestamp=current_msgs[-1].get("timestamp", 0),
                messages=current_msgs.copy(),
                message_ids=current_ids.copy(),
                session_id=session.session_id,
            ))
            current_msgs = []
            current_ids = []
            current_tokens = 0
        
        current_msgs.append(msg)
        current_ids.append(msg_id)
        current_tokens += msg_tokens
    
    # Don't forget the last chunk
    if current_msgs:
        chunks.append(Session(
            start_timestamp=current_msgs[0].get("timestamp", 0),
            end_timestamp=current_msgs[-1].get("timestamp", 0),
            messages=current_msgs.copy(),
            message_ids=current_ids.copy(),
            session_id=session.session_id,
        ))
    
    logger.info(f"Split session with {session.message_count} messages into {len(chunks)} chunks")
    return chunks


def summarize_session_with_llm(
    session: Session,
    service_url: str = "http://127.0.0.1:8642",
    timeout: float = 120.0,  # Increased from 30s - small models need more time
    config: Any | None = None,
) -> str | None:
    """Summarize a session using the LLM service.

    Calls /v1/llm/complete which uses the already-loaded model in the service.

    Args:
        session: The session to summarize
        service_url: Base URL of the llm-service
        timeout: Request timeout in seconds

    Returns:
        Summary text if successful, None on failure
    """
    conversation_text = format_session_for_summarization(session)
    from ..prompt_registry import PromptResolver

    prompt = PromptResolver(config).render(
        key="history.summarization.single",
        variables={"messages": conversation_text},
    )
    
    # Log the prompt size for debugging
    logger.debug(f"Summarization prompt: {len(prompt)} chars for {session.message_count} messages")

    try:
        response = httpx.post(
            f"{service_url}/v1/llm/complete",
            json={
                "prompt": prompt,
                "system": "You are a helpful assistant that summarizes conversations concisely.",
                "max_tokens": 320,
                "temperature": 0.3,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("content", "").strip()
        
        # Check for error responses that got saved as content
        error_indicators = [
            "Error:",
            "exception occurred",
            "exceed context window",
            "tokens exceed",
            "CUDA error",
            "out of memory",
        ]
        for indicator in error_indicators:
            if indicator.lower() in content.lower():
                logger.error(f"LLM returned error in content: {content[:100]}...")
                return None
        
        return content
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            logger.warning("LLM service has no model loaded")
        else:
            logger.error(f"LLM service error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.TimeoutException:
        logger.error(f"LLM service timed out after {timeout}s - model may be slow or overloaded")
        return None
    except httpx.RequestError as e:
        logger.error(f"LLM service connection error: {e}")
        return None


SUMMARY_SYSTEM_PROMPT = "You are a helpful assistant that summarizes conversations concisely."

# Substrings that mean the "summary" is really an error message that leaked
# into the completion body — never persist those as a summary.
_SUMMARY_ERROR_INDICATORS = (
    "error:",
    "exception occurred",
    "exceed context window",
    "tokens exceed",
    "cuda error",
    "out of memory",
)


def summarize_session_with_client(
    session: Session,
    client: Any,
    config: Any | None = None,
    *,
    max_tokens: int = 320,
    temperature: float = 0.3,
    max_prompt_tokens: int = 6000,
) -> str | None:
    """Summarize a session with an in-process LLM client (TASK-857).

    The ONE summarize-with-an-LLM unit shared by the background
    HISTORY_SUMMARIZATION job and the ``/new`` pre-seed summarize. ``client``
    is whatever ``_get_background_client`` handed out — the global job model,
    never a chat model or a bot identity.

    Returns the summary text, or ``None`` when the caller should fall back to
    :func:`summarize_session_heuristic` (no client, prompt too large, error).
    """
    if client is None:
        return None
    from ..models.message import Message
    from ..prompt_registry import PromptResolver

    conversation_text = format_session_for_summarization(session)
    prompt = PromptResolver(config).render(
        key="history.summarization.single",
        variables={"messages": conversation_text},
    )
    # Conservative budget to reduce context overflows on smaller models.
    if len(prompt) // 4 > max_prompt_tokens:
        logger.warning(
            "Summarization prompt too large (%d chars) for session %s; skipping LLM",
            len(prompt), session.session_id,
        )
        return None
    try:
        response = client.query(
            messages=[
                Message(role="system", content=SUMMARY_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            plaintext_output=True,
            stream=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"History summarization LLM call failed: {e}")
        return None
    content = (response or "").strip()
    if not content:
        return None
    lower = content.lower()
    if any(ind in lower for ind in _SUMMARY_ERROR_INDICATORS):
        logger.error(f"LLM returned error in content: {content[:100]}...")
        return None
    return content


def summarize_session_heuristic(session: Session) -> str:
    """Generate a heuristic summary without LLM.

    This is a fallback when the LLM service is unavailable.
    Produces the same structured sections as the LLM prompt contract.
    """
    def clean(text: str) -> str:
        return " ".join((text or "").strip().split())

    def clip(text: str, limit: int = 140) -> str:
        text = clean(text)
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    user_messages = [clean(msg.get("content", "")) for msg in session.messages if speaker_of(msg).is_human_user]
    assistant_messages = [clean(msg.get("content", "")) for msg in session.messages if msg.get("role") == "assistant"]
    user_messages = [m for m in user_messages if m]
    assistant_messages = [m for m in assistant_messages if m]

    start_label = session.start_datetime.strftime("%Y-%m-%d %H:%M")
    end_label = session.end_datetime.strftime("%H:%M")
    duration_min = max(int(session.duration_seconds // 60), 0)
    header = f"Session on {start_label}-{end_label} ({duration_min}m, {session.message_count} msgs)."

    if not user_messages:
        return (
            f"Summary: {header}\n"
            "Intent: No clear user intent detected.\n"
            "Tone: Neutral.\n"
            "Open Loops: None identified."
        )

    first_user = clip(user_messages[0], 100)
    last_user = clip(user_messages[-1], 100)

    thread = f"Primary thread started with: \"{first_user}\"."
    if last_user != first_user:
        thread += f" Ended around: \"{last_user}\"."

    intent_candidates = [
        m for m in reversed(user_messages)
        if "?" in m or any(k in m.lower() for k in ("need", "want", "can you", "please", "should", "help"))
    ]
    intent_text = clip(intent_candidates[0], 120) if intent_candidates else clip(last_user, 120)

    all_user_text = " ".join(user_messages).lower()
    frustrated_markers = ("wtf", "fuck", "frustrat", "annoy", "upset", "angry", "bullshit")
    warm_markers = ("thanks", "thank you", "good call", "appreciate", "cool", "sounds good")
    if any(marker in all_user_text for marker in frustrated_markers):
        tone = "Tense/frustrated at points; direct language."
    elif any(marker in all_user_text for marker in warm_markers):
        tone = "Mostly cooperative and positive."
    else:
        tone = "Neutral-pragmatic."

    recent_questions = [clip(m, 120) for m in user_messages[-8:] if "?" in m]
    if recent_questions:
        open_loops = "; ".join(recent_questions[-2:])
    elif assistant_messages:
        open_loops = f"Confirm outcome/follow-up from assistant's latest guidance: \"{clip(assistant_messages[-1], 120)}\"."
    else:
        open_loops = "None explicit."

    return (
        f"Summary: {header} {thread}\n"
        f"Intent: {intent_text}\n"
        f"Tone: {tone}\n"
        f"Open Loops: {open_loops}"
    )


class HistorySummarizer:
    """Manages history summarization for a bot.

    Usage:
        summarizer = HistorySummarizer(config, bot_id="nova")

        # Dry run to preview
        preview = summarizer.preview_summarizable_sessions()

        # Actually summarize
        results = summarizer.summarize_eligible_sessions()
    """

    def __init__(
        self,
        config: Any,
        bot_id: str = "nova",
        service_url: str | None = None,
        summarize_fn: "Callable[[Session], str | None] | None" = None,
        summarize_batch_fn: "Callable[[list[Session]], dict[int, str] | None] | None" = None,
        settings_getter: "Callable[[str, Any], Any] | None" = None,
        max_context_tokens: int = 0,
    ):
        self.config = config
        self.bot_id = bot_id
        self._settings_getter = settings_getter

        # Get service URL from config or use default
        host = getattr(config, "SERVICE_HOST", "127.0.0.1")
        port = getattr(config, "SERVICE_PORT", 8642)
        self.service_url = service_url or f"http://{host}:{port}"

        # Custom summarization function (provided by API to avoid HTTP self-call)
        self._summarize_fn = summarize_fn
        self._summarize_batch_fn = summarize_batch_fn

        # Budget-driven summarization settings — TASK-610: the ONE global-only
        # Tier-1 job dict (was three per-bot _setting reads with legacy env
        # fallbacks). Job scheduling params are global, not bot-aware.
        job = config.resolve_summarization_job()
        self.session_gap_seconds = job["session_gap_seconds"]
        self.min_messages_per_session = job["min_messages_per_session"]
        self.protected_recent_turns = int(job["protected_recent_turns"])
        # Token budget: explicit override > Tier-2 budget authority. TASK-615:
        # the legacy max_context_tokens runtime/env read is DELETED (that key
        # became history_tokens, the Tier-3 raw-bucket cap — a different
        # concept). The summarizer's overflow budget comes from the ONE Tier-2
        # authority when the caller doesn't pass one.
        if max_context_tokens > 0:
            self.max_context_tokens = max_context_tokens
        else:
            budget_fn = getattr(config, "resolve_context_budget", None)
            self.max_context_tokens = budget_fn(None)[2] if budget_fn else 0

        # Initialize the PostgreSQL backend
        from .postgresql import PostgreSQLMemoryBackend
        self._backend = PostgreSQLMemoryBackend(config, bot_id)

    def _setting(self, key: str, fallback: Any) -> Any:
        """Resolve a setting through the runtime settings getter if available."""
        if self._settings_getter:
            return self._settings_getter(key, fallback)
        return fallback

    def _get_all_messages(self) -> list[dict]:
        """Fetch all messages from the database."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT id, role, content, timestamp, summarized, recalled_history, summary_metadata, session_id,
                       author_entity_type, author_entity_id
                FROM {self._backend._messages_table_name}
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql).fetchall()
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "summarized": row.summarized,
                    "recalled_history": row.recalled_history,
                    "summary_metadata": row.summary_metadata,
                    "session_id": row.session_id,
                    "author_entity_type": row.author_entity_type,
                    "author_entity_id": row.author_entity_id,
                }
                for row in rows
            ]

    def _find_existing_summary_ids(self, conn: Any, session: Session) -> list[str]:
        """Find historical summaries that already cover this exact message set."""
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT id, summary_metadata
            FROM {self._backend._messages_table_name}
            WHERE role = 'summary'
              AND timestamp >= :start_ts - 1
              AND timestamp <= :end_ts + 1
            ORDER BY timestamp DESC
            """
        )
        rows = conn.execute(
            sql,
            {"start_ts": session.start_timestamp, "end_ts": session.end_timestamp},
        ).fetchall()

        matching_ids: list[str] = []
        for row in rows:
            meta = row.summary_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue

            if not isinstance(meta, dict):
                continue

            if meta.get("summary_type") != "historical":
                continue

            existing_ids = meta.get("message_ids", [])
            if isinstance(existing_ids, list) and existing_ids == session.message_ids:
                matching_ids.append(row.id)

        return matching_ids

    def _find_replaceable_summary_ids(self, conn: Any, session: Session) -> list[str]:
        """Find historical summary rows that should be replaced for this session/chunk.

        This is intentionally broader than exact message-id matching so rebuild mode can
        replace legacy summaries and previously unchunked summaries that overlap the
        same source messages.
        """
        from sqlalchemy import text

        sql = text(
            f"""
            SELECT id, summary_metadata, timestamp
            FROM {self._backend._messages_table_name}
            WHERE role = 'summary'
              AND timestamp >= :start_ts - 1
              AND timestamp <= :end_ts + 1
            ORDER BY timestamp DESC
            """
        )
        rows = conn.execute(
            sql,
            {"start_ts": session.start_timestamp, "end_ts": session.end_timestamp},
        ).fetchall()

        chunk_ids = set(session.message_ids)
        replace_ids: list[str] = []

        for row in rows:
            meta = row.summary_metadata or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            if not isinstance(meta, dict):
                meta = {}

            # Never touch active-session rolling summaries here.
            if meta.get("summary_type") == "active_session":
                continue

            existing_ids = meta.get("message_ids", [])
            if isinstance(existing_ids, list) and existing_ids:
                # Exact match or any overlap with current source messages.
                if existing_ids == session.message_ids or bool(chunk_ids.intersection(existing_ids)):
                    replace_ids.append(row.id)
                    continue

            # Legacy/metadata-light fallback: overlapping session time window.
            try:
                existing_start = float(meta.get("session_start"))
                existing_end = float(meta.get("session_end"))
                if not (existing_end < session.start_timestamp or existing_start > session.end_timestamp):
                    replace_ids.append(row.id)
                    continue
            except (TypeError, ValueError):
                pass

            # Last-resort legacy fallback: non-active summary row in the timestamp window.
            replace_ids.append(row.id)

        return list(dict.fromkeys(replace_ids))

    def detect_all_sessions(self) -> list[Session]:
        """Detect all conversation sessions."""
        messages = self._get_all_messages()
        return detect_sessions(messages, self.session_gap_seconds)

    def preview_summarizable_sessions(self) -> list[Session]:
        """Preview sessions that would be summarized (dry run).

        Uses budget-driven eligibility: only sessions whose messages would
        overflow the token budget are candidates for summarization.
        """
        messages = self._get_all_messages()
        # Only conversation turns are summarizable.  Internal/system/tool rows
        # are not shown as messages in the UI and some are omitted from the
        # summarization prompt entirely; counting them here can make an
        # apparently empty bot eligible and trigger a useless LLM request.
        regular_messages = [
            m
            for m in messages
            if m.get("role") in ("user", "assistant")
            and not bool(m.get("recalled_history"))
            and not bool(m.get("summarized"))
        ]
        eligible = find_budget_overflow_sessions(
            regular_messages,
            session_gap_seconds=self.session_gap_seconds,
            max_context_tokens=self.max_context_tokens,
            protected_recent_turns=self.protected_recent_turns,
            min_messages_per_session=self.min_messages_per_session,
        )
        return prioritize_summarizable_sessions(eligible)

    def summarize_session(
        self,
        session: Session,
        use_heuristic_fallback: bool = True,
        replace_existing: bool = False,
        summary_text_override: str | None = None,
        method_override: str | None = None,
    ) -> dict:
        """Summarize a single session and store the result.

        Args:
            session: The session to summarize
            use_heuristic_fallback: If True, use heuristic if LLM fails

        Returns:
            Dict with 'success', 'summary_id', 'summary_text', 'method'
        """
        # Try LLM first - use custom function if provided, otherwise HTTP
        if summary_text_override is not None:
            summary_text = summary_text_override
            method = method_override or "llm"
        else:
            if self._summarize_fn:
                # Use provided function (avoids HTTP self-call when run in API)
                summary_text = self._summarize_fn(session)
            else:
                # Use HTTP endpoint (for CLI usage)
                summary_text = summarize_session_with_llm(session, self.service_url, config=self.config)
            method = method_override or "llm"

        if not summary_text and use_heuristic_fallback:
            summary_text = summarize_session_heuristic(session)
            method = "heuristic"

        if not summary_text:
            return {"success": False, "error": "Failed to generate summary"}

        normalized_text = compress_structured_summary_text(
            normalize_structured_summary_text(summary_text),
            source_session=session,
        )
        if not normalized_text.startswith("[HISTORICAL SUMMARY]"):
            normalized_text = f"[HISTORICAL SUMMARY]\n{normalized_text.strip()}"
        sections = extract_summary_sections(normalized_text)

        # Store the summary
        summary_id = str(uuid.uuid4())
        summary_metadata = {
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
            "message_ids": session.message_ids,
            "message_count": session.message_count,
            "summarization_method": method,
            "created_at": time.time(),
            "summary_type": "historical",
            "summary_sections": sections,
            "intent": sections.get("intent", ""),
            "tone": sections.get("tone", ""),
            "open_loops": sections.get("open_loops", ""),
        }

        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            # TASK-284 step 13: summary provenance — record which durable DB
            # thread(s) the source rows belong to, and stamp the summary row's
            # own session_id when the source is a single thread. Raw reads
            # always filter role <> 'summary', so the stamp can never cause a
            # summary to double-load into a scoped raw transcript.
            source_session_ids = resolve_source_session_ids(
                conn, self._backend._messages_table_name, session.message_ids
            )
            summary_metadata["source_session_ids"] = source_session_ids

            existing_summary_ids = self._find_existing_summary_ids(conn, session)
            if existing_summary_ids and not replace_existing:
                return {
                    "success": True,
                    "summary_id": existing_summary_ids[0],
                    "summary_text": None,
                    "method": "existing",
                    "message_count": session.message_count,
                    "created": False,
                    "replaced_existing": 0,
                }
            replaced_existing = 0
            if replace_existing:
                replace_ids = self._find_replaceable_summary_ids(conn, session)
                if replace_ids:
                    delete_sql = text(
                        f"""
                        DELETE FROM {self._backend._messages_table_name}
                        WHERE id = ANY(:ids)
                        """
                    )
                    conn.execute(delete_sql, {"ids": replace_ids})
                    replaced_existing = len(replace_ids)

            # Mark source messages as summarized, but keep them in place.
            if session.message_ids:
                update_sql = text(
                    f"""
                    UPDATE {self._backend._messages_table_name}
                    SET summarized = TRUE
                    WHERE id = ANY(:ids)
                    """
                )
                conn.execute(update_sql, {"ids": session.message_ids})

            # Insert summary as a message with role='summary'
            # Timestamp is session end time for natural ordering
            insert_sql = text(f"""
                INSERT INTO {self._backend._messages_table_name}
                (id, role, content, timestamp, summary_metadata, session_id, created_at)
                VALUES (:id, 'summary', :content, :timestamp, :metadata, :session_id, CURRENT_TIMESTAMP)
            """)
            conn.execute(insert_sql, {
                "id": summary_id,
                "content": normalized_text,
                "timestamp": session.end_timestamp,
                "metadata": json.dumps(summary_metadata),
                # TASK-641: the Session's own thread id is authoritative when
                # the window was cut along thread boundaries (the live path);
                # otherwise fall back to the single-distinct-source stamp.
                # Multi/none -> NULL (provenance detail stays in
                # summary_metadata.source_session_ids).
                "session_id": session.session_id or (
                    source_session_ids[0] if len(source_session_ids) == 1 else None
                ),
            })

            conn.commit()

        logger.info(f"Created summary {summary_id[:8]} for session with {session.message_count} messages ({method})")

        return {
            "success": True,
            "summary_id": summary_id,
            "summary_text": normalized_text,
            "method": method,
            "message_count": session.message_count,
            "created": True,
            "replaced_existing": replaced_existing,
            "session_start": session.start_timestamp,
            "session_end": session.end_timestamp,
        }

    def summarize_thread(
        self,
        session_id: str,
        message_ids: list[str] | None = None,
        use_heuristic_fallback: bool = True,
        protect_recent_turns: bool = False,
        min_messages: int | None = None,
        max_tokens_per_chunk: int = 4000,
    ) -> dict:
        """THE common per-thread summarization unit (TASK-641).

        Shared by BOTH callers so they cannot drift:
        - the background job passes the ``message_ids`` its budget-overflow
          pass selected for this thread (protection already applied there);
        - the ``/new`` path passes no ids — the whole thread's eligible
          unsummarized rows are summarized, minus the protected recent tail
          when ``protect_recent_turns`` is set (keeps the seed's
          high-fidelity raw tail instead of degrading it to summaries-only).
          ``min_messages`` may lower the job threshold for a caller that has no
          raw-history fallback, such as a summary-only continuity seed.

        Chunks large threads exactly like the job path and stores each chunk
        via ``summarize_session`` — so every summary row created here carries
        this thread's ``session_id`` structurally.

        Returns an aggregate dict: ``{"success", "summaries_created",
        "messages_summarized", "results", "errors"}``. Never raises.
        """
        from sqlalchemy import text as _sql_text

        try:
            with self._backend.engine.connect() as conn:
                rows = conn.execute(
                    _sql_text(f"""
                        SELECT id, role, content, timestamp, session_id,
                               author_entity_type, author_entity_id
                        FROM {self._backend._messages_table_name}
                        WHERE session_id = :sid
                          AND role IN ('user', 'assistant')
                          AND COALESCE(summarized, FALSE) = FALSE
                          AND COALESCE(recalled_history, FALSE) = FALSE
                        ORDER BY timestamp ASC, id ASC
                    """),
                    {"sid": session_id},
                ).fetchall()
        except Exception as e:
            logger.error(f"summarize_thread: failed to load thread {session_id}: {e}")
            return {"success": False, "summaries_created": 0,
                    "messages_summarized": 0, "results": [], "errors": [str(e)]}

        msgs = [
            {"id": r.id, "role": r.role, "content": r.content,
             "timestamp": r.timestamp, "session_id": r.session_id,
             "author_entity_type": r.author_entity_type,
             "author_entity_id": r.author_entity_id}
            for r in rows
        ]
        if message_ids is not None:
            wanted = set(message_ids)
            msgs = [m for m in msgs if m["id"] in wanted]
        elif protect_recent_turns:
            n_protected = self.protected_recent_turns * 2
            if n_protected > 0:
                msgs = msgs[:-n_protected] if len(msgs) > n_protected else []

        minimum = (
            self.min_messages_per_session
            if min_messages is None
            else max(1, int(min_messages))
        )
        if len(msgs) < minimum:
            return {"success": True, "summaries_created": 0,
                    "messages_summarized": 0, "results": [], "errors": []}

        session = Session(
            start_timestamp=msgs[0]["timestamp"],
            end_timestamp=msgs[-1]["timestamp"],
            messages=msgs,
            message_ids=[m["id"] for m in msgs],
            session_id=str(session_id),
        )

        results: list[dict] = []
        errors: list[str] = []
        created = 0
        total_messages = 0
        for i, chunk in enumerate(split_session_into_chunks(session, max_tokens_per_chunk)):
            try:
                result = self.summarize_session(chunk, use_heuristic_fallback)
            except Exception as e:
                errors.append(f"thread {session_id} chunk {i + 1}: {e}")
                continue
            if result.get("success"):
                results.append(result)
                if result.get("created", True):
                    created += 1
                    total_messages += chunk.message_count
            else:
                errors.append(
                    f"thread {session_id} chunk {i + 1}: "
                    f"{result.get('error', 'Unknown error')}"
                )

        return {
            "success": not errors or created > 0,
            "summaries_created": created,
            "messages_summarized": total_messages,
            "results": results,
            "errors": errors,
        }

    def _summarize_sessions_batch(
        self,
        sessions: list[Session],
        use_heuristic_fallback: bool,
        replace_existing: bool,
    ) -> tuple[list[dict], list[str], int, int, int]:
        """Summarize multiple sessions in one LLM call when batch callable is available.

        Returns:
            (results, errors, created_count, total_messages, replaced_count)
        """
        if not sessions or not self._summarize_batch_fn:
            return ([], [], 0, 0, 0)

        batch_size = int(self._setting("summarization_batch_size", 5) or 5)
        batch_size = max(1, batch_size)

        # Build micro-batch chunks.
        micro_batches: list[tuple[int, list[Session]]] = []
        for start in range(0, len(sessions), batch_size):
            micro_batches.append((start, sessions[start : start + batch_size]))

        # Run micro-batches concurrently — each is an independent API call.
        concurrency = int(self._setting("summarization_batch_concurrency", 3) or 3)
        concurrency = max(1, min(concurrency, len(micro_batches)))

        full_summary_map: dict[int, str] = {}

        def _run_micro_batch(item: tuple[int, list[Session]]) -> dict[int, str]:
            start, chunk = item
            partial = self._summarize_batch_fn(chunk)
            result: dict[int, str] = {}
            if partial:
                for local_idx, summary_text in partial.items():
                    global_idx = start + int(local_idx)
                    if 1 <= global_idx <= len(sessions):
                        result[global_idx] = summary_text
            return result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_run_micro_batch, mb) for mb in micro_batches]
            for fut in as_completed(futures):
                try:
                    full_summary_map.update(fut.result())
                except Exception as exc:
                    logger.error("Micro-batch failed: %s", exc)

        if not full_summary_map:
            return ([], [], 0, 0, 0)

        results: list[dict] = []
        errors: list[str] = []
        created_count = 0
        total_messages = 0
        replaced_count = 0
        covered_indices: set[int] = set()

        for idx, session in enumerate(sessions, start=1):
            summary_text = full_summary_map.get(idx)
            method = "llm-batch"
            if not summary_text and use_heuristic_fallback:
                summary_text = summarize_session_heuristic(session)
                method = "heuristic"

            if not summary_text:
                errors.append(f"Session {idx}: missing summary in batch output")
                continue

            covered_indices.add(idx)

            result = self.summarize_session(
                session,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=replace_existing,
                summary_text_override=summary_text,
                method_override=method,
            )
            if result.get("success"):
                results.append(result)
                created_count += 1
                total_messages += session.message_count
                replaced_count += int(result.get("replaced_existing", 0) or 0)
            else:
                errors.append(f"Session {idx}: {result.get('error', 'Unknown error')}")

        # Fallback: sessions the batch missed get heuristic summaries instead of
        # expensive per-session LLM calls.  This avoids N extra API round-trips
        # for sessions the model chose to skip or that failed quality checks.
        missing_indices = [i for i in range(1, len(sessions) + 1) if i not in covered_indices]
        if missing_indices:
            logger.info(
                "Batch output missing %s/%s sessions; using heuristic for indices: %s",
                len(missing_indices),
                len(sessions),
                missing_indices,
            )
            # Remove placeholder missing-summary errors now that we'll handle them.
            errors = [e for e in errors if "missing summary in batch output" not in e]

            generated_fallback_texts: dict[int, str | None] = {}
            for index in missing_indices:
                session = sessions[index - 1]
                generated_fallback_texts[index] = summarize_session_heuristic(session)

            for idx in missing_indices:
                session = sessions[idx - 1]
                text = generated_fallback_texts.get(idx)
                if not text:
                    errors.append(f"Session {idx}: Missing in batch and fallback generation failed")
                    continue

                fallback_result = self.summarize_session(
                    session,
                    use_heuristic_fallback=use_heuristic_fallback,
                    replace_existing=replace_existing,
                    summary_text_override=text,
                    method_override="heuristic-fallback",
                )
                if fallback_result.get("success"):
                    results.append(fallback_result)
                    created_count += 1
                    total_messages += session.message_count
                    replaced_count += int(fallback_result.get("replaced_existing", 0) or 0)
                else:
                    errors.append(f"Session {idx}: {fallback_result.get('error', 'Fallback store failed')}")

        return (results, errors, created_count, total_messages, replaced_count)

    def _purge_historical_summaries(
        self,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> int:
        """Delete historical summaries in optional time range.

        Active-session summaries are never deleted by this helper.
        """
        from sqlalchemy import text

        with self._backend.engine.connect() as conn:
            sql = text(
                f"""
                SELECT id, summary_metadata, timestamp
                FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
                ORDER BY timestamp ASC
                """
            )
            rows = conn.execute(sql).fetchall()

            to_delete: list[str] = []
            for row in rows:
                ts = float(row.timestamp or 0)
                if start_timestamp is not None and ts < start_timestamp:
                    continue
                if end_timestamp is not None and ts > end_timestamp:
                    continue

                meta = row.summary_metadata or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if isinstance(meta, dict) and meta.get("summary_type") == "active_session":
                    continue

                to_delete.append(row.id)

            if not to_delete:
                return 0

            delete_sql = text(
                f"""
                DELETE FROM {self._backend._messages_table_name}
                WHERE id = ANY(:ids)
                """
            )
            conn.execute(delete_sql, {"ids": to_delete})
            conn.commit()
            return len(to_delete)

    def _cleanup_active_session_summaries(self) -> int:
        """Remove any legacy active-session summary rows from the database.

        These are no longer created by the budget-driven summarization system.
        """
        from sqlalchemy import text

        with self._backend.engine.connect() as conn:
            sql = text(
                f"""
                SELECT id, summary_metadata
                FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
                ORDER BY created_at DESC
                LIMIT 200
                """
            )
            rows = conn.execute(sql).fetchall()
            active_ids: list[str] = []
            for row in rows:
                meta = row.summary_metadata or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        continue
                if not isinstance(meta, dict):
                    continue
                if meta.get("summary_type") == "active_session":
                    active_ids.append(row.id)

            if active_ids:
                delete_sql = text(
                    f"""
                    DELETE FROM {self._backend._messages_table_name}
                    WHERE id = ANY(:ids)
                    """
                )
                conn.execute(delete_sql, {"ids": active_ids})
                conn.commit()
                logger.info("Cleaned up %d legacy active-session summary rows", len(active_ids))

            return len(active_ids)

    def summarize_eligible_sessions(
        self,
        use_heuristic_fallback: bool = True,
        max_tokens_per_chunk: int = 4000,
    ) -> dict:
        """Summarize sessions whose messages would overflow the token budget.

        Large sessions are automatically split into chunks that fit within
        the model's context window.

        Returns:
            Dict with 'sessions_summarized', 'messages_summarized', 'results', 'errors'
        """
        # Clean up any legacy active-session summary rows
        self._cleanup_active_session_summaries()

        eligible = self.preview_summarizable_sessions()

        if not eligible:
            return {
                "sessions_summarized": 0,
                "messages_summarized": 0,
                "results": [],
                "errors": [],
            }

        results = []
        errors = []
        total_messages = 0
        created_count = 0
        skipped_existing_count = 0

        if self._summarize_batch_fn and max_tokens_per_chunk <= 0 and len(eligible) > 1:
            batch_results, batch_errors, batch_created, batch_messages, _ = self._summarize_sessions_batch(
                sessions=eligible,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=False,
            )
            if batch_created > 0 or batch_errors:
                results = batch_results
                errors = batch_errors
                created_count = batch_created
                total_messages = batch_messages
            else:
                logger.warning("Batch summarization returned no output; falling back to per-session mode")
        if created_count == 0 and not results and not errors:
            for session in eligible:
                try:
                    if session.session_id:
                        # TASK-641: thread-keyed window — route through THE
                        # common per-thread unit (same function the /new path
                        # calls) so job and /new summaries cannot drift.
                        thread_result = self.summarize_thread(
                            session_id=session.session_id,
                            message_ids=session.message_ids,
                            use_heuristic_fallback=use_heuristic_fallback,
                            max_tokens_per_chunk=max_tokens_per_chunk,
                        )
                        results.extend(thread_result.get("results", []))
                        errors.extend(thread_result.get("errors", []))
                        created_count += thread_result.get("summaries_created", 0)
                        total_messages += thread_result.get("messages_summarized", 0)
                        skipped_existing_count += sum(
                            1 for r in thread_result.get("results", [])
                            if not r.get("created", True)
                        )
                        continue

                    # Legacy fallback: NULL-session rows (gap-detected window)
                    chunks = split_session_into_chunks(session, max_tokens_per_chunk)

                    for i, chunk in enumerate(chunks):
                        chunk_label = f" (chunk {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                        try:
                            result = self.summarize_session(chunk, use_heuristic_fallback)
                            if result.get("success"):
                                results.append(result)
                                if result.get("created", True):
                                    created_count += 1
                                    total_messages += chunk.message_count
                                    logger.info(f"Summarized{chunk_label}: {chunk.message_count} messages")
                                else:
                                    skipped_existing_count += 1
                                    logger.debug(f"Skipped already summarized session{chunk_label}")
                            else:
                                error_msg = result.get("error", "Unknown error")
                                errors.append(f"Session{chunk_label}: {error_msg}")
                                logger.error(f"Failed to summarize{chunk_label}: {error_msg}")
                        except Exception as e:
                            logger.error(f"Error summarizing session{chunk_label}: {e}")
                            errors.append(f"Session{chunk_label}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing session: {e}")
                    errors.append(str(e))

        return {
            "sessions_summarized": created_count,
            "messages_summarized": total_messages,
            "sessions_skipped_existing": skipped_existing_count,
            "results": results,
            "errors": errors,
        }

    def resummarize_heuristic_summaries(
        self,
        limit: int = 25,
        *,
        thread_id: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Regenerate summaries that were written by the heuristic fallback.

        TASK-857: for months every summary path silently fell back to
        :func:`summarize_session_heuristic` because the job-model resolver
        borrowed an agent bot's identity. Those rows are still on disk and are
        what ``/new`` seeds from, so fixing the resolver alone leaves the bad
        text in place. This walks the stored heuristic summaries newest-first,
        rebuilds each one's source :class:`Session` from the message ids in its
        own metadata, and re-runs the normal
        :meth:`summarize_session` path with ``replace_existing=True`` — same
        normalization, same provenance, no LLM fallback to heuristic (a failed
        row is left untouched rather than rewritten with the same slop).
        """
        from sqlalchemy import text

        table = self._backend._messages_table_name
        with self._backend.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT id, summary_metadata
                    FROM {table}
                    WHERE role = 'summary'
                      AND summary_metadata->>'summarization_method' = 'heuristic'
                      AND (:thread_id IS NULL OR session_id = :thread_id)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                    """
                ),
                {"limit": max(int(limit), 0), "thread_id": thread_id},
            ).fetchall()

            targets: list[Session] = []
            skipped: list[str] = []
            for row in rows:
                meta = row.summary_metadata or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                message_ids = meta.get("message_ids") or []
                if not isinstance(message_ids, list) or not message_ids:
                    skipped.append(f"{row.id[:8]}: no source message_ids")
                    continue
                src = conn.execute(
                    text(
                        f"""
                        SELECT id, role, content, timestamp, session_id,
                               author_entity_type, author_entity_id
                        FROM {table}
                        WHERE id = ANY(:ids) AND role <> 'summary'
                        ORDER BY timestamp ASC
                        """
                    ),
                    {"ids": [str(m) for m in message_ids]},
                ).fetchall()
                if len(src) != len(message_ids):
                    skipped.append(
                        f"{row.id[:8]}: {len(src)}/{len(message_ids)} source rows still present"
                    )
                    continue
                thread_ids = {r.session_id for r in src if r.session_id}
                targets.append(
                    Session(
                        start_timestamp=float(src[0].timestamp),
                        end_timestamp=float(src[-1].timestamp),
                        messages=[
                            {"role": r.role, "content": r.content, "timestamp": r.timestamp,
                             "author_entity_type": r.author_entity_type,
                             "author_entity_id": r.author_entity_id}
                            for r in src
                        ],
                        message_ids=[r.id for r in src],
                        session_id=(thread_ids.pop() if len(thread_ids) == 1 else None),
                    )
                )

        if dry_run:
            return {
                "targeted": len(targets),
                "regenerated": 0,
                "skipped": skipped,
                "errors": [],
            }

        regenerated = 0
        errors: list[str] = []
        for session in targets:
            try:
                result = self.summarize_session(
                    session,
                    use_heuristic_fallback=False,
                    replace_existing=True,
                )
                if result.get("success") and result.get("created", True):
                    regenerated += 1
                else:
                    errors.append(
                        f"{(session.session_id or '?')[:8]}: "
                        f"{result.get('error') or result.get('method')}"
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{(session.session_id or '?')[:8]}: {e}")

        logger.info(
            "Re-summarized %s/%s heuristic summaries for %s",
            regenerated, len(targets), self.bot_id,
        )
        return {
            "targeted": len(targets),
            "regenerated": regenerated,
            "skipped": skipped,
            "errors": errors,
        }

    def rebuild_recent_sessions(
        self,
        session_limit: int = 5,
        use_heuristic_fallback: bool = True,
        max_tokens_per_chunk: int = 4000,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        purge_existing: bool = False,
    ) -> dict:
        """Recalculate summaries for the most recent eligible sessions.

        Unlike normal summarization, this uses raw history regardless of existing
        summarized flags and replaces existing summaries for the targeted sessions.
        Uses budget-driven eligibility.
        """
        messages = self._get_all_messages()
        regular_messages = [
            m
            for m in messages
            if m.get("role") != "summary"
            and not bool(m.get("recalled_history"))
        ]
        eligible = find_budget_overflow_sessions(
            regular_messages,
            session_gap_seconds=self.session_gap_seconds,
            max_context_tokens=self.max_context_tokens,
            protected_recent_turns=self.protected_recent_turns,
            min_messages_per_session=self.min_messages_per_session,
        )
        eligible = prioritize_summarizable_sessions(eligible)

        if start_timestamp is not None or end_timestamp is not None:
            eligible = [
                s
                for s in eligible
                if (start_timestamp is None or s.end_timestamp >= start_timestamp)
                and (end_timestamp is None or s.start_timestamp <= end_timestamp)
            ]

        if session_limit > 0:
            eligible = eligible[-session_limit:]

        purged_count = 0
        if purge_existing:
            purged_count = self._purge_historical_summaries(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

        if not eligible:
            return {
                "sessions_targeted": 0,
                "sessions_summarized": 0,
                "messages_summarized": 0,
                "summaries_replaced": 0,
                "results": [],
                "errors": [],
                "summaries_purged": purged_count,
            }

        results = []
        errors = []
        total_messages = 0
        created_count = 0
        replaced_count = 0

        if self._summarize_batch_fn and max_tokens_per_chunk <= 0 and len(eligible) > 1:
            batch_results, batch_errors, batch_created, batch_messages, batch_replaced = self._summarize_sessions_batch(
                sessions=eligible,
                use_heuristic_fallback=use_heuristic_fallback,
                replace_existing=True,
            )
            if batch_created > 0 or batch_errors:
                results = batch_results
                errors = batch_errors
                created_count = batch_created
                total_messages = batch_messages
                replaced_count = batch_replaced
            else:
                logger.warning("Batch rebuild returned no output; falling back to per-session mode")

        if created_count == 0 and not results and not errors:
            for session in eligible:
                try:
                    chunks = split_session_into_chunks(session, max_tokens_per_chunk)
                    for i, chunk in enumerate(chunks):
                        chunk_label = f" (chunk {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                        result = self.summarize_session(
                            chunk,
                            use_heuristic_fallback=use_heuristic_fallback,
                            replace_existing=True,
                        )
                        if result.get("success"):
                            results.append(result)
                            created_count += 1
                            total_messages += chunk.message_count
                            replaced_count += int(result.get("replaced_existing", 0) or 0)
                            logger.info(
                                "Rebuilt summary%s: %s messages (%s replaced)",
                                chunk_label,
                                chunk.message_count,
                                result.get("replaced_existing", 0),
                            )
                        else:
                            error_msg = result.get("error", "Unknown error")
                            errors.append(f"Session{chunk_label}: {error_msg}")
                            logger.error(f"Failed to rebuild{chunk_label}: {error_msg}")
                except Exception as e:
                    logger.error(f"Error rebuilding session: {e}")
                    errors.append(str(e))

        return {
            "sessions_targeted": len(eligible),
            "sessions_summarized": created_count,
            "messages_summarized": total_messages,
            "summaries_replaced": replaced_count,
            "results": results,
            "errors": errors,
            "summaries_purged": purged_count,
        }

    def list_summaries(self) -> list[dict]:
        """List all existing summaries."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT id, content, timestamp, summary_metadata
                FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
                ORDER BY timestamp DESC
            """)
            rows = conn.execute(sql).fetchall()

            summaries = []
            for row in rows:
                metadata = row.summary_metadata or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                summaries.append({
                    "id": row.id,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_start": metadata.get("session_start"),
                    "session_end": metadata.get("session_end"),
                    "message_count": metadata.get("message_count", 0),
                    "method": metadata.get("summarization_method", "unknown"),
                })

            return summaries

    def delete_summary(self, summary_id: str) -> dict:
        """Delete a summary and unmark the original messages.

        Args:
            summary_id: Full or prefix of summary UUID

        Returns:
            Dict with 'success', 'messages_restored'
        """
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text

            # Find the summary (support prefix matching)
            if len(summary_id) < 36:
                find_sql = text(f"""
                    SELECT id, summary_metadata
                    FROM {self._backend._messages_table_name}
                    WHERE role = 'summary' AND id LIKE :pattern
                    LIMIT 1
                """)
                row = conn.execute(find_sql, {"pattern": f"{summary_id}%"}).fetchone()
            else:
                find_sql = text(f"""
                    SELECT id, summary_metadata
                    FROM {self._backend._messages_table_name}
                    WHERE role = 'summary' AND id = :id
                """)
                row = conn.execute(find_sql, {"id": summary_id}).fetchone()

            if not row:
                return {"success": False, "error": f"Summary '{summary_id}' not found"}

            full_id = row.id
            metadata = row.summary_metadata or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            message_ids = metadata.get("message_ids", [])

            # Unmark the original messages
            rows_unmarked = 0
            if message_ids:
                update_sql = text(f"""
                    UPDATE {self._backend._messages_table_name}
                    SET summarized = FALSE
                    WHERE id = ANY(:ids)
                """)
                update_result = conn.execute(update_sql, {"ids": message_ids})
                rows_unmarked = int(update_result.rowcount or 0)

            # Delete the summary
            delete_sql = text(f"""
                DELETE FROM {self._backend._messages_table_name}
                WHERE id = :id
            """)
            conn.execute(delete_sql, {"id": full_id})

            conn.commit()

        logger.info(f"Deleted summary {full_id[:8]}, unmarked {rows_unmarked} source messages")

        return {
            "success": True,
            "summary_id": full_id,
            "messages_restored": rows_unmarked,
        }

    def get_summary_count(self) -> int:
        """Get the count of existing summaries."""
        with self._backend.engine.connect() as conn:
            from sqlalchemy import text
            sql = text(f"""
                SELECT COUNT(*) FROM {self._backend._messages_table_name}
                WHERE role = 'summary'
            """)
            return conn.execute(sql).scalar() or 0
