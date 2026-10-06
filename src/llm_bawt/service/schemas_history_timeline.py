"""Response schemas for the chat timeline index (TASK-980 / TASK-971)."""

from typing import Literal

from pydantic import BaseModel


class TimelineSegment(BaseModel):
    """One thread's messages within one day: a conversation blip on the rail."""

    session_id: str
    first_message_id: str  # jump target: the thread's first message that day
    first_ts: float
    last_ts: float
    message_count: int
    user_prompt_count: int
    continued: bool = False  # earliest message in the session precedes this segment


class TimelineDay(BaseModel):
    """One active calendar day in the viewer's timezone."""

    date: str  # YYYY-MM-DD, local to the requested tz
    first_message_id: str
    first_ts: float
    last_ts: float
    message_count: int
    user_prompt_count: int
    session_ids: list[str]  # same order as `segments`
    segments: list[TimelineSegment]  # chronological by first message


class TimelineSession(BaseModel):
    """One thread with messages inside the indexed scope."""

    id: str
    title: str | None
    # "title" = session_metadata.title; "first_prompt" = first non-slash user
    # prompt; "none" = neither exists (frontend picks its own placeholder).
    label_source: Literal["title", "first_prompt", "none"]
    first_message_id: str
    first_ts: float
    last_ts: float
    message_count: int
    status: str | None


class TimelineResponse(BaseModel):
    bot_id: str
    user_id: str
    session_id: str | None
    tz: str
    from_utc: float
    to_utc: float
    coverage: dict  # start/end, overview, bounded continuation and exclusions
    # Opaque revalidation token: changes whenever a visible message is added
    # or removed. Compare for equality only.
    version: str
    days: list[TimelineDay]
    sessions: list[TimelineSession]


class TimelinePrompt(BaseModel):
    id: str
    ts: float
    session_id: str | None
    snippet: str


class TimelinePromptsResponse(BaseModel):
    bot_id: str
    user_id: str
    date: str | None
    session_id: str | None
    tz: str
    prompts: list[TimelinePrompt]
    truncated: bool


class TimelineAnchorsRequest(BaseModel):
    bot_id: str
    user_id: str
    session_id: str | None = None
    message_ids: list[str]


class TimelineAnchor(BaseModel):
    id: str
    session_id: str
    ts: float


class TimelineAnchorsResponse(BaseModel):
    anchors: list[TimelineAnchor]
