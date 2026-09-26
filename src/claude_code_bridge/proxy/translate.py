"""Anthropic Messages body ↔ OpenAI Responses API body translation.

Anthropic shape we accept (subset of the public Messages API):

    {
      "model": "<provider>/<upstream_model>",
      "system": "...string..." | [{"type":"text","text":"..."}, ...],
      "messages": [
        {"role": "user", "content": "string" | [{"type":"text","text":...}, ...]},
        {"role": "assistant", "content": [
            {"type":"text","text":"..."},
            {"type":"tool_use","id":"tu_X","name":"...","input":{...}},
        ]},
        {"role": "user", "content": [
            {"type":"tool_result","tool_use_id":"tu_X","content":"..."}
        ]},
      ],
      "tools": [{"name":"...","description":"...","input_schema":{...}}],
      "max_tokens": 4096,
      "temperature": ...,
      "stream": true
    }

Responses API shape we emit:

    {
      "model": "<upstream_model>",
      "instructions": "...",   # merged system content
      "input": [
        {"role":"user","content":"..."} | with multimodal parts,
        {"role":"assistant","content":"..."},
        {"type":"function_call","name":"...","arguments":"<json>","call_id":"tu_X"},
        {"type":"function_call_output","call_id":"tu_X","output":"..."},
      ],
      "tools": [{"type":"function","name":"...","description":"...","parameters":{...}}],
      "max_output_tokens": 4096,
      "temperature": ...,
      "store": false,
      "stream": true
    }

Conversion follows ``llm_bawt/clients/responses_client.py`` for the
Responses-API side so the codebase stays consistent on that shape.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .tool_discovery import model_tool_description
from .reasoning import ReasoningCodec

logger = logging.getLogger(__name__)

# Anthropic server-side / built-in tool types carry a versioned suffix
# (e.g. "web_search_20260209", "bash_20250124"). They only execute on
# api.anthropic.com, so they must not be forwarded to a proxied provider.
_SERVER_TOOL_TYPE_RE = re.compile(
    r"^(web_search|web_fetch|bash|text_editor|code_execution|computer)_\d{6,}$"
)


def _effort_from_budget(budget: Any) -> str | None:
    """Map an Anthropic thinking ``budget_tokens`` to a Responses reasoning
    effort level. Coarse by design — the two scales don't line up exactly."""
    try:
        n = int(budget)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n <= 4096:
        return "low"
    if n <= 16384:
        return "medium"
    return "high"


def _flatten_system(system: Any) -> str | None:
    """Anthropic ``system`` can be a string or list of content blocks."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text") or ""
                if text:
                    parts.append(text)
        return "\n\n".join(parts) if parts else None
    return None


def inline_system_text(content: Any) -> str | None:
    """Render a mid-conversation ``role: "system"`` message as user text.

    Against a custom base URL the Claude Code CLI sends harness context as
    positional ``system`` messages: the environment/skills block, and mid-turn
    messages absorbed at a tool boundary ("The user sent a new message while
    you were working: …"). Upstreams have no universally accepted positional
    system role, and hoisting into ``instructions`` would break its byte
    stability (prompt cache), so they ride in place as a user message wrapped
    the way the CLI presents the same content to Claude. Dropping them made
    OpenAI-routed bots blind to that context and to ``next`` steers (TASK-937).
    """
    text = _flatten_system(content)
    if not text:
        return None
    return f"<system-reminder>\n{text}\n</system-reminder>"


def _split_leading_temporal_context(system_text: str | None) -> tuple[str | None, str | None]:
    """Move the volatile leading datetime line out of ``instructions``.

    The app injects a first-line ``Current date/time: ...`` system section with
    minute-level resolution, which busts the Responses API prompt prefix on
    every turn. Mirror codex CLI: keep ``instructions`` byte-stable and carry
    the volatile temporal context in the first user input item instead.
    """
    if not system_text:
        return None, None
    if not system_text.startswith("Current date/time:"):
        return system_text, None
    first_line, sep, remainder = system_text.partition("\n")
    stable = remainder.lstrip("\n") if sep else ""
    # Coarsen the relocated timestamp to DATE resolution. At minute resolution
    # it changes every minute, re-busting the input prefix AND the
    # content-derived prompt_cache_key at each minute boundary — capping cache
    # hits to ~1-minute windows on long turns. Date resolution keeps the first
    # input item byte-stable for the whole turn/day so history caches
    # continuously. Exact time stays available to bots via the `time` tool.
    temporal = re.sub(r"\s+\d{1,2}:\d{2}\s*(?:AM|PM)\b.*$", "", first_line).rstrip()
    return stable or None, temporal or None


def image_block_to_url(block: dict) -> str | None:
    """Anthropic image block → a URL string usable by any OpenAI-family API.

    Handles both base64 (``{source:{type:base64, media_type, data}}``, rendered
    as a ``data:`` URI) and URL (``{source:{type:url, url}}``) sources. Shared
    by the Responses translator below AND by ``translate_cc`` (Chat Completions
    packs the same URL as ``{"type":"image_url","image_url":{"url":...}}``), so
    the source-parsing rules can't drift between the two surfaces.
    """
    src = block.get("source") or {}
    stype = src.get("type")
    if stype == "base64":
        data = src.get("data") or ""
        if not data:
            return None
        media = src.get("media_type") or "image/png"
        return f"data:{media};base64,{data}"
    if stype == "url":
        return (src.get("url") or "") or None
    return None


def _image_block_to_input_image(block: dict) -> dict | None:
    """Anthropic image block → Responses ``input_image`` part (or ``None``)."""
    url = image_block_to_url(block)
    if not url:
        return None
    return {"type": "input_image", "image_url": url, "detail": "auto"}


def _user_content_to_responses(content: Any) -> tuple[list[dict], list[dict], list[dict]]:
    """Split a user-role content payload into (parts, tool_result items, followup images).

    Anthropic packs tool_result blocks inside a user message; Responses API
    breaks them out as separate ``function_call_output`` input items. So a
    single Anthropic user message can produce 0–1 user content items + N
    function_call_output items.

    A ``function_call_output.output`` MUST be a plain string, so an image
    returned *inside* a tool_result (e.g. the ``generate_image`` tool) cannot
    ride in it. Such images are collected into the third return value so the
    caller can re-surface them as a trailing user ``input_image`` message — the
    only way a Responses-API model actually SEES a tool-generated image.
    """
    parts: list[dict] = []
    tool_results: list[dict] = []
    followup_images: list[dict] = []

    if isinstance(content, str):
        if content:
            parts.append({"type": "input_text", "text": content})
        return parts, tool_results, followup_images

    if not isinstance(content, list):
        return parts, tool_results, followup_images

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text") or ""
            if txt:
                parts.append({"type": "input_text", "text": txt})
        elif btype == "image":
            img = _image_block_to_input_image(block)
            if img:
                parts.append(img)
        elif btype == "tool_result":
            call_id = block.get("tool_use_id") or ""
            # tool_result.content can be a string OR a list of content blocks
            # (text/image). A Responses ``function_call_output.output`` must be
            # a STRING, so we flatten the text here and pull any image blocks
            # OUT into ``followup_images`` — the caller re-surfaces them as a
            # trailing user input_image message so the model can see them.
            raw = block.get("content")
            if isinstance(raw, str):
                output = raw
            elif isinstance(raw, list):
                chunks: list[str] = []
                block_imgs: list[dict] = []
                for sub in raw:
                    if not isinstance(sub, dict):
                        continue
                    stype = sub.get("type")
                    if stype == "text":
                        chunks.append(sub.get("text") or "")
                    elif stype == "image":
                        img = _image_block_to_input_image(sub)
                        if img:
                            block_imgs.append(img)
                output = "\n".join(chunks)
                # If only an image came back, give the output a non-empty note
                # so the model knows an image accompanies this tool result.
                if block_imgs and not output.strip():
                    output = "[image returned by tool — shown in the following message]"
                followup_images.extend(block_imgs)
            else:
                output = ""
            tool_results.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

    return parts, tool_results, followup_images


def _assistant_content_to_responses(
    content: Any, reasoning_codec: ReasoningCodec | None = None,
) -> list[dict]:
    """Assistant content → text item(s) + function_call items.

    Anthropic packs text and tool_use blocks together in one assistant
    message; Responses API wants them as separate input items.
    """
    items: list[dict] = []

    if isinstance(content, str):
        if content:
            items.append({"role": "assistant", "content": content})
        return items

    if not isinstance(content, list):
        return items

    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            joined = "".join(text_buf)
            if joined:
                items.append({"role": "assistant", "content": joined})
            text_buf.clear()

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_buf.append(block.get("text") or "")
        elif btype in ("thinking", "redacted_thinking"):
            # Replay only native state with compatible, explicit provenance.
            # store:false makes client-carried encrypted state necessary; it is
            # not a reason to discard it. Legacy/foreign blocks stay display-only.
            native = reasoning_codec.decode(block.get("signature")) if reasoning_codec else None
            if native is not None:
                flush_text()
                items.append(native)
        elif btype == "tool_use":
            flush_text()
            tool_input = block.get("input") or {}
            items.append({
                "type": "function_call",
                "name": block.get("name") or "",
                # sort_keys + compact separators so replayed history is
                # byte-stable regardless of SDK dict-key ordering.  Without
                # this, a key-order shuffle between turns busts the upstream
                # prompt cache at this input item.
                "arguments": (
                    json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
                    if not isinstance(tool_input, str)
                    else tool_input
                ),
                "call_id": block.get("id") or "",
            })
    flush_text()
    return items


def _tools_to_responses(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tools schema → Responses API tools schema.

    Anthropic:  {name, description, input_schema}
    Responses:  {type:"function", name, description, parameters}

    The converted list is sorted by tool name so the cache prefix stays
    byte-stable even if the Claude SDK reorders tools between turns.
    """
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Drop Anthropic *server-side* / built-in tools. These carry a versioned
        # ``type`` (e.g. "web_search_20260209", "web_fetch_20250910",
        # "bash_20250124", "text_editor_...", "code_execution_...") and NO
        # input_schema — they only execute against api.anthropic.com. On the
        # proxy path they'd otherwise collapse into a bogus parameter-less
        # function that the OpenAI/grok upstream can't run, hanging the turn.
        # Client/custom tools have no such type (or type=="custom") and DO carry
        # input_schema, so they pass through. Belt to the bridge's
        # disallowed_tools suspenders.
        ttype = tool.get("type")
        if isinstance(ttype, str) and _SERVER_TOOL_TYPE_RE.match(ttype):
            logger.debug("Stripping Anthropic server-side tool type=%r from proxy request", ttype)
            continue
        item: dict[str, Any] = {
            "type": "function",
            "name": tool.get("name") or "",
        }
        if "description" in tool or tool.get("name") == "ToolSearch":
            item["description"] = model_tool_description(tool)
        # ``input_schema`` is the Anthropic name; some callers already use
        # ``parameters`` so honor both.
        params = tool.get("input_schema") or tool.get("parameters")
        if params is not None:
            item["parameters"] = params
        converted.append(item)
    if not converted:
        return None
    # Deterministic ordering: sort by tool name so an SDK-side reorder
    # doesn't bust the upstream prompt cache.
    converted.sort(key=lambda t: t.get("name") or "")
    return converted


def anthropic_to_responses(
    body: dict, upstream_model: str, *, reasoning_codec: ReasoningCodec | None = None,
) -> dict:
    """Translate an Anthropic Messages request body to a Responses API body.

    ``upstream_model`` is the post-prefix model name (e.g. ``gpt-5.4``),
    already split off from the Anthropic ``model`` field by the route.
    """
    instructions, temporal_prefix = _split_leading_temporal_context(
        _flatten_system(body.get("system"))
    )
    input_items: list[dict] = []
    temporal_prefix_attached = False

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")

        if role == "user":
            parts, tool_results, followup_images = _user_content_to_responses(content)
            if temporal_prefix and not temporal_prefix_attached:
                parts = [{"type": "input_text", "text": temporal_prefix}, *parts]
                temporal_prefix_attached = True
            # ORDER IS LOAD-BEARING (mirrors translate_cc): function_call_output
            # items must immediately follow the assistant function_call, then
            # tool images, then the user's own text. Anthropic puts text AFTER
            # tool_result blocks, so emitting text first placed it before the
            # tool output it arrived after. (The CLI's absorbed mid-turn steer
            # is NOT a sibling block — it is appended inside the tool_result
            # string; hoisting that safely is TASK-937.)
            input_items.extend(tool_results)
            # Images returned inside a tool_result can't ride in a
            # function_call_output (its output must be a string), so surface
            # them as a trailing user image message right after the tool
            # outputs. This is what lets the model actually SEE a tool-generated
            # image (e.g. generate_image) and iterate on it.
            if followup_images:
                input_items.append({
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Image(s) returned by the tool call above:"},
                        *followup_images,
                    ],
                })
            if parts:
                # All parts are input_text/input_image; user content items
                # take the list directly.
                input_items.append({"role": "user", "content": parts})
        elif role == "assistant":
            input_items.extend(_assistant_content_to_responses(content, reasoning_codec))
        elif role == "system":
            text = inline_system_text(content)
            if text:
                input_items.append(
                    {"role": "user", "content": [{"type": "input_text", "text": text}]}
                )
        else:
            logger.warning("Dropping unsupported message role=%r", role)

    payload: dict[str, Any] = {
        "model": upstream_model,
        "input": input_items,
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions

    if "max_tokens" in body:
        try:
            payload["max_output_tokens"] = int(body["max_tokens"])
        except (TypeError, ValueError):
            pass

    if "temperature" in body:
        try:
            payload["temperature"] = float(body["temperature"])
        except (TypeError, ValueError):
            pass

    # Anthropic extended-thinking → Responses reasoning. When the SDK enables
    # thinking (driven by the bot's `effort` setting), forward it as a
    # reasoning effort so the upstream actually reasons. budget_tokens is a
    # coarse proxy for effort level. Adapters may override/default this.
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        effort = _effort_from_budget(budget)
        if effort:
            # `summary: auto` makes the upstream stream human-readable reasoning
            # summary deltas (surfaced as Anthropic thinking blocks). Without it
            # the model still reasons but emits only an opaque encrypted item,
            # so the UI sees a signature with no visible thinking text.
            payload["reasoning"] = {"effort": effort, "summary": "auto"}

    if body.get("stream"):
        payload["stream"] = True

    if temporal_prefix and not temporal_prefix_attached:
        input_items.insert(
            0,
            {
                "role": "user",
                "content": [{"type": "input_text", "text": temporal_prefix}],
            },
        )

    tools = _tools_to_responses(body.get("tools"))
    if tools:
        payload["tools"] = tools
        tc = body.get("tool_choice")
        if tc:
            payload["tool_choice"] = tc

    return payload
