"""Anthropic Messages body → OpenAI **Chat Completions** body translation.

The sibling :mod:`translate` module targets the OpenAI *Responses* API. This one
targets the older *Chat Completions* shape, needed by upstreams that expose
only that surface — currently Kimi For Coding
(``https://api.kimi.com/coding/v1``), which has no Responses endpoint and no
Anthropic-compatible endpoint (probed live 2026-07-27; see
``adapters/kimi_coding.py`` for the receipts).

Chat Completions shape we emit::

    {
      "model": "k3",
      "messages": [
        {"role":"system","content":"..."},
        {"role":"user","content":"..." | [{"type":"text",...},
                                          {"type":"image_url","image_url":{"url":...}}]},
        {"role":"assistant","content":"...",
         "tool_calls":[{"id":"tu_X","type":"function",
                        "function":{"name":"...","arguments":"<json>"}}]},
        {"role":"tool","tool_call_id":"tu_X","content":"..."},
      ],
      "tools": [{"type":"function",
                 "function":{"name","description","parameters"}}],
      "max_tokens": 4096,
      "reasoning_effort": "low"|"high"|"max",
      "stream": true,
      "stream_options": {"include_usage": true}
    }

Key structural difference from the Responses translator: Chat Completions keeps
assistant text and tool calls in ONE message (``content`` + ``tool_calls``),
whereas Responses splits them into separate input items. Tool *results* become
``role:"tool"`` messages that must immediately follow the assistant message
carrying the matching ``tool_calls`` — so ordering here is load-bearing, not
cosmetic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .translate import (
    _SERVER_TOOL_TYPE_RE,
    _flatten_system,
    image_block_to_url,
    inline_system_text,
)
from .tool_discovery import model_tool_description

logger = logging.getLogger(__name__)

# Kimi advertises these via GET /v1/models -> think_efforts.valid_efforts.
# NOTE there is deliberately no "medium": the API tolerates it silently, but
# mapping onto an advertised value is the correct behavior.
VALID_EFFORTS = ("low", "high", "max")


def effort_from_budget(budget: Any) -> str | None:
    """Anthropic thinking ``budget_tokens`` → a Kimi ``reasoning_effort``.

    Coarse by design; the scales don't line up. Distinct from
    ``translate._effort_from_budget`` because that one emits ``medium`` (a valid
    Responses effort) which Kimi does not advertise.
    """
    try:
        n = int(budget)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n <= 4096:
        return "low"
    if n <= 32768:
        return "high"
    return "max"


def _tool_result_content_to_str(raw: Any) -> tuple[str, list[str]]:
    """Flatten an Anthropic ``tool_result.content`` into (text, image_urls).

    A Chat Completions ``role:"tool"`` message carries a plain string, so images
    returned *inside* a tool result cannot ride along. They're handed back for
    the caller to re-surface as a following user message — the only way the
    model actually SEES a tool-generated image (mirrors the Responses path).
    """
    if isinstance(raw, str):
        return raw, []
    if not isinstance(raw, list):
        return "", []
    chunks: list[str] = []
    images: list[str] = []
    for sub in raw:
        if not isinstance(sub, dict):
            continue
        stype = sub.get("type")
        if stype == "text":
            chunks.append(sub.get("text") or "")
        elif stype == "image":
            url = image_block_to_url(sub)
            if url:
                images.append(url)
    text = "\n".join(chunks)
    if images and not text.strip():
        text = "[image returned by tool — shown in the following message]"
    return text, images


def _user_content_to_cc(content: Any) -> tuple[Any, list[dict], list[str]]:
    """Split user content into (content_payload, tool_messages, followup_images).

    ``content_payload`` is a plain string when there's only text (keeps the
    request byte-stable and cache-friendly) or a list of parts when images are
    present. ``None`` when the message carried nothing but tool results.
    """
    if isinstance(content, str):
        return (content or None), [], []
    if not isinstance(content, list):
        return None, [], []

    texts: list[str] = []
    images: list[str] = []
    tool_messages: list[dict] = []
    followup_images: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text") or ""
            if txt:
                texts.append(txt)
        elif btype == "image":
            url = image_block_to_url(block)
            if url:
                images.append(url)
        elif btype == "tool_result":
            text, imgs = _tool_result_content_to_str(block.get("content"))
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": text,
            })
            followup_images.extend(imgs)

    if images:
        parts: list[dict] = [{"type": "text", "text": t} for t in texts if t]
        parts.extend(
            {"type": "image_url", "image_url": {"url": u}} for u in images
        )
        return parts, tool_messages, followup_images

    joined = "\n".join(t for t in texts if t)
    return (joined or None), tool_messages, followup_images


def _assistant_content_to_cc(content: Any) -> dict | None:
    """Anthropic assistant content → ONE Chat Completions assistant message."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content} if content else None
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text") or "")
        elif btype in ("thinking", "redacted_thinking"):
            # Reasoning we synthesised on the way OUT (stream_cc turns Kimi's
            # reasoning_content into Anthropic thinking blocks). DROP it on the
            # return trip: the upstream is stateless, reasoning is regenerated
            # each turn, and the signature we mint is a local sentinel — not
            # something Kimi would accept back as input.
            continue
        elif btype == "tool_use":
            tool_input = block.get("input")
            if isinstance(tool_input, str):
                arguments = tool_input
            else:
                # sort_keys + compact separators so replayed history is
                # byte-stable regardless of SDK dict ordering; a key shuffle
                # between turns would otherwise bust the upstream prompt cache.
                arguments = json.dumps(
                    tool_input or {}, sort_keys=True, separators=(",", ":")
                )
            tool_calls.append({
                "id": block.get("id") or "",
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": arguments,
                },
            })

    text = "".join(texts)
    if not text and not tool_calls:
        return None
    # Chat Completions expects ``content`` present even alongside tool_calls;
    # a live round-trip against Kimi confirmed an empty string is accepted.
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tools_to_cc(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tools → Chat Completions tools (nested under ``function``).

    Sorted by name so an SDK-side reorder doesn't bust the prompt cache, and
    Anthropic *server-side* tools are dropped: they carry a versioned ``type``
    (``web_search_20260209``, ``bash_20250124``, …), no ``input_schema``, and
    only execute on api.anthropic.com — forwarding one would collapse into a
    bogus parameter-less function and hang the turn.
    """
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if isinstance(ttype, str) and _SERVER_TOOL_TYPE_RE.match(ttype):
            logger.debug(
                "Stripping Anthropic server-side tool type=%r from CC request", ttype
            )
            continue
        fn: dict[str, Any] = {"name": tool.get("name") or ""}
        if "description" in tool or tool.get("name") == "ToolSearch":
            fn["description"] = model_tool_description(tool)
        params = tool.get("input_schema") or tool.get("parameters")
        if params is not None:
            fn["parameters"] = params
        converted.append({"type": "function", "function": fn})
    if not converted:
        return None
    converted.sort(key=lambda t: (t.get("function") or {}).get("name") or "")
    return converted


def _tool_choice_to_cc(tc: Any) -> Any | None:
    """Anthropic ``tool_choice`` → Chat Completions ``tool_choice``."""
    if not isinstance(tc, dict):
        return tc or None
    ttype = tc.get("type")
    if ttype == "auto":
        return "auto"
    if ttype == "any":
        return "required"
    if ttype == "none":
        return "none"
    if ttype == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def anthropic_to_chat_completions(body: dict, upstream_model: str) -> dict:
    """Translate an Anthropic Messages request body to a Chat Completions body.

    ``upstream_model`` is the post-prefix model name (e.g. ``k3``), already
    split off the namespaced Anthropic ``model`` field by the route.
    """
    messages: list[dict] = []

    system = _flatten_system(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")

        if role == "user":
            payload, tool_messages, followup_images = _user_content_to_cc(content)
            # ORDER IS LOAD-BEARING: role:"tool" messages must immediately
            # follow the assistant message that requested them, before any new
            # user turn.
            messages.extend(tool_messages)
            if followup_images:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Image(s) returned by the tool call above:"},
                        *({"type": "image_url", "image_url": {"url": u}}
                          for u in followup_images),
                    ],
                })
            if payload is not None:
                messages.append({"role": "user", "content": payload})
        elif role == "assistant":
            converted = _assistant_content_to_cc(content)
            if converted is not None:
                messages.append(converted)
        elif role == "system":
            text = inline_system_text(content)
            if text:
                messages.append({"role": "user", "content": text})
        else:
            logger.warning("Dropping unsupported message role=%r", role)

    payload_out: dict[str, Any] = {"model": upstream_model, "messages": messages}

    if "max_tokens" in body:
        try:
            payload_out["max_tokens"] = int(body["max_tokens"])
        except (TypeError, ValueError):
            pass

    # ``temperature`` is deliberately NOT forwarded. Kimi For Coding rejects
    # anything but 1 with HTTP 400 ("invalid temperature: only 1 is allowed for
    # this model") — verified live — and the Claude SDK does send a temperature.
    # Omitting it lets the server apply its own default instead of 400-ing the
    # turn. (Note this differs from Moonshot's OpenPlatform, which instead
    # silently rescales temperature by 0.6.)

    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        effort = effort_from_budget(thinking.get("budget_tokens"))
        if effort:
            payload_out["reasoning_effort"] = effort

    if body.get("stream"):
        payload_out["stream"] = True
        # Without this the final chunk carries no usage block, so we'd report
        # zero tokens and lose all cache-hit visibility.
        payload_out["stream_options"] = {"include_usage": True}

    tools = _tools_to_cc(body.get("tools"))
    if tools:
        payload_out["tools"] = tools
        choice = _tool_choice_to_cc(body.get("tool_choice"))
        if choice is not None:
            # Live Kimi For Coding receipt (2026-07-27): a named/forced tool
            # choice with reasoning enabled returns HTTP 400
            # "tool_choice 'specified' is incompatible with thinking enabled".
            # Keep reasoning and relax the forcing to auto; the alternative is
            # a turn that cannot run at all.
            if "reasoning_effort" in payload_out and choice != "auto":
                choice = "auto"
            payload_out["tool_choice"] = choice

    return payload_out
