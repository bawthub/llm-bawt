from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json
from .reset_evidence import CodexResetEvidence
from ._bridge_helpers import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _ModelInfoCache,
    _bot_slug_from_session_key,
    _is_auth_failure,
    _is_codex_session_error,
)

logger = logging.getLogger("codex_bridge.bridge")


class CodexEventMixin:
    """Codex SDK event translation + tool/usage emission (TASK-555).

    Split out of ``CodexBridge`` (TASK-555). Composed back on via
    inheritance, so ``self.*`` state set in ``CodexBridge.__init__`` and
    methods on sibling mixins all resolve on the assembled instance.
    """

    async def _iter_events(self, streamed):
        """Async-iterate ``streamed.events`` with a per-event timeout.

        TimeoutError surfaces so the retry-on-session-error path can clear
        a stuck thread and retry fresh (TASK-206 acceptance criteria).
        """
        agen = streamed.events.__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(
                    agen.__anext__(),
                    timeout=self._request_timeout,
                )
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise TimeoutError(
                    f"No Codex events for {self._request_timeout}s — codex CLI may be hung"
                )
            yield event

    @staticmethod
    def _already_sent_done(seq: int, item_buffers: dict) -> bool:
        # turn.completed handler sets this sentinel.
        return bool(item_buffers.get("__done_emitted__"))

    @staticmethod
    def _build_prompt_input(
        message: str,
        attachments: list[dict],
        *,
        system_prompt: str | None = None,
    ) -> tuple[Any, list[str]]:
        """Compose the codex turn input from text + image attachments.

        Returns ``(prompt, tmp_paths)`` where ``tmp_paths`` are temp image
        files the caller must clean up after the turn completes. Codex SDK
        only accepts ``LocalImageInput`` (file path), not inline base64, so
        attachments are decoded into a tempdir.

        ``system_prompt``, when supplied, is prepended to ``message``. The
        codex CLI has no separate system-prompt slot, so this is the only
        way the bot's persona/MCP-context reaches the agent.
        """
        from openai_codex_sdk import LocalImageInput, TextInput

        text = message
        if system_prompt:
            text = f"{system_prompt}\n\n---\n\n{message}"

        if not attachments:
            return text, []

        tmp_paths: list[str] = []
        items: list = [TextInput(type="text", text=text)]
        for att in attachments:
            data = att.get("content")
            if not data:
                continue
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                logger.warning("Skipping attachment with invalid base64 payload")
                continue
            mime = (att.get("mimeType") or "image/png").lower()
            ext = ".png"
            if "jpeg" in mime or "jpg" in mime:
                ext = ".jpg"
            elif "gif" in mime:
                ext = ".gif"
            elif "webp" in mime:
                ext = ".webp"
            tmp = tempfile.NamedTemporaryFile(
                suffix=ext, prefix="codex_img_", delete=False
            )
            try:
                tmp.write(raw)
            finally:
                tmp.close()
            tmp_paths.append(tmp.name)
            items.append(LocalImageInput(type="local_image", path=tmp.name))

        return items, tmp_paths

    @staticmethod
    def _cleanup_tmp_files(paths: list[str]) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        paths.clear()

    @staticmethod
    def _extract_token_usage(usage_or_carrier: Any, *, actual_model_or_default: str) -> dict | None:
        """Normalize a Codex ``Usage`` (or carrier with a .usage attr) into
        the AgentEvent token_usage dict.
        """
        if usage_or_carrier is None:
            return None
        # Allow callers to pass either the Usage directly or a wrapper with
        # a .usage attribute (e.g. TurnCompletedEvent).
        usage = usage_or_carrier
        if hasattr(usage, "usage") and not hasattr(usage, "input_tokens"):
            usage = getattr(usage, "usage")
        if isinstance(usage_or_carrier, dict) and "usage" in usage_or_carrier:
            usage = usage_or_carrier["usage"]
        if usage is None:
            return None

        def _g(key: str, default=0):
            if hasattr(usage, key):
                return getattr(usage, key)
            if isinstance(usage, dict):
                return usage.get(key, default)
            return default
        try:
            # OpenAI/Codex ``input_tokens`` is INCLUSIVE of the cached subset
            # (``cached_input_tokens`` ⊆ ``input_tokens``). The frontend context
            # pill and cost estimator both SUM input + cache_read + cache_creation
            # (Anthropic's convention, where those buckets are disjoint), so
            # passing the raw inclusive value double-counts every cached token —
            # a fully-cached turn shows ~2× its real size (e.g. 2.62M). Report the
            # buckets disjoint so the sum equals the true prompt size and cache
            # reads get billed at the cached rate, not full input.
            raw_input = int(_g("input_tokens", 0) or 0)
            cached = int(_g("cached_input_tokens", 0) or 0)
            fresh_input = max(raw_input - cached, 0)
            return {
                "input_tokens": fresh_input,
                "cache_read_tokens": cached,
                "cache_creation_tokens": 0,
                "output_tokens": int(_g("output_tokens", 0) or 0),
                "context_window": None,  # filled in by _merge_model_info
                "max_output_tokens": None,
                "total_cost_usd": None,
            }
        except Exception:
            return None

    async def _merge_model_info(self, usage: dict | None, model: str) -> dict | None:
        if not usage:
            return usage
        info = await self._model_info.get(model)
        usage["context_window"] = info.get("context_window")
        usage["max_output_tokens"] = info.get("max_output_tokens")
        return usage

    @staticmethod
    def _sanitize_context_usage(usage: dict | None) -> dict | None:
        """Suppress impossible Codex context counters before they reach the UI."""
        if not usage:
            return usage
        try:
            ctx = int(usage.get("context_window", 0) or 0)
            total_in = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_tokens", 0) or 0)
                + int(usage.get("cache_creation_tokens", 0) or 0)
            )
        except Exception:
            return usage
        if ctx <= 0 or total_in <= 0:
            return usage
        # Codex turn.completed usage can accumulate prompt reads across internal
        # tool iterations. Those totals are not a truthful "context this turn"
        # measurement and can dwarf the real model window.
        if total_in <= max(ctx + 4096, int(ctx * 1.05)):
            return usage
        sanitized = dict(usage)
        sanitized["input_tokens"] = 0
        sanitized["cache_read_tokens"] = 0
        sanitized["cache_creation_tokens"] = 0
        logger.info(
            "Suppressing impossible Codex context usage: total_in=%s ctx=%s",
            total_in, ctx,
        )
        return sanitized

    async def _handle_event(
        self,
        event: Any,
        *,
        bot_slug: str,
        session_key: str,
        request_id: str,
        seq: int,
        text_parts: list[str],
        item_text: dict[str, str],
        model: str,
        actual_model_ref: list[str],
        resume_id: str | None,
        session_persisted: bool,
        item_buffers: dict[str, dict[str, Any]],
        thread: Any,
        thread_session_id: str | None = None,
        reset_evidence: CodexResetEvidence | None = None,
    ) -> tuple[int, bool, Any]:
        """Translate a single ``ThreadEvent`` into OpenClawEvents.

        Returns ``(next_seq, did_persist_session, usage_or_none)``.
        ``usage`` is non-None only on ``turn.completed``.
        """
        et = getattr(event, "type", None) or self._dig(event, "type") or ""
        # Structured reset proof is terminal exactly once. Drain trailing SDK
        # events, but never turn an error into success or duplicate its receipt.
        if reset_evidence is not None and item_buffers.get("__done_emitted__"):
            return seq, False, None

        # ---- thread.started: capture thread_id, persist on first turn ----
        if et == "thread.started":
            if reset_evidence is not None or (not session_persisted and not resume_id and bot_slug):
                thread_id = (
                    getattr(event, "thread_id", None)
                    or self._dig(event, "thread_id")
                    or self._dig(event, "thread", "id")
                    or getattr(thread, "id", None)
                )
                if thread_id and thread_session_id:
                    if reset_evidence is not None:
                        await reset_evidence.bind(self, bot_id=bot_slug,
                                                  provider_id=thread_id, model=model)
                    else:
                        await self._set_thread_session(
                            thread_session_id, bot_slug,
                            str(thread_id), model,
                        )
                    return seq, True, None
            return seq, False, None

        # ---- turn.started: no-op ----
        if et == "turn.started":
            return seq, False, None

        # ---- turn.completed: ASSISTANT_DONE with usage ----
        if et == "turn.completed":
            reset_ack = None
            if reset_evidence is not None:
                await reset_evidence.bind(
                    self, bot_id=bot_slug,
                    provider_id=getattr(thread, "id", None) or reset_evidence.provider_session_id,
                    model=model,
                )
                reset_ack = reset_evidence.acknowledgment()
            usage = getattr(event, "usage", None) or self._dig(event, "usage")
            full_text = "".join(text_parts)
            token_usage = None
            if usage is not None:
                token_usage = self._extract_token_usage(
                    usage, actual_model_or_default=actual_model_ref[0],
                )
                token_usage = await self._merge_model_info(
                    token_usage, actual_model_ref[0],
                )
                token_usage = self._sanitize_context_usage(token_usage)

            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ASSISTANT_DONE,
                text=full_text,
                model=actual_model_ref[0],
                token_usage=token_usage,
                extra_raw={"new_session_ack": reset_ack} if reset_ack else None,
            )
            item_buffers["__done_emitted__"] = True
            return seq, False, usage

        # ---- turn.failed: ERROR ----
        if et == "turn.failed":
            err = getattr(event, "error", None) or self._dig(event, "error")
            err_msg = (
                getattr(err, "message", None)
                or self._dig(err, "message")
                or "Codex turn failed"
            )
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ERROR,
                text=str(err_msg),
                model=actual_model_ref[0],
            )
            item_buffers["__done_emitted__"] = True
            return seq, False, None

        # ---- item events: started / updated / completed --------------------
        item = getattr(event, "item", None) or self._dig(event, "item")

        if et == "item.started":
            if item is None:
                return seq, False, None
            item_type = getattr(item, "type", None) or self._dig(item, "type") or ""
            item_id = getattr(item, "id", None) or self._dig(item, "id") or ""

            if item_type == "agent_message":
                # Track baseline text so item.updated can compute deltas.
                item_text[item_id] = getattr(item, "text", "") or ""
                # Emit any non-empty initial text as the first delta.
                initial = item_text[item_id]
                if initial:
                    text_parts.append(initial)
                    seq += 1
                    self._publish_event(
                        request_id, session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DELTA,
                        text=initial,
                    )
                return seq, False, None

            seq = self._emit_tool_start(
                item, request_id, session_key, seq, item_buffers,
            )
            return seq, False, None

        if et == "item.updated":
            if item is None:
                return seq, False, None
            item_type = getattr(item, "type", None) or self._dig(item, "type") or ""
            item_id = getattr(item, "id", None) or self._dig(item, "id") or ""

            if item_type == "agent_message":
                full = getattr(item, "text", "") or ""
                prev = item_text.get(item_id, "")
                if len(full) > len(prev) and full.startswith(prev):
                    delta = full[len(prev):]
                else:
                    delta = full[len(prev):] if len(full) > len(prev) else ""
                if delta:
                    item_text[item_id] = full
                    text_parts.append(delta)
                    seq += 1
                    self._publish_event(
                        request_id, session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DELTA,
                        text=delta,
                    )
                return seq, False, None

            if item_type == "command_execution":
                # Buffer aggregated_output growth for the eventual TOOL_END.
                aggregated = (
                    getattr(item, "aggregated_output", None)
                    or self._dig(item, "aggregated_output")
                    or ""
                )
                if aggregated:
                    buf = item_buffers.setdefault(item_id, {})
                    buf["aggregated_output"] = aggregated
            return seq, False, None

        if et == "item.completed":
            if item is None:
                return seq, False, None
            item_type = getattr(item, "type", None) or self._dig(item, "type") or ""
            item_id = getattr(item, "id", None) or self._dig(item, "id") or ""

            if item_type == "agent_message":
                # Final text — make sure text_parts captures the full message
                # in case we missed an updated event.
                full = getattr(item, "text", "") or ""
                prev = item_text.get(item_id, "")
                if len(full) > len(prev):
                    delta = full[len(prev):] if full.startswith(prev) else full[len(prev):]
                    if delta:
                        item_text[item_id] = full
                        text_parts.append(delta)
                        seq += 1
                        self._publish_event(
                            request_id, session_key, seq,
                            kind=AgentEventKind.ASSISTANT_DELTA,
                            text=delta,
                        )
                return seq, False, None

            seq = self._emit_tool_end(
                item, request_id, session_key, seq, item_buffers, text_parts,
            )
            return seq, False, None

        # ---- error event ----
        if et == "error":
            err_msg = (
                getattr(event, "message", None)
                or self._dig(event, "message")
                or "Codex stream error"
            )
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ERROR,
                text=str(err_msg),
                model=actual_model_ref[0],
            )
            if reset_evidence is not None:
                item_buffers["__done_emitted__"] = True
            return seq, False, None

        logger.debug("Unhandled Codex event: type=%s", et)
        return seq, False, None

    def _emit_tool_start(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
    ) -> int:
        """Emit TOOL_START for a codex thread item.

        Tool names are emitted as their **native codex names** (not aliased
        to Claude tool names). The UI's provider-aware tool-render registry
        dispatches on (provider="codex", tool_name) so each item type can
        render its true shape — e.g. ``file_change`` carries only
        ``path + kind`` (no diff content, since the SDK doesn't ship it).
        """
        item_type = self._dig(item, "type") or self._dig(item, "kind") or ""
        item_id = self._dig(item, "id") or ""

        tool_name: str | None = None
        tool_args: dict = {}

        if item_type in ("commandExecution", "command_execution"):
            tool_name = "shell"
            cmd = self._dig(item, "command") or self._dig(item, "cmd") or ""
            cwd = self._dig(item, "cwd") or self._dig(item, "working_dir") or ""
            tool_args = {"command": cmd, "cwd": cwd}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "command"})
        elif item_type in ("fileChange", "file_change"):
            # One TOOL_START per change so the UI gets per-file cards.
            return self._emit_filechange_starts(item, request_id, session_key, seq, item_buffers)
        elif item_type in ("webSearch", "web_search"):
            action = self._dig(item, "action") or {}
            atype = self._dig(action, "type") or ""
            tool_name = "web_search"
            tool_args = {
                "action": atype,
                "query": self._dig(action, "query") or "",
                "url": self._dig(action, "url") or "",
                "pattern": self._dig(action, "pattern") or "",
            }
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "web"})
        elif item_type in ("mcpToolCall", "mcp_tool_call"):
            # Pass through the MCP tool's real name; UI dispatches generically.
            tool_name = self._dig(item, "tool") or self._dig(item, "name") or "mcp_tool"
            tool_args = self._dig(item, "arguments") or self._dig(item, "args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"raw": str(tool_args)}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "mcp"})
        elif item_type in ("dynamicToolCall", "dynamic_tool_call"):
            tool_name = self._dig(item, "tool") or self._dig(item, "name") or "tool"
            tool_args = self._dig(item, "arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"raw": str(tool_args)}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "dynamic"})
        elif item_type in ("imageView", "image_view"):
            tool_name = "image_view"
            tool_args = {"path": self._dig(item, "path") or ""}
            item_buffers.setdefault(item_id, {"tool_name": tool_name, "type": "image"})
        elif item_type in ("agentMessage", "agent_message"):
            # Final agent message — accumulated via deltas; no tool card.
            return seq

        if tool_name is None:
            return seq

        seq += 1
        self._publish_event(
            request_id, session_key, seq,
            kind=AgentEventKind.TOOL_START,
            tool_name=tool_name,
            tool_arguments=tool_args,
        )
        return seq

    def _emit_filechange_starts(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
    ) -> int:
        """Emit one ``file_change`` TOOL_START per touched file.

        The codex SDK exposes ``file_change`` items as ``{path, kind}`` per
        change — no diff content, no before/after text. We pass that
        through verbatim; the UI's ``FileChangeBody`` (registered for
        ``provider="codex", tool_name="file_change"``) renders just the
        path + a kind badge ("Updated" / "Created" / "Deleted").
        """
        item_id = self._dig(item, "id") or ""
        changes = self._dig(item, "changes") or []

        emitted: list[dict] = []
        for change in changes or []:
            path = self._dig(change, "path") or self._dig(change, "file_path") or ""
            if not path:
                continue
            kind = self._dig(change, "kind") or self._dig(change, "type") or "update"
            args = {"file_path": path, "kind": kind}

            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_START,
                tool_name="file_change",
                tool_arguments=args,
            )
            emitted.append({"tool_name": "file_change", "args": args, "path": path})

        if item_id:
            item_buffers.setdefault(item_id, {})["filechange_started"] = emitted
            item_buffers[item_id]["type"] = "file"
        return seq

    def _emit_tool_end(
        self,
        item: Any,
        request_id: str,
        session_key: str,
        seq: int,
        item_buffers: dict,
        text_parts: list[str],
    ) -> int:
        item_type = self._dig(item, "type") or self._dig(item, "kind") or ""
        item_id = self._dig(item, "id") or ""
        buf = item_buffers.get(item_id, {})

        if item_type in ("agentMessage", "agent_message"):
            phase = self._dig(item, "phase") or self._dig(item, "stage") or ""
            if phase == "final_answer":
                final_text = (
                    self._dig(item, "text")
                    or self._dig(item, "content")
                    or ""
                )
                if final_text and not text_parts:
                    text_parts.append(str(final_text))
            return seq

        if item_type in ("commandExecution", "command_execution"):
            output = (
                buf.get("aggregated_output")
                or self._dig(item, "aggregated_output")
                or "".join(buf.get("output", []))
            )
            exit_code = self._dig(item, "exit_code") or self._dig(item, "exitCode") or 0
            duration_ms = self._dig(item, "duration_ms") or self._dig(item, "durationMs") or 0
            try:
                exit_code_int = int(exit_code) if exit_code is not None else 0
            except (ValueError, TypeError):
                exit_code_int = 0
            if exit_code_int:
                result = json.dumps({
                    "output": output,
                    "exitCode": exit_code_int,
                    "durationMs": duration_ms,
                })
            else:
                result = output
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_END,
                tool_name="shell",
                tool_result=result,
            )
            return seq

        if item_type in ("fileChange", "file_change"):
            # One TOOL_END per matching TOOL_START. Result is the change
            # status (completed/failed) — codex doesn't expose patch text.
            started = buf.get("filechange_started") or []
            status = self._dig(item, "status") or "completed"
            for _entry in started:
                seq += 1
                self._publish_event(
                    request_id, session_key, seq,
                    kind=AgentEventKind.TOOL_END,
                    tool_name="file_change",
                    tool_arguments=_entry.get("args") or {},
                    tool_result=str(status),
                )
            return seq

        if item_type in ("webSearch", "web_search"):
            results = self._dig(item, "results") or self._dig(item, "result") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_END,
                tool_name="web_search",
                tool_result=str(results),
            )
            return seq

        if item_type in ("mcpToolCall", "mcp_tool_call"):
            tool_name = buf.get("tool_name") or self._dig(item, "tool") or "mcp_tool"
            result = self._dig(item, "result") or self._dig(item, "error") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_END,
                tool_name=tool_name,
                tool_result=str(result),
            )
            return seq

        if item_type in ("dynamicToolCall", "dynamic_tool_call"):
            tool_name = buf.get("tool_name") or self._dig(item, "tool") or "tool"
            result = self._dig(item, "result") or self._dig(item, "error") or ""
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_END,
                tool_name=tool_name,
                tool_result=str(result),
            )
            return seq

        if item_type in ("imageView", "image_view"):
            seq += 1
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.TOOL_END,
                tool_name="image_view",
                tool_result="[image]",
            )
            return seq

        return seq

    @staticmethod
    def _note_method(note: Any) -> str:
        if hasattr(note, "method"):
            return getattr(note, "method") or ""
        if isinstance(note, dict):
            return note.get("method", "") or ""
        return ""

    @staticmethod
    def _note_payload(note: Any) -> Any:
        if hasattr(note, "payload"):
            return getattr(note, "payload")
        if isinstance(note, dict):
            return note.get("payload") or note.get("params") or note
        return note

    @staticmethod
    def _dig(obj: Any, *keys: str) -> Any:
        cur: Any = obj
        for k in keys:
            if cur is None:
                return None
            if hasattr(cur, k):
                cur = getattr(cur, k)
                continue
            if isinstance(cur, dict):
                cur = cur.get(k)
                continue
            return None
        return cur
