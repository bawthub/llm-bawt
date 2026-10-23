"""Codex bridge: Redis command listener + Codex SDK event translator.

Mirrors ``src/claude_code_bridge/bridge.py`` but talks to the OpenAI Codex
SDK (``openai_codex_sdk``) instead of the Claude Agent SDK. Architectural
decisions are documented in the project's contextPrompt and the section
headers below.
"""

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
from agent_bridge.changed_file_lifecycle import ChangedFileLifecycleMixin
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue


from .transport import CodexTransport, validate_auth_json
from .rollout_usage import RolloutUsageReader
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
from .session_ops import CodexSessionMixin
from .command_ops import CodexCommandMixin
from .event_ops import CodexEventMixin

logger = logging.getLogger(__name__)



# --- Bridge -----------------------------------------------------------------


class CodexBridge(
    ChangedFileLifecycleMixin,
    CodexSessionMixin,
    CodexCommandMixin,
    CodexEventMixin,
):
    """Reads chat.send commands from Redis, runs them through the Codex
    SDK, and publishes AgentEvent-formatted results back to Redis."""

    DEFAULT_REQUEST_TIMEOUT = 900

    # TASK-209: cache cleanup parameters
    _CLEANUP_INTERVAL = 6 * 3600  # every 6 hours
    _CACHE_MAX_AGE = 24 * 3600
    _CACHE_DIRS = (
        "cache",
        "log",
        "tmp",
        ".tmp",
        "shell_snapshots",
    )
    # auth/config files we never delete:
    _PRESERVE_FILENAMES = frozenset(
        {"auth.json", "config.toml", "installation_id", "version.json"}
    )

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "codex",
        app_api_url: str = "",
        default_model: str = "gpt-5.4",
        cwd: str = "/home/bridge/dev",
        codex_bin: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._default_model = default_model
        self._cwd = cwd
        self._request_timeout = request_timeout or self.DEFAULT_REQUEST_TIMEOUT
        self._command_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._redis = None
        self._session_queue = SessionQueue()
        self._transport = CodexTransport(codex_bin=codex_bin)
        self._model_info = _ModelInfoCache(app_api_url)
        self._rollout_usage = RolloutUsageReader()
        # request_id → frontend user-message UUID, populated in _handle_send
        # and read by _publish_event so every emitted AgentEvent (tool_*,
        # assistant_*, etc.) carries the originating message id.  Cleared on
        # the end of the send.
        self._trigger_message_ids: dict[str, str] = {}

    # ----- Public properties (used by __main__ for log lines) -------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def cwd(self) -> str:
        return self._cwd

    # ----- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        self._cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        logger.info(
            "CodexBridge started (backend=%s, model=%s, cwd=%s)",
            self._backend_name, self._default_model, self._cwd,
        )

    async def stop(self) -> None:
        for task in (self._command_task, self._cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._transport.reset()
        self._publisher.close()
        logger.info("CodexBridge stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ----- TASK-204: supervisor / self-heal -------------------------------

    def _ensure_codex(self):
        """Return the shared ``Codex`` SDK handle (lazy-built on first use)."""
        return self._transport.ensure_codex()

    def _supervisor_teardown(self) -> None:
        """Drop the cached ``Codex`` so the next ensure rebuilds.

        Re-validates auth.json defensively — useful when a fresh
        ``codex login`` is the recovery path. The codex Rust binary is
        spawned per-turn, so there's no subprocess to kill here; the next
        ``run_streamed`` will pick up the rewritten ``auth.json`` on disk.
        """
        try:
            validate_auth_json()
        except RuntimeError as e:
            logger.warning("auth.json still invalid during teardown: %s", e)
        self._transport.reset()


    # ----- TASK-206: session persistence helpers --------------------------




    # ----- TASK-205: Redis command listener -------------------------------


    # ----- TASK-205+207+208: chat.send handler ----------------------------


    # ----- Event iteration with timeout -----------------------------------



    # ----- TASK-208: prompt input building --------------------------------



    # ----- TASK-208: token usage extraction -------------------------------




    # ----- TASK-207: ThreadEvent → AgentEvent mapping ------------------


    # ----- Item-shape helpers ---------------------------------------------







    # ----- TASK-205: RPC handler -----------------------------------------



    # ----- TASK-209: cache cleanup --------------------------------------

    async def _periodic_cache_cleanup(self) -> None:
        codex_dir = Path.home() / ".codex"
        sessions_dir = codex_dir / "sessions"
        while True:
            try:
                await asyncio.sleep(self._CLEANUP_INTERVAL)
                now = time.time()
                total_removed = 0

                # Per-name top-level dirs to prune
                for dirname in self._CACHE_DIRS:
                    cache_path = codex_dir / dirname
                    if not cache_path.is_dir():
                        continue
                    for entry in cache_path.iterdir():
                        if entry.name in self._PRESERVE_FILENAMES:
                            continue
                        try:
                            age = now - entry.stat().st_mtime
                            if age > self._CACHE_MAX_AGE:
                                if entry.is_dir():
                                    shutil.rmtree(entry, ignore_errors=True)
                                else:
                                    entry.unlink()
                                total_removed += 1
                        except OSError:
                            pass

                # NEVER prune rollout JSONLs under sessions/YYYY/MM/DD/*.jsonl.
                # They are the transcripts `resume=` replays — each thread's
                # memory. A 24h sweep here lobotomizes any thread idle for a
                # day (same bug as the claude-code bridge, fixed together).
                    # Best-effort empty-dir cleanup
                    self._prune_empty_dirs(sessions_dir)

                if total_removed:
                    logger.info(
                        "Cache cleanup: removed %d stale entries", total_removed,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cache cleanup error")

    @staticmethod
    def _prune_empty_dirs(root: Path) -> None:
        # Walk bottom-up; rmdir empty leaves.
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                pass

    # ----- Event publishing ---------------------------------------------

    @staticmethod
    def _shorten_paths(s: str | None) -> str | None:
        if s is None:
            return None
        return s.replace("/home/bridge/", "~/").replace("/home/bridge", "~")

    @classmethod
    def _shorten_paths_in_dict(cls, d: dict | None) -> dict | None:
        if d is None:
            return None
        out: dict = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = cls._shorten_paths(v)
            elif isinstance(v, dict):
                out[k] = cls._shorten_paths_in_dict(v)
            elif isinstance(v, list):
                out[k] = [
                    cls._shorten_paths(item) if isinstance(item, str)
                    else cls._shorten_paths_in_dict(item) if isinstance(item, dict)
                    else item
                    for item in v
                ]
            else:
                out[k] = v
        return out

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: AgentEventKind,
        text: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: str | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
        extra_raw: dict | None = None,
    ) -> None:
        text = self._shorten_paths(text)
        tool_result = self._shorten_paths(tool_result)
        tool_arguments = self._shorten_paths_in_dict(tool_arguments)
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "tool": tool_name, "seq": seq},
            seq,
        )
        event = AgentEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=request_id,
            kind=kind,
            origin="system",
            text=text,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            raw=dict(extra_raw or {}),
            token_usage=token_usage,
            provider=self._backend_name,
            # Inherit from the active run so every event for this request
            # (TOOL_START, TOOL_END, ASSISTANT_*, RUN_*) carries the same
            # originating user-message UUID without each call site needing
            # to remember to thread it through.
            trigger_message_id=self._trigger_message_ids.get(request_id),
        )
        self._publish_run_event_with_changed_file(request_id, event)

    def _publish_session_reset_unified(
        self,
        bot_id: str,
        session_key: str,
        *,
        had_session: bool = False,
        user_id: str = "nick",
    ) -> None:
        """Publish SESSION_RESET on the unified SSE stream.

        Lets the frontend clear its visible message buffer for ``bot_id``
        deterministically rather than racing turn_complete timing.
        See TASK-249.
        """
        if not bot_id:
            return
        try:
            self._publisher.publish_unified_event(bot_id, user_id, {
                "_type": "session_reset",
                "bot_id": bot_id,
                "user_id": user_id,
                "session_key": session_key,
                "had_session": bool(had_session),
                "text": "Session reset. Ready for a new conversation.",
                "provider": self._backend_name,
                "ts": datetime.now(timezone.utc).timestamp(),
            })
        except Exception:
            logger.exception(
                "Failed to publish unified session_reset for bot=%s session=%s",
                bot_id, session_key,
            )
