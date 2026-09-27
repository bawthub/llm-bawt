"""Local model bridge: Redis command listener + local-inference text streamer.

Mirrors ``src/codex_bridge/bridge.py`` structurally, but with a key
simplification: local models do plain synchronous text generation.  There is
NO interactive-tool machinery here — no AskUserQuestion, no can_use_tool, no
tool_result handling.  Only two things are supported:

  * ``chat.send``  — stream generated text back as AgentEvents.
  * ``rpc.call`` method ``chat.abort`` — cooperatively cancel an active turn.

All CUDA work runs on the bridge's own single-worker ThreadPoolExecutor, which
serializes generation across sessions (one local model loaded at a time).  A
CUDA abort() therefore can only crash this process — never the main app.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from datetime import datetime, timezone
from typing import Any

from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from llm_bawt.models.message import Message
from llm_bawt.utils.config import Config

from .inference import LocalModelLoader

logger = logging.getLogger(__name__)

CONSUMER_GROUP = "local-model-bridge"
CONSUMER_NAME = "worker-0"


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


class LocalModelBridge:
    """Reads chat.send commands from Redis, runs them through a locally loaded
    model, and publishes AgentEvent-formatted text deltas back to Redis."""

    DEFAULT_REQUEST_TIMEOUT = 1800

    def __init__(
        self,
        publisher: RedisPublisher,
        *,
        backend_name: str = "local",
        app_api_url: str = "",
        request_timeout: float | None = None,
    ) -> None:
        self._publisher = publisher
        self._backend_name = backend_name
        self._app_api_url = app_api_url
        self._request_timeout = request_timeout or self.DEFAULT_REQUEST_TIMEOUT
        self._command_task: asyncio.Task | None = None
        self._redis = None
        self._session_queue = SessionQueue()
        self._config = Config()
        self._loader = LocalModelLoader(self._config, app_api_url)
        # Single worker so all CUDA / generation work is serialized — only one
        # local model is ever loaded, and only one generation runs at a time.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-infer")
        # request_id -> originating frontend user-message UUID (stamped on every
        # emitted event); cleared on publish_run_done.
        self._trigger_message_ids: dict[str, str] = {}

    # ----- Public properties ----------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    # ----- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        self._command_task = asyncio.create_task(self._command_listener())
        logger.info("LocalModelBridge started (backend=%s)", self._backend_name)

    async def stop(self) -> None:
        if self._command_task:
            self._command_task.cancel()
            try:
                await self._command_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self._loader.unload()
        except Exception:
            logger.debug("loader.unload raised during stop", exc_info=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._publisher.close()
        logger.info("LocalModelBridge stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ----- Command listener -----------------------------------------------

    async def _command_listener(self) -> None:
        import redis.asyncio as aioredis

        conn_kwargs = self._publisher._redis.connection_pool.connection_kwargs
        host = conn_kwargs.get("host", "localhost")
        port = conn_kwargs.get("port", 6379)
        db = conn_kwargs.get("db", 0)
        # socket_timeout=None: redis-py 8.0 defaults to 5s, which races our
        # blocking XREADGROUP(block=5000) reads. Bound only the connect.
        async_redis = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=5,
        )
        await async_redis.ping()
        self._redis = async_redis

        try:
            # id="$" so a brand-new group skips historical commands (authored
            # before this bridge existed / targeting other backends).
            await async_redis.xgroup_create(
                COMMANDS_STREAM, CONSUMER_GROUP, id="$", mkstream=True,
            )
        except Exception:
            pass

        logger.info(
            "Command listener started on %s (group=%s)",
            COMMANDS_STREAM, CONSUMER_GROUP,
        )

        while True:
            try:
                results = await async_redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {COMMANDS_STREAM: ">"},
                    count=1,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        action = fields.get("action", "")
                        backend = fields.get("backend", "")

                        if action == "chat.send" and backend != self._backend_name:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                            continue
                        # RPCs that explicitly target a different backend are
                        # not ours. Legacy callers omit `backend`; for those we
                        # only act on methods we understand (chat.abort).
                        if action == "rpc.call" and backend and backend != self._backend_name:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
                            continue

                        if action == "chat.send":
                            asyncio.create_task(
                                self._handle_send(fields, msg_id, async_redis)
                            )
                        elif action == "rpc.call":
                            asyncio.create_task(
                                self._handle_rpc(fields, msg_id, async_redis)
                            )
                        else:
                            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

            except asyncio.CancelledError:
                await async_redis.aclose()
                raise
            except Exception:
                logger.exception("Command listener error")
                await asyncio.sleep(2)

    # ----- chat.send handler ----------------------------------------------

    async def _handle_send(self, fields: dict, msg_id: str, async_redis) -> None:
        request_id = fields.get("request_id", "")
        session_key = fields.get("session_key", "")
        message = fields.get("message", "")
        system_prompt = fields.get("system_prompt") or None
        model_alias = fields.get("model") or ""
        trigger_message_id = (fields.get("trigger_message_id") or "").strip() or None

        # The app nests the original local model definition in the command so
        # the bridge can resolve the model even if the /v1/models lookup fails.
        fallback_def: dict[str, Any] | None = None
        raw_def = fields.get("local_model_definition")
        if raw_def:
            try:
                parsed = json.loads(raw_def) if isinstance(raw_def, str) else raw_def
                if isinstance(parsed, dict):
                    fallback_def = parsed
            except (json.JSONDecodeError, TypeError):
                fallback_def = None

        if not request_id or not message:
            logger.warning("Invalid send command: missing request_id or message")
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
            return

        if not model_alias:
            logger.warning("Invalid send command: missing model alias")
            self._publish_event(
                request_id, session_key, 1,
                kind=AgentEventKind.ERROR,
                text="Local model bridge: no model alias supplied.",
            )
            self._publisher.publish_run_done(request_id)
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)
            return

        if trigger_message_id:
            self._trigger_message_ids[request_id] = trigger_message_id

        finalized = False

        async def finalize() -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            self._publisher.publish_run_done(request_id)
            self._trigger_message_ids.pop(request_id, None)
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

        async with self._session_queue.active(
            session_key, request_id=request_id, on_cancel=finalize,
        ):
            logger.info(
                "Handling send: request_id=%s session=%s model=%s system_prompt=%s msg=%.60s...",
                request_id, session_key, model_alias,
                f"{len(system_prompt)} chars" if system_prompt else "none",
                message,
            )

            # Fresh cooperative-cancel event for this turn.
            cancel_event = (
                self._session_queue.cancel_event(session_key) if session_key else None
            )
            seq = 0
            text_parts: list[str] = []
            loop = asyncio.get_running_loop()
            stop_generation = Event()
            worker = None

            try:
                # Load (or switch to) the requested model on the inference
                # worker thread so VRAM allocation doesn't block the event loop.
                worker = loop.run_in_executor(
                    self._executor,
                    lambda: self._loader.get_client(
                        model_alias, fallback_definition=fallback_def
                    ),
                )
                client = await asyncio.shield(worker)

                messages = self._build_messages(system_prompt, message)

                # Drain the synchronous generator on the inference worker,
                # bridging chunks back to the event loop via a thread-safe
                # queue so we can publish deltas and check cancellation.
                chunk_queue: asyncio.Queue = asyncio.Queue()
                _SENTINEL = object()

                def _produce() -> None:
                    try:
                        for chunk in client.stream_raw(messages):
                            if stop_generation.is_set():
                                break
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                    except Exception as exc:  # noqa: BLE001 — forwarded below
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait, ("__error__", exc)
                        )
                    finally:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, _SENTINEL)

                worker = loop.run_in_executor(self._executor, _produce)

                aborted = False
                while True:
                    item = await chunk_queue.get()
                    if item is _SENTINEL:
                        break
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                        raise item[1]
                    # Stop the worker at its next token boundary.
                    if cancel_event is not None and cancel_event.is_set():
                        aborted = True
                        stop_generation.set()
                        logger.info(
                            "chat.abort honored mid-stream: session=%s request_id=%s",
                            session_key, request_id,
                        )
                        break
                    if not item:
                        continue
                    seq += 1
                    text_parts.append(str(item))
                    self._publish_event(
                        request_id, session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DELTA,
                        text=str(item),
                        model=model_alias,
                    )

                # Wait for the producer to finish so the worker is free for the
                # next turn (it cannot be cancelled mid-token).
                try:
                    await asyncio.shield(worker)
                except Exception:
                    logger.debug("producer task raised after drain", exc_info=True)

                full_text = "".join(text_parts)
                seq += 1
                if not aborted:
                    self._publish_event(
                        request_id, session_key, seq,
                        kind=AgentEventKind.ASSISTANT_DONE,
                        text=full_text,
                        model=model_alias,
                    )
                if aborted:
                    logger.info(
                        "Send aborted: request_id=%s session=%s (%d chars streamed)",
                        request_id, session_key, len(full_text),
                    )

            except asyncio.CancelledError:
                logger.info(
                    "Send cancelled: request_id=%s session=%s", request_id, session_key,
                )
                stop_generation.set()
                # Cancellation cannot interrupt native model loading/a token.
                # Retain the lock until the worker stops, not merely until the
                # asyncio wrapper is cancelled. Repeated aborts are idempotent.
                if worker is not None:
                    try:
                        await asyncio.shield(worker)
                    except Exception:
                        logger.debug("worker failed during abort", exc_info=True)
                raise
            except Exception as e:
                logger.exception("Local model send failed: %s", e)
                self._publish_event(
                    request_id, session_key, seq + 1,
                    kind=AgentEventKind.ERROR,
                    text=f"Local model error: {e}",
                    model=model_alias,
                )
            finally:
                await finalize()

    @staticmethod
    def _build_messages(system_prompt: str | None, message: str) -> list[Message]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=message))
        return messages

    # ----- rpc.call handler -----------------------------------------------

    async def _handle_rpc(self, fields: dict, msg_id: str, async_redis) -> None:
        request_id = fields.get("request_id", "")
        method = fields.get("method", "")
        params_raw = fields.get("params", "{}")
        try:
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        except (json.JSONDecodeError, TypeError):
            params = {}
        if not isinstance(params, dict):
            params = {}

        try:
            if method == "chat.abort":
                session_key = params.get("sessionKey", "")
                target_request_id = params.get("requestId", "")
                if (
                    not isinstance(session_key, str) or not session_key.strip()
                    or not isinstance(target_request_id, str) or not target_request_id.strip()
                ):
                    raise ValueError("chat.abort requires sessionKey and requestId")
                cancelled = self._session_queue.cancel_request(session_key, target_request_id)
                self._publisher.publish_rpc_result(
                    request_id,
                    {"ok": True, "aborted": cancelled, "requestId": target_request_id,
                     "detail": "task_cancelled" if cancelled else "no_matching_request"},
                )
            else:
                # Not a method this bridge handles — only ack legacy/no-backend
                # calls so we don't swallow another bridge's RPC.
                if (fields.get("backend") or "").strip() == self._backend_name:
                    self._publisher.publish_rpc_result(
                        request_id, {"ok": False, "error": f"unknown method: {method}"},
                    )
        except Exception as e:
            logger.warning("RPC %s failed: %s", method, e)
            self._publisher.publish_rpc_result(
                request_id, {"ok": False, "error": str(e)},
            )
        finally:
            await async_redis.xack(COMMANDS_STREAM, CONSUMER_GROUP, msg_id)

    # ----- event publish helper -------------------------------------------

    def _publish_event(
        self,
        request_id: str,
        session_key: str,
        seq: int,
        *,
        kind: AgentEventKind,
        text: str | None = None,
        model: str | None = None,
        token_usage: dict | None = None,
    ) -> None:
        event_id = synthesize_event_id(
            session_key, kind.value,
            {"text": text, "seq": seq},
            seq,
        )
        event = AgentEvent(
            event_id=event_id,
            session_key=session_key,
            run_id=request_id,
            kind=kind,
            origin="system",
            text=text,
            model=model,
            seq=seq,
            timestamp=datetime.now(timezone.utc),
            raw={},
            token_usage=token_usage,
            provider=self._backend_name,
            trigger_message_id=self._trigger_message_ids.get(request_id),
        )
        self._publisher.publish_run_event(request_id, event)
