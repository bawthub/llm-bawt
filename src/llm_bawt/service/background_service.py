"""Background task service and chat execution runtime.

Core orchestrator that composes focused mixins:
- InstanceManagerMixin  — model/bot instance management
- TurnLifecycleMixin    — turn persistence + generation cancellation
- ChatStreamingMixin    — streaming chat completions + bridge
- ChatNonStreamMixin    — non-streaming (stream=false) chat completions
- BackgroundTasksMixin  — background task processing
"""

import asyncio
import json
import threading
import time
from queue import PriorityQueue
from typing import Any
from urllib.parse import urlparse

from ..utils.config import Config
from .background_tasks import BackgroundTasksMixin
from .chat_nonstream import ChatNonStreamMixin
from .chat_streaming import ChatStreamingMixin
from .instance_manager import InstanceManagerMixin
from .logging import get_service_logger
from .schemas import ServiceStatusResponse
from .tasks import Task, TaskResult, TaskStatus
from .turn_lifecycle import TurnLifecycleMixin

log = get_service_logger(__name__)

SERVICE_VERSION = "0.1.0"


def _is_tcp_listening(host: str, port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


_mcp_thread: threading.Thread | None = None


def _ensure_mcp_server(config: Config) -> None:
    """Ensure the llm-bawt MCP server is running and configure the service to use it.

    This makes memory retrieval happen via MCP tool calls (e.g. tools/memory_search),
    which can be logged distinctly from embedded DB access.
    """
    global _mcp_thread

    # Default to local llm-bawt MCP server for llm-service if not configured.
    if not config.MCP_SERVER_URL:
        config.MCP_SERVER_URL = "http://127.0.0.1:8001"

    parsed = urlparse(config.MCP_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8001

    # If something is already listening, assume it's the llm-bawt MCP server.
    if _is_tcp_listening(host, port):
        log.info("llm-bawt MCP server already listening at %s", config.MCP_SERVER_URL)
        return

    # Start the MCP server in-process (HTTP transport) on a daemon thread.
    def _run():
        try:
            from ..mcp_server.server import run_server
            run_server(transport="streamable-http", host=host, port=port)
        except Exception as e:
            log.error("Failed to start llm-bawt MCP server: %s", e)

    _mcp_thread = threading.Thread(target=_run, daemon=True, name="llm-bawt-mcp")
    _mcp_thread.start()
    log.info("Started llm-bawt MCP server at %s", config.MCP_SERVER_URL)


class BackgroundService(
    InstanceManagerMixin,
    TurnLifecycleMixin,
    ChatStreamingMixin,
    ChatNonStreamMixin,
    BackgroundTasksMixin,
):
    """
    The main background service that processes async tasks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self.tasks_processed = 0
        self._task_queue: PriorityQueue = PriorityQueue()
        self._results: dict[str, TaskResult] = {}
        self._submitted_tasks: dict[str, Task] = {}
        self._result_events: dict[str, asyncio.Event] = {}
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

        # Initialize model lifecycle manager (singleton)
        from ..core.model_lifecycle import get_model_lifecycle
        self._model_lifecycle = get_model_lifecycle(config)

        # Cached LLMBawt instances keyed by (model_alias, bot_id, user_id)
        self._llm_bawt_cache: dict[tuple[str, str, str], Any] = {}
        self._cache_lock = asyncio.Lock()

        # Cached LLM clients keyed by model_alias only
        # This prevents loading the same model (especially GGUF) multiple times
        # when different bot contexts need the same underlying model
        self._client_cache: dict[str, Any] = {}

        # Bounded pool for background jobs (memory extraction, history
        # summarization, profile/consolidation maintenance, media GC). Kept
        # SEPARATE from the default executor so a burst of per-bot background
        # fan-out can never starve the workers that interactive agent-bridge
        # relays need — that contention stalled continuation turns by minutes
        # (TASK-283). Local GPU inference no longer runs in-process — it moved to
        # the standalone local_model_bridge (TASK-276/278) — so there is no
        # single-thread LLM executor here anymore; all chat streams on the
        # default pool.
        from concurrent.futures import ThreadPoolExecutor
        self._bg_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg")

        # Cancellation support: when a new request comes in, cancel the current one
        # This handles the case where UI sends partial transcriptions that build up
        self._current_generation_cancel: threading.Event | None = None
        self._generation_done: threading.Event | None = None  # Signals when generation finishes
        self._cancel_lock = threading.Lock()

        # Memory client cache keyed by (bot_id, user_id).
        self._memory_clients: dict[tuple[str, str | None], Any] = {}
        self._memory_clients_lock = threading.Lock()
        from .turn_logs import TurnLogStore
        self._turn_log_store = TurnLogStore(config)
        self._inter_bot_dispatcher: Any | None = None
        self._mcp_continuation_task: asyncio.Task | None = None
        # Persistent registry of in-flight AskUserQuestion pauses.  Lets the
        # chat UI hydrate open pickers on page load / second tab / after a
        # bridge restart instead of relying solely on the live SSE event.
        from .chat_pending_questions import PendingQuestionStore
        self._pending_question_store = PendingQuestionStore(config)

        # TASK-290: durable registry + audit of approval-gated tool calls. Used
        # by chat_streaming to persist a request row when the bridge emits
        # APPROVAL_REQUIRED, and by the resolve endpoint.
        from ..approval_policies import ToolApprovalPolicyStore
        self._tool_approval_policy_store = ToolApprovalPolicyStore(config)

        # OpenClaw WS session bridge (set by api.py lifespan if enabled)
        self._session_bridge: Any | None = None

        # Keyed pool of background-task API clients, one per resolved model
        # alias (TASK-281). Replaces the old single reassignable _bg_client slot
        # that thrashed/raced when concurrent background jobs resolved different
        # models — the loser got a torn-down/None client and silently produced
        # zero results. Created once per model, thereafter only read. Isolated
        # from the chat model lifecycle.
        self._bg_client_cache: dict[str, Any] = {}
        # Guards create-and-insert into _bg_client_cache. A plain threading.Lock
        # (NOT the asyncio _cache_lock) because _get_background_client is sync and
        # runs in the _bg_executor thread pool. Steady-state reads are lock-free.
        self._bg_client_lock = threading.Lock()

        # Session model overrides: when user switches model via tool,
        # remember it for the rest of the session. Keyed by (bot_id, user_id).
        self._session_model_overrides: dict[tuple[str, str], str] = {}

        # Model configuration
        self._available_models: list[str] = []
        self._default_model: str | None = None
        self._default_bot: str = config.DEFAULT_BOT or "nova"
        self._load_available_models()

        # Register callback for model unload - clears caches when model changes
        self._model_lifecycle.on_model_unloaded(self._on_model_unloaded)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def model_lifecycle(self):
        """Get the model lifecycle manager for tool access."""
        return self._model_lifecycle

    # ---- Task queue ----

    def submit_task(self, task: Task) -> str:
        """Submit a task to the processing queue."""
        from dataclasses import dataclass, field as dataclass_field

        @dataclass(order=True)
        class PrioritizedTask:
            priority: int
            timestamp: float
            task: Task = dataclass_field(compare=False)

        prioritized = PrioritizedTask(
            priority=-task.priority,
            timestamp=time.time(),
            task=task,
        )
        self._task_queue.put(prioritized)
        self._submitted_tasks[task.task_id] = task
        self._result_events[task.task_id] = asyncio.Event()
        return task.task_id

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get the result of a completed task."""
        return self._results.get(task_id)

    async def wait_for_result(self, task_id: str, timeout: float = 30.0) -> TaskResult | None:
        """Wait for a task result with timeout."""
        event = self._result_events.get(task_id)
        if not event:
            return self._results.get(task_id)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._results.get(task_id)
        except asyncio.TimeoutError:
            return None

    async def worker_loop(self):
        """Main worker loop that processes tasks from the queue."""
        log.debug("Background task worker started")

        while not self._shutdown_event.is_set():
            try:
                try:
                    prioritized = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._task_queue.get(timeout=1.0)
                    )
                except Exception:
                    continue

                task = prioritized.task
                log.task_submitted(task.task_id, task.task_type.value, task.bot_id, task.payload)

                result = await self.process_task(task)
                self._results[task.task_id] = result
                self.tasks_processed += 1

                if task.task_id in self._result_events:
                    self._result_events[task.task_id].set()

                # Cleanup old results
                if len(self._results) > 1000:
                    oldest = sorted(self._results.keys())[:100]
                    for key in oldest:
                        self._results.pop(key, None)
                        self._submitted_tasks.pop(key, None)
                        self._result_events.pop(key, None)

            except Exception as e:
                log.exception(f"Worker loop error: {e}")
                await asyncio.sleep(1)

        log.debug("Background task worker stopped")

    def start_worker(self):
        """Start the background worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker_loop())

    async def shutdown(self):
        """Gracefully shutdown the service."""
        log.shutdown()
        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
        # Drop in-flight background jobs; don't block shutdown waiting on them.
        self._bg_executor.shutdown(wait=False, cancel_futures=True)

    def get_status(self) -> ServiceStatusResponse:
        """Get service status."""
        from ..core.status import _collect_mcp_info, _collect_memory_info
        from ..utils.config import has_database_credentials

        current = self._model_lifecycle.current_model
        memory_info = _collect_memory_info(self.config, self._default_bot)
        mcp_info = _collect_mcp_info(self.config, self._default_bot)

        db_required = has_database_credentials(self.config)
        db_ok = memory_info.postgres_connected if db_required else True
        mcp_ok = mcp_info.status == "up"
        healthy = db_ok and mcp_ok

        checks = {
            "service": "up",
            "database": "up" if db_ok else ("disabled" if not db_required else "down"),
            "mcp": mcp_info.status,
        }

        return ServiceStatusResponse(
            status="ok" if healthy else "degraded",
            healthy=healthy,
            uptime_seconds=self.uptime_seconds,
            tasks_processed=self.tasks_processed,
            tasks_pending=self._task_queue.qsize(),
            worker_running=bool(self._worker_task and not self._worker_task.done()),
            models_loaded=[current] if current else [],
            current_model=current,
            default_model=self._default_model,
            default_bot=self._default_bot,
            available_models=self._available_models,
            checks=checks,
            database_connected=memory_info.postgres_connected,
            database_host=memory_info.postgres_host,
            database_error=memory_info.postgres_error,
            messages_count=memory_info.messages_count,
            memories_count=memory_info.memories_count,
            pgvector_available=memory_info.pgvector_available,
            embeddings_available=memory_info.embeddings_available,
            mcp_mode=mcp_info.mode,
            mcp_status=mcp_info.status,
            mcp_url=mcp_info.url,
            mcp_http_status=mcp_info.http_status,
        )
