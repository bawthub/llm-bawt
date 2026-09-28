"""FastAPI-based background service wiring for llm-bawt."""

import asyncio
from contextlib import asynccontextmanager

from ..utils.config import Config
from .background_service import BackgroundService, _ensure_mcp_server
from .dependencies import (
    get_media_asset_store,
    get_media_generation_store,
    get_profile_manager,
    set_service,
)
from .logging import get_service_logger, setup_service_logging
from .partial_text import PartialText

log = get_service_logger(__name__)

# Configuration
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"

def _is_tcp_listening(host: str, port: int) -> bool:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for startup/shutdown."""
    config = Config()

    # Prefer MCP tool-based memory retrieval for llm-service.
    # This ensures memory retrieval happens via MCP tools and can be logged clearly.
    _ensure_mcp_server(config)

    # The normalized catalog is the only model authority. Migration/repair is
    # transactional and intentionally fatal: serving with unresolved bot pins is
    # worse than refusing startup with a concrete error.
    from ..memory.model_catalog_migration import migrate_model_catalog
    from ..model_catalog import ModelCatalogStore
    from ..utils.db import get_shared_engine

    model_engine = get_shared_engine(config)
    if model_engine is None:
        raise RuntimeError("Normalized model catalog database unavailable")
    catalog_result = migrate_model_catalog(model_engine)
    log.info("normalized model catalog ready: %s", catalog_result)
    normalized_catalog = ModelCatalogStore(model_engine).load()
    config.install_model_catalog(normalized_catalog)
    log.info("Loaded %d normalized model endpoints", len(normalized_catalog))

    # Ensure media_generations table exists
    try:
        get_media_generation_store(config)
        log.info("media_generations table ready")
    except Exception as e:
        log.warning("Failed to initialise media_generations table: %s", e)

    # TASK-222: ensure media_assets table exists. Separate from
    # media_generations (different concern — that one tracks long-running
    # generation jobs; this one is the canonical registry for normalized
    # chat-upload / tool-output / agent-attachment blobs).
    try:
        get_media_asset_store(config)
        log.info("media_assets table ready")
    except Exception as e:
        log.warning("Failed to initialise media_assets table: %s", e)

    # TASK-216: warm the sentence-transformer used by the animation
    # classifier (and the memory subsystem) in the background so the first
    # voice response after a restart doesn't pay a ~6s lazy-load. Fire it
    # off-thread — the model load is CPU-bound and ~6s, and we don't want
    # to block the event loop or delay the rest of startup.
    def _warm_embedding_model() -> None:
        try:
            from ..memory.embeddings import generate_embedding
            generate_embedding("warmup")
            log.info("🔥 embedding model warm")
        except Exception as warm_err:
            log.debug("embedding warmup skipped: %s", warm_err)

    import threading as _threading
    _threading.Thread(target=_warm_embedding_model, daemon=True, name="embedding-warmup").start()

    service = BackgroundService(config)
    set_service(service)
    service.start_worker()

    # Start job scheduler if enabled
    scheduler = None
    if config.SCHEDULER_ENABLED:
        from .scheduler import JobScheduler, create_scheduler_tables, init_default_jobs

        # Get engine from profile manager (reuse existing connection)
        pm = get_profile_manager(config)
        create_scheduler_tables(pm.engine)
        init_default_jobs(pm.engine, config)

        from .prompt_delivery import PromptDeliveryWorker
        prompt_worker = PromptDeliveryWorker(pm.engine, service)
        scheduler = JobScheduler(
            engine=pm.engine,
            task_processor=service,
            check_interval=config.SCHEDULER_CHECK_INTERVAL_SECONDS,
            prompt_sweep=prompt_worker.sweep,
        )
        # Start below, after the shared transport and durable dispatcher exist.

    # Start the shared Redis agent transport for every bridge harness. OpenClaw,
    # Claude Code, and Codex all depend on this subscriber; OpenClaw-specific
    # session mapping remains optional inside the shared setup.
    redis_subscriber = None
    history_drain_task = None
    if config.REDIS_URL:
        try:
            # Build session_key -> bot_id mapping from openclaw bots
            from ..bots import BotManager
            session_to_bot: dict[str, str] = {}
            bot_mgr = BotManager(config)

            for bot in bot_mgr.list_bots():
                if bot.agent_backend != "openclaw":
                    continue
                bc = bot.agent_backend_config or {}
                sk = bc.get("session_key")
                if sk:
                    session_to_bot[sk] = bot.slug

            if session_to_bot:
                log.info("OpenClaw session->bot mapping: %s", session_to_bot)

            from agent_bridge.subscriber import RedisSubscriber
            from ..agent_backends.agent_bridge import set_agent_subscriber

            redis_subscriber = RedisSubscriber(config.REDIS_URL)
            await redis_subscriber.connect()
            service._redis_subscriber = redis_subscriber

            # Make subscriber available to every Redis-backed agent backend.
            set_agent_subscriber(redis_subscriber)

            # DEACTIVATED: passive history drain causes duplicates with
            # finalize_response().  The active chat.send path persists via
            # finalize_response() and does not need this consumer.

            # Start background task to clean up stale consumer groups
            async def _cleanup_stale_groups():
                """Periodically destroy idle ui:* consumer groups."""
                while True:
                    await asyncio.sleep(300)
                    try:
                        streams = await redis_subscriber.list_unified_streams()
                        total = 0
                        for stream_key in streams:
                            total += await redis_subscriber.cleanup_stale_groups(stream_key)
                        if total:
                            log.info("Cleaned up %d stale consumer groups", total)
                    except Exception:
                        log.debug("Stale group cleanup error", exc_info=True)

            service._group_cleanup_task = asyncio.create_task(_cleanup_stale_groups())

            # Start persistence consumer for tool events → Postgres.
            # TASK-286: also accumulates assistant text deltas per turn so a
            # COLD reload (and a turn that completes after the client
            # disconnects) recovers the response TEXT, not just tool calls.
            _partial_text: dict[str, str] = {}
            _partial_text_chunks: dict[str, PartialText] = {}
            # TASK-360 (P4): accumulate per-turn reasoning ("thinking") the
            # same way, flushed to turn_logs.reasoning so a COLD reload
            # mid-turn recovers already-produced reasoning, not just text.
            _partial_reasoning: dict[str, str] = {}

            def _tool_event_sink(event_data: dict) -> None:
                """Persist tool_start/tool_end + text_delta + reasoning_delta events to Postgres."""
                store = service._turn_log_store
                _type = event_data.get("_type")
                if _type == "reasoning_delta":
                    turn_id = event_data.get("turn_id")
                    if not turn_id:
                        return
                    delta = event_data.get("delta", "") or ""
                    rbuf = _partial_reasoning.get(turn_id, "") + delta
                    _partial_reasoning[turn_id] = rbuf
                    # Pass the current text buffer too so the write never
                    # clobbers already-flushed partial text (reasoning
                    # usually precedes text, so the buffer is often "").
                    store.update_partial_response(
                        turn_id=turn_id,
                        response_text=_partial_text.get(turn_id, ""),
                        reasoning=rbuf,
                    )
                    return
                if _type == "text_delta":
                    turn_id = event_data.get("turn_id")
                    if not turn_id:
                        return
                    delta = event_data.get("delta", "") or ""
                    offset = event_data.get("text_offset")
                    buf = _partial_text_chunks.setdefault(turn_id, PartialText()).add(delta, offset)
                    _partial_text[turn_id] = buf
                    store.update_partial_response(turn_id=turn_id, response_text=buf)
                    return
                if _type == "turn_complete":
                    tid = event_data.get("turn_id")
                    if tid:
                        _partial_text.pop(tid, None)
                        _partial_text_chunks.pop(tid, None)
                        _partial_reasoning.pop(tid, None)
                    return
                event_type = event_data.get("event", "")
                # Active chat turns persist synchronously through
                # ToolEventCoordinator before public publication. The drain
                # remains the compatibility path for passive/legacy producers.
                if event_data.get("_tool_persisted"):
                    return
                if event_type == "tool_start":
                    store.save_tool_call(
                        turn_id=event_data.get("turn_id"),
                        bot_id=event_data.get("bot_id"),
                        user_id=event_data.get("user_id"),
                        call_id=event_data.get("call_id"),
                        tool_name=event_data.get("tool_name", "unknown"),
                        arguments=event_data.get("arguments"),
                        iteration=event_data.get("iteration", 1),
                        started_at=event_data.get("ts"),
                        text_offset=event_data.get("text_offset"),
                        tool_use_id=event_data.get("tool_use_id"),
                        parent_tool_use_id=event_data.get("parent_tool_use_id"),
                    )
                elif event_type == "tool_end":
                    call_id = event_data.get("call_id")
                    if call_id:
                        store.update_tool_call_result(
                            call_id=call_id,
                            result=event_data.get("result", ""),
                            ended_at=event_data.get("ts"),
                            is_error=event_data.get("is_error"),
                        )
                    else:
                        # No call_id — save as complete record
                        store.save_tool_call(
                            turn_id=event_data.get("turn_id"),
                            bot_id=event_data.get("bot_id"),
                            user_id=event_data.get("user_id"),
                            call_id=None,
                            tool_name=event_data.get("tool_name", "unknown"),
                            result=event_data.get("result", ""),
                            iteration=event_data.get("iteration", 1),
                            ended_at=event_data.get("ts"),
                            is_error=event_data.get("is_error"),
                            tool_use_id=event_data.get("tool_use_id"),
                            parent_tool_use_id=event_data.get("parent_tool_use_id"),
                        )

            service._tool_persist_task = asyncio.create_task(
                redis_subscriber.drain_tool_events(_tool_event_sink)
            )

            log.info(
                "Shared agent Redis subscriber started (redis=%s, openclaw_sessions=%s)",
                config.REDIS_URL,
                list(session_to_bot.keys()),
            )
        except Exception:
            log.exception("Failed to start shared agent Redis transport")
            redis_subscriber = None

    # Durable callbacks target agent harnesses and require the shared Redis
    # transport. Start recovery only after subscriber initialization; when Redis
    # is unavailable rows remain QUEUED and no retry attempt is consumed.
    from .inter_bot_dispatcher import InterBotDeliveryDispatcher
    service._inter_bot_dispatcher = InterBotDeliveryDispatcher(service)
    service._inter_bot_dispatcher.start()
    if scheduler is not None:
        await scheduler.start()
        log.info("Scheduler started (maintenance interval=%ss; prompt sweep=5s)",
                 config.SCHEDULER_CHECK_INTERVAL_SECONDS)

    # TASK-861: recover stranded approved MCP claims (uncertain generic calls
    # are not replayed), repair old result/outbox gaps, and deliver MCP results
    # plus server-owned harness decisions through the fenced durable outbox.
    from .approval_continuations import run_mcp_continuation_outbox
    service._mcp_continuation_task = asyncio.create_task(
        run_mcp_continuation_outbox(service, service._tool_approval_policy_store)
    )

    # Log startup with rich formatting
    log.startup(
        version=SERVICE_VERSION,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        models=service._available_models,
        default_model=service._default_model,
    )

    log.info(
        "Memory mode: %s (%s)",
        "mcp" if config.MCP_SERVER_URL else "embedded",
        config.MCP_SERVER_URL or "",
    )

    # TASK-635: the app is the SOLE refresher of the app-owned Claude
    # credential (bridge + usage are read-only consumers). Keep it fresh
    # proactively so readers never see a lapsed token, even when idle.
    from .usage.claude_oauth import proactive_refresh_loop
    service._claude_refresh_task = asyncio.create_task(proactive_refresh_loop())

    # TASK-636 Phase 2: same pattern for the ChatGPT/codex OAuth bundle.
    from .usage.codex_oauth import proactive_refresh_loop as _codex_loop
    service._codex_refresh_task = asyncio.create_task(_codex_loop())

    # GPU transition intent is durable. An app restart cannot replay an
    # unconfirmed stop/start or assume that the old owner still holds the GPU.
    from ..media.gpu_handoff_store import GpuHandoffStore
    gpu_handoff = await asyncio.to_thread(GpuHandoffStore, model_engine)
    gpu_state = await asyncio.to_thread(gpu_handoff.mark_interrupted)
    if gpu_state.phase == "recovery_required":
        log.warning("GPU handoff requires operator reconciliation: %s", gpu_state.last_error)

    # Durable speech playback belongs to the app, not a short-lived MCP call.
    from ..integrations.home_audio_store import HomeAudioStore
    from ..integrations.home_audio_worker import HomeAudioWorker
    home_audio_worker = HomeAudioWorker(config, HomeAudioStore(model_engine))
    home_audio_worker.start()

    try:
        yield
    finally:
        await home_audio_worker.stop()
        # DEACTIVATED: history drain task (see above)
        # if history_drain_task:
        #     history_drain_task.cancel()
        #     try:
        #         await history_drain_task
        #     except (asyncio.CancelledError, Exception):
        #         pass
        # Stop the producer before its consumer/Redis transport.
        if scheduler:
            await scheduler.stop()
        # Stop durable claims first while Redis is still available. Cancellation
        # leaves the claim leased; the next singleton dispatcher recovers it
        # after the short lease rather than counting orderly shutdown as failure.
        if service._inter_bot_dispatcher is not None:
            await service._inter_bot_dispatcher.stop()

        # Cancel background tasks
        for task_attr in (
            "_group_cleanup_task",
            "_tool_persist_task",
            "_mcp_continuation_task",
            "_claude_refresh_task",
            "_codex_refresh_task",
        ):
            task = getattr(service, task_attr, None)
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if redis_subscriber:
            from ..agent_backends.agent_bridge import set_agent_subscriber
            set_agent_subscriber(None)
            await redis_subscriber.close()
        await service.shutdown()
        set_service(None)


# Create FastAPI app
try:
    from fastapi import FastAPI

    from .routes import all_routers

    app = FastAPI(
        title="llm-bawt API",
        description="OpenAI-compatible API with integrated memory system",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    for router in all_routers:
        app.include_router(router)

except ImportError:
    # FastAPI not installed - create stub
    app = None
    log.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _find_service_pid(port: int) -> int | None:
    """Find the PID of a process listening on the given port."""
    import subprocess
    try:
        # Use lsof to find the process listening on the port
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs (parent/child), get the first one
            pids = result.stdout.strip().split("\n")
            return int(pids[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def _is_service_running(host: str, port: int) -> bool:
    """Check if the service is already running by attempting to connect."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            # Use 127.0.0.1 for 0.0.0.0 since we can't connect to 0.0.0.0
            check_host = "127.0.0.1" if host == "0.0.0.0" else host
            result = sock.connect_ex((check_host, port))
            return result == 0
    except (OSError, socket.error):
        return False


def _kill_service(port: int) -> bool:
    """Kill the service running on the given port. Returns True if successful."""
    import signal
    import os
    
    pid = _find_service_pid(port)
    if pid is None:
        return False
    
    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait briefly for the process to terminate
        import time
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)  # Check if process still exists
            except OSError:
                return True  # Process terminated
        
        # If still running, send SIGKILL
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
        return True
    except OSError:
        return False


def main():
    """Entry point for the background service."""
    import argparse
    
    # Load config for defaults
    config = Config()
    
    parser = argparse.ArgumentParser(description="llm-bawt background service")
    parser.add_argument("--host", default=config.SERVICE_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT, help="Port to listen on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional detail (payloads, timing)")
    parser.add_argument("--debug", action="store_true", help="Enable low-level DEBUG messages (unformatted)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--restart", action="store_true", help="Kill existing service and start a new one")
    parser.add_argument("--stop", action="store_true", help="Stop the running service and exit")
    args = parser.parse_args()
    
    # Setup logging with the new rich-formatted logger
    # Note: --verbose enables payload logging, --debug enables low-level DEBUG
    setup_service_logging(verbose=args.verbose, debug=args.debug)
    
    # Also update config.VERBOSE so BackgroundService can use it
    config.VERBOSE = args.verbose
    
    # Handle --stop: kill the service and exit
    if args.stop:
        if _is_service_running(args.host, args.port):
            print(f"Stopping service on port {args.port}...")
            if _kill_service(args.port):
                print("Service stopped.")
                return 0
            else:
                print("Failed to stop service.")
                return 1
        else:
            print(f"No service running on port {args.port}.")
            return 0
    
    # Check if service is already running
    if _is_service_running(args.host, args.port):
        if args.restart:
            print(f"Restarting service on port {args.port}...")
            if not _kill_service(args.port):
                print("Warning: Could not kill existing service, attempting to start anyway...")
            # Brief pause to ensure port is released
            import time
            time.sleep(0.5)
        else:
            print(f"Service is already running on port {args.port}.")
            print("Use --restart to restart the service, or --stop to stop it.")
            return 0
    
    if app is None:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        import uvicorn
        
        # Configure uvicorn log level
        # When using our rich logging, set uvicorn to warning to reduce noise
        uvicorn_log_level = "debug" if args.debug else "warning"
        
        # Exclude generated/data files from reload watching to prevent feedback loops
        reload_excludes = [
            "__pycache__", "*.pyc", ".git",
            ".logs", ".run", "models",
            "*.log", "*.pid",
        ] if args.reload else None
        
        uvicorn.run(
            "llm_bawt.service.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            reload_excludes=reload_excludes,
            log_level=uvicorn_log_level,
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return 1
