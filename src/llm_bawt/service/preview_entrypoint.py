"""Opt-in chat entrypoint for an isolated backend preview process.

This process never imports the production app or executes its lifespan. Its DB,
Redis, bridge and network boundaries are provided by the preview deployment;
startup refuses to serve chat unless those dependencies are present and seeded.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .schemas import ChatCompletionRequest


class PreviewSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str
    user_id: str
    title: str | None = None


_REQUIRED_ENV = (
    "LLM_BAWT_POSTGRES_HOST",
    "LLM_BAWT_POSTGRES_DATABASE",
    "LLM_BAWT_POSTGRES_USER",
    "LLM_BAWT_POSTGRES_PASSWORD_FILE",
    "LLM_BAWT_REDIS_URL",
    "LLM_BAWT_PREVIEW_AGENT_BACKEND",
)
_ALLOWED_ENV = frozenset(_REQUIRED_ENV) | {
    "LLM_BAWT_PREVIEW_ENABLED",
    "LLM_BAWT_POSTGRES_PORT",
    "LLM_BAWT_SERVICE_HOST",
    "LLM_BAWT_SERVICE_PORT",
    # Explicitly permitted non-credential container runtime settings. Unknown
    # variables fail startup rather than implicitly granting a new secret.
    "DEBIAN_FRONTEND", "GPG_KEY", "HOME", "HOSTNAME", "LANG", "LC_ALL",
    "LC_CTYPE", "LOGNAME", "PATH", "PWD", "PYTHONDONTWRITEBYTECODE",
    "PYTHONPYCACHEPREFIX", "PYTHONUNBUFFERED", "PYTHON_VERSION",
    "PYTHON_SHA256", "SHLVL", "TERM", "TMPDIR", "TZ", "USER",
    "UV_HTTP_TIMEOUT", "VIRTUAL_ENV", "_",
}


def _privileged_paths() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".ssh",
        home / ".claude",
        home / ".codex",
        home / ".config/llm-bawt/secret.key",
        home / ".config/llm-bawt/claude-usage/.credentials.json",
        home / ".config/llm-bawt/.env",
        Path.cwd() / ".env",
        Path("/run/llm-bawt-bridge-tokens"),
    )


_RUN_DIRECTORIES = (Path("/run"), Path("/var/run"))


def _privileged_sockets() -> tuple[Path, ...]:
    """A preview cannot inherit any host control socket under /run."""
    try:
        return tuple(
            path for directory in _RUN_DIRECTORIES
            if directory.is_dir()
            for path in directory.iterdir()
            if path.is_socket()
        )
    except OSError as exc:
        raise RuntimeError("preview cannot inspect host control sockets") from exc


def _check_environment() -> Path:
    if os.environ.get("LLM_BAWT_PREVIEW_ENABLED") != "1":
        raise RuntimeError("explicit preview role required")
    missing = [key for key in _REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"preview requires isolated configuration: {', '.join(missing)}")
    forbidden = [key for key in os.environ if key not in _ALLOWED_ENV]
    if forbidden:
        raise RuntimeError(f"preview refuses production-capable environment: {', '.join(forbidden)}")
    mounted = [str(path) for path in _privileged_paths() if path.exists()]
    mounted.extend(str(path) for path in _privileged_sockets())
    if mounted:
        raise RuntimeError(f"preview refuses privileged mounts: {', '.join(mounted)}")
    secret_file = Path(os.environ["LLM_BAWT_POSTGRES_PASSWORD_FILE"])
    if not secret_file.is_file() or secret_file.is_symlink():
        raise RuntimeError("preview database secret file is unavailable or linked")
    return secret_file


async def _start_preview_runtime(app: FastAPI, config) -> tuple[object, object]:
    """Read the pre-seeded catalog and connect only to the isolated transport."""
    from sqlalchemy import text

    from ..bots import BotManager
    from ..model_catalog import ModelCatalogStore
    from ..utils.db import get_shared_engine
    from .background_service import BackgroundService
    from .dependencies import set_service
    from ..agent_backends.agent_bridge import set_agent_subscriber
    from agent_bridge.subscriber import RedisSubscriber

    engine = get_shared_engine(config)
    if engine is None:
        raise RuntimeError("preview database unavailable")
    with engine.connect() as conn:
        database, role = conn.execute(text("SELECT current_database(), current_user")).one()
        if database != config.POSTGRES_DATABASE or role != config.POSTGRES_USER:
            raise RuntimeError("preview database identity mismatch")
        if conn.execute(text("SELECT 1 FROM preview_seed_meta LIMIT 1")).scalar() != 1:
            raise RuntimeError("preview seed is not complete")
    config.install_model_catalog(ModelCatalogStore(engine).load())
    backend = os.environ["LLM_BAWT_PREVIEW_AGENT_BACKEND"]
    bots = BotManager(config).list_bots()
    if not any(bot.agent_backend == backend and bot.endpoint_id for bot in bots):
        raise RuntimeError("no seeded bot matches the configured preview bridge")

    subscriber = RedisSubscriber(config.REDIS_URL)
    await subscriber.connect()
    if not subscriber.connected:
        await subscriber.close()
        raise RuntimeError("isolated preview Redis is unavailable")
    service = None
    try:
        service = BackgroundService(config)
        service._redis_subscriber = subscriber
        set_agent_subscriber(subscriber)
        set_service(service)
        app.state.preview_backend = backend
        app.state.preview_ready = True
        return service, subscriber
    except BaseException:
        set_service(None)
        set_agent_subscriber(None)
        await subscriber.close()
        if service is not None:
            await service.shutdown()
        raise


@asynccontextmanager
async def _preview_lifespan(app: FastAPI):
    app.state.preview_ready = False
    secret_file = _check_environment()
    # Existing bot/registry code constructs Config() internally. Supply ONLY the
    # preview DB password to this process, and disable Config's implicit repo .env.
    from ..utils.config import Config

    previous_env_file = Config.model_config.get("env_file")
    Config.model_config["env_file"] = None
    try:
        password = secret_file.read_text(encoding="utf-8").strip()
        if not password:
            raise RuntimeError("preview database secret is empty")
        os.environ["LLM_BAWT_POSTGRES_PASSWORD"] = password
        config = Config(_env_file=None, MCP_SERVER_URL=None, SCHEDULER_ENABLED=False)
        service, subscriber = await _start_preview_runtime(app, config)
        try:
            yield
        finally:
            app.state.preview_ready = False
            from .dependencies import set_service
            from ..agent_backends.agent_bridge import set_agent_subscriber

            set_service(None)
            set_agent_subscriber(None)
            await subscriber.close()
            await service.shutdown()
    finally:
        os.environ.pop("LLM_BAWT_POSTGRES_PASSWORD", None)
        Config.model_config["env_file"] = previous_env_file


def create_preview_app() -> FastAPI:
    """Register only selected routes: never mount the production router/MCP."""
    app = FastAPI(
        title="llm-bawt preview",
        lifespan=_preview_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.preview_ready = False

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = app.state.preview_ready
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"service": "preview", "state": "ready" if ready else "not_configured", "preview": True},
        )

    def require_preview(http_request: Request) -> None:
        if http_request.headers.get("X-BawtHub-Preview") != "1":
            raise HTTPException(status_code=403, detail="explicit preview marker required")
        if not app.state.preview_ready:
            raise HTTPException(status_code=503, detail="preview backend is unavailable")

    def require_bot(bot_id: str) -> None:
        from ..bots import BotManager
        from .dependencies import get_service

        bot = BotManager(get_service().config).get_bot(bot_id)
        if bot is None or bot.agent_backend != app.state.preview_backend or not bot.endpoint_id:
            raise HTTPException(status_code=503, detail="selected bot has no isolated preview bridge")

    @app.post("/v1/sessions")
    async def preview_session(body: PreviewSessionRequest, http_request: Request):
        require_preview(http_request)
        body.bot_id = body.bot_id.strip()
        body.user_id = body.user_id.strip()
        if not body.bot_id or not body.user_id:
            raise HTTPException(status_code=400, detail="preview requires bot and user namespaces")
        require_bot(body.bot_id)
        from ..mcp_server.storage import get_storage
        from ..working_memory.session_state import mode_from_flag

        title = body.title.strip() if body.title is not None else None
        if title is not None and not title:
            raise HTTPException(status_code=422, detail="Preview title must be nonempty")
        storage = get_storage()
        session_id = await storage.rotate_session(
            bot_id=body.bot_id, user_id=body.user_id,
            working_memory_mode=mode_from_flag(False),
        )
        if title:
            await storage.update_session_metadata(
                session_id, {"title": title, "title_source": "user"}, bot_id=body.bot_id,
            )
        return {"session_id": session_id, "bot_id": body.bot_id,
                "user_id": body.user_id, "status": "active", "rotated": True}

    @app.post("/v1/chat/completions")
    async def preview_chat(request: ChatCompletionRequest, http_request: Request):
        require_preview(http_request)
        raw = await http_request.json()
        if isinstance(raw, dict) and any(
            key.startswith("inter_bot_") or key in {"parent_turn_id", "answered_question_id", "continuation_payload"}
            for key in raw
        ):
            raise HTTPException(status_code=403, detail="delivery/continuation authority is unavailable in preview")
        permitted = {"bot_id", "user", "session_id", "messages", "stream",
                     "user_message_id", "assistant_message_id"}
        if not isinstance(raw, dict) or set(raw) - permitted:
            raise HTTPException(status_code=403, detail="Preview accepts only isolated text chat fields")
        messages = raw.get("messages")
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict)
                or set(messages[0]) != {"role", "content"}
                or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)
                or not messages[0]["content"].strip()):
            raise HTTPException(status_code=400, detail="Preview accepts one text user message")
        request.bot_id = request.bot_id.strip() if request.bot_id else None
        request.user = request.user.strip() if request.user else None
        if not request.bot_id:
            raise HTTPException(status_code=400, detail="preview requires a selected bot")
        if not request.user:
            raise HTTPException(status_code=400, detail="preview requires a user namespace")
        if http_request.headers.get("X-LLM-Bawt-Inter-Bot-Sender") or any((
            request.inter_bot_delivery_id, request.inter_bot_turn_id,
            request.inter_bot_bridge_request_id, request.inter_bot_claim_token,
            request.inter_bot_session_policy, request.parent_turn_id,
            request.answered_question_id, request.continuation_payload,
        )) or request.inter_bot_reset_only or request.inter_bot_typed_follow_up:
            raise HTTPException(status_code=403, detail="delivery/continuation authority is unavailable in preview")
        require_bot(request.bot_id)
        if request.session_id:
            from ..mcp_server.storage import get_storage

            row = await get_storage().get_session(request.session_id, bot_id=request.bot_id)
            if not row or row.get("bot_id") != request.bot_id or row.get("user_id") != request.user:
                raise HTTPException(status_code=404, detail="Preview session not found")
            if row.get("status") == "deleted":
                raise HTTPException(status_code=410, detail="Preview session deleted")
        from .chat_response_adapter import completion_response, completion_stream
        from .dependencies import get_service

        request.augment_memory = False
        request.extract_memory = False
        service = get_service()
        if request.stream:
            admission_comment = http_request.headers.get("X-LLM-Bawt-Admission-Events") == "comment"
            extensions = (
                "comment" if http_request.headers.get("X-LLM-Bawt-Extension-Events") == "comment"
                else "data" if admission_comment else "none"
            )
            from fastapi.responses import StreamingResponse

            return StreamingResponse(
                completion_stream(service, request, admission_comment=admission_comment,
                                  extensions=extensions),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        from ..request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity

        try:
            return await completion_response(service, request)
        except (IdentityConflict, AdmissionUnavailable, InvalidIdentity) as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    from ..preview.chat_routes import register_chat_routes

    register_chat_routes(app, require_preview, require_bot)
    return app


app = create_preview_app()
