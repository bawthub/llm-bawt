"""HTTP embedding server for the local-model bridge (TASK-277).

Hosts the MiniLM (``all-MiniLM-L6-v2``) sentence-transformers model that used
to load in-process inside the main app, exposing it as a tiny FastAPI service
on its own port (default 8684). The app's ``memory/embeddings.py`` is now a
thin HTTP client that POSTs here, so the GPU/embedding model lives only in the
bridge — the app process holds no local model at all.

Endpoints:
    POST /embed   {"texts": [str, ...], "model"?: str}
                  -> {"embeddings": [[float, ...], ...], "model": str, "dim": int}
    GET  /health  -> {"ok": bool, "model": str, "dim": int}
                  200 when the model is loaded, 503 until it is (or if it failed).

Environment:
    LOCAL_MODEL_EMBED_MODEL   — sentence-transformers model (default all-MiniLM-L6-v2)
    LOCAL_MODEL_EMBED_DEVICE  — torch device for the model (default "cpu"; MiniLM
                                is tiny, and CPU keeps embeddings independent of
                                GPU/NVML state and off the contended card)
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("local_model_bridge.embed")

EMBED_MODEL_NAME = os.getenv("LOCAL_MODEL_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("LOCAL_MODEL_EMBED_DEVICE", "cpu")
# MiniLM is 384-d. Exposed in /health and used by the app's static dim fallback.
EMBED_DIM = 384

# Module-global cached model (loaded once at startup via load_model()).
_model: Any = None


def pinned_model_dir() -> str | None:
    """Directory of the pinned local-only model, or None for the named/cache path.

    The public text profile ships without the video server and always uses the
    fixed baked directory; it ignores ``LOCAL_MODEL_EMBED_PINNED_DIR`` so it can
    never load other bytes or fall back to a network download. The private tree
    uses the pinned directory only when that variable is set.
    """
    from llm_bawt.public_text_profile import capability_present

    from .pinned_embed_model import DEFAULT_DIR, PINNED_DIR_ENV

    if not capability_present("local_model_bridge.video_server"):
        return DEFAULT_DIR
    return os.getenv(PINNED_DIR_ENV, "").strip() or None


def load_model() -> Any:
    """Load and cache the sentence-transformers model.

    Prefers the on-disk HF cache (``local_files_only=True``) so a warm container
    never round-trips to HuggingFace on startup; falls back to a network load
    only when the cache is cold. Returns the model, or ``None`` if the library
    or weights are unavailable (caller decides how to degrade).
    """
    global _model
    if _model is not None:
        return _model

    pinned_dir = pinned_model_dir()
    if pinned_dir is not None:
        # Before importing sentence-transformers: HF reads these at import.
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})

    # sentence-transformers v5 (multimodal) hard-imports torchcodec — a
    # video/audio codec lib — at package import. When that torchcodec is built
    # for a different CUDA than the container ships (here: torchcodec wants
    # libnvrtc.so.13 / CUDA 13, container has CUDA 12.9), the import raises
    # RuntimeError, which s-t's own guard (`except (ImportError, OSError)`) does
    # NOT catch — so the whole `import sentence_transformers` blows up and /embed
    # dies. torchcodec is irrelevant to MiniLM text embeddings, so if (and only
    # if) it fails to import, neutralize it: `sys.modules[...] = None` makes the
    # downstream `import torchcodec` raise ImportError instead, which the guard
    # DOES catch (falling back to text-only). A healthy torchcodec is untouched.
    # See TASK-629.
    import sys as _sys

    if "torchcodec" not in _sys.modules:
        try:
            import torchcodec  # noqa: F401
        except Exception as e:  # RuntimeError/OSError/ImportError — any load failure
            logger.warning(
                "torchcodec failed to import (%s); neutralizing it so "
                "sentence-transformers loads text-only (torchcodec is unused for "
                "MiniLM embeddings)",
                type(e).__name__,
            )
            _sys.modules["torchcodec"] = None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed in the bridge image; "
            "/embed cannot serve embeddings"
        )
        return None

    # Suppress the per-file HEAD-request spam HF emits during the ETag check.
    noisy = [
        "sentence_transformers",
        "transformers",
        "transformers.modeling_utils",
        "torch.distributed",
        "httpx",
        "httpcore",
        "huggingface_hub",
    ]
    old_levels = {n: logging.getLogger(n).level for n in noisy}
    for n in noisy:
        logging.getLogger(n).setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*layers were not sharded.*")
            warnings.filterwarnings("ignore", category=FutureWarning)
            if pinned_dir is not None:
                # Local-only: verified pinned bytes or no model (never a download).
                from .pinned_embed_model import load as load_pinned

                model = load_pinned(pinned_dir, device=EMBED_DEVICE)
                logger.info("Loaded pinned embedding model from %s (device=%s)", pinned_dir, EMBED_DEVICE)
            else:
                model = _load_named_model(SentenceTransformer)
    except Exception as e:
        logger.error("Failed to load embedding model %s: %s", EMBED_MODEL_NAME, e)
        return None
    finally:
        for n, lvl in old_levels.items():
            logging.getLogger(n).setLevel(lvl)
    return _finish_load(model)


def _load_named_model(SentenceTransformer: Any) -> Any:
    """Private/default path: local HF cache first, network download on a miss."""
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE, local_files_only=True)
        logger.info("Loaded embedding model %s from local cache (device=%s)", EMBED_MODEL_NAME, EMBED_DEVICE)
        return model
    except Exception as cache_miss:
        logger.info(
            "Embedding model %s not cached (%s); downloading from HuggingFace…",
            EMBED_MODEL_NAME,
            cache_miss,
        )
        return SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)


def _finish_load(model: Any) -> Any:
    global _model
    try:
        dim = int(model.get_sentence_embedding_dimension())
        if dim != EMBED_DIM:
            logger.warning(
                "Embedding model %s reports dim=%d but EMBED_DIM=%d; app's static "
                "dimension fallback will be wrong — update EMBED_DIM/DDL",
                EMBED_MODEL_NAME,
                dim,
                EMBED_DIM,
            )
    except Exception:
        pass

    _model = model
    return _model


def embed_ready() -> bool:
    """True once the embedding model is loaded and ``/embed`` can serve.

    Surfaced in the bridge's own health body (:8683) so operators can see embed
    readiness alongside Redis without a second probe.
    """
    return _model is not None


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="local-model-bridge embed", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> JSONResponse:
        # 503 until the model is actually loaded so a slow / hung / failed load
        # is caught by the docker healthcheck (`curl -f`) instead of the port
        # answering 200 while /embed can't serve. NO CPU/keyword fallback — a
        # not-ready embed server reports unhealthy, it does not silently degrade.
        ready = _model is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ok": ready, "model": EMBED_MODEL_NAME, "dim": EMBED_DIM},
        )

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict:
        if req.model and req.model != EMBED_MODEL_NAME:
            # All current callers use MiniLM; serve the configured model and
            # note the mismatch once rather than failing the hot path.
            logger.warning(
                "embed request asked for model=%s but server serves %s",
                req.model,
                EMBED_MODEL_NAME,
            )
        if not req.texts:
            return {"embeddings": [], "model": EMBED_MODEL_NAME, "dim": EMBED_DIM}

        model = load_model()
        if model is None:
            # 503 so the thin client treats it as a transient outage and falls
            # back to None (non-vector search) rather than caching a bad result.
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="embedding model unavailable")

        vectors = model.encode(
            req.texts, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
        return {"embeddings": vectors, "model": EMBED_MODEL_NAME, "dim": EMBED_DIM}

    return app


async def serve_embed(port: int) -> None:
    """Serve the embed API on ``port``, loading the model in the background.

    Run as an asyncio task inside the bridge's existing event loop.

    Ordering matters: we bind and START the HTTP server FIRST, then load the
    model in a thread executor. This is the opposite of the old "await load,
    then serve" flow, which left ``:port`` completely unbound while the model
    loaded — so a slow or (worse) hung load (e.g. torch's CUDA probe wedging on
    a stale NVML handle) made the port answer *connection refused*, indistinguishable
    from the service being absent, and invisible to any healthcheck.

    Now the port is up immediately: ``/health`` answers 503 until the model is
    ready and 200 afterwards, so the docker healthcheck can see "not ready yet"
    and mark the container unhealthy if the load never completes. The load runs
    off the event loop so the server (and the Redis bridge/health tasks) stay
    responsive during a cold load.
    """
    import asyncio

    import uvicorn

    loop = asyncio.get_running_loop()

    config = uvicorn.Config(
        create_app(),
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    logger.info(
        "Embed API listening on :%d (loading model %s, device=%s; /health is 503 until ready)…",
        port,
        EMBED_MODEL_NAME,
        EMBED_DEVICE,
    )

    # Load off the loop. On success /health flips to 200; on failure the model
    # stays None and /health stays 503 — the outage is surfaced, never papered
    # over with a CPU/keyword fallback.
    model = await loop.run_in_executor(None, load_model)
    if model is not None:
        logger.info("Embed model %s ready on :%d", EMBED_MODEL_NAME, port)
    else:
        logger.error(
            "Embed model %s FAILED to load; /embed will 503 and /health reports "
            "unhealthy until resolved",
            EMBED_MODEL_NAME,
        )

    await serve_task
