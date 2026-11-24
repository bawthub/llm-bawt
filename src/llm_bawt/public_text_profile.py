"""Source omissions for the public text profile (Chat, Setup, memory, Ops).

The public export deletes ``OMITTED_PATHS`` from the committed tree before any
image build copies source. Media generation, the GPU/video listener, spoken
response audio and speech/home-audio/ARR tools are then absent bytes, not
switched-off features: there is no runtime setting that can re-enable them.

Registration points load these modules only when their source is present
(``capability_present``), so the full private tree keeps every capability and
the exported tree assembles without them. Uploads, ``attachment_get``, the
shared ``llm_bawt.media`` storage package and schema stay in both trees.
"""

from __future__ import annotations

import importlib.util

OMITTED_PATHS: tuple[str, ...] = (
    # HTTP: /v1/media/* generation, GPU handoff and spoken response audio.
    "src/llm_bawt/service/routes/media.py",
    "src/llm_bawt/service/routes/media_handoff.py",
    "src/llm_bawt/service/routes/response_audio.py",
    # Generation-only implementations (providers, preflight).
    "src/llm_bawt/media/generation_service.py",
    "src/llm_bawt/media/gpu_preflight.py",
    "src/llm_bawt/media/clients",
    # MCP: generate_image, ARR media library, speech and home audio.
    "src/llm_bawt/mcp_server/media_generation_tools.py",
    "src/llm_bawt/mcp_server/media_tools.py",
    "src/llm_bawt/mcp_server/home_audio_tools.py",
    # Local model bridge: video/GPU listener and worker.
    "src/local_model_bridge/video_server.py",
    "src/local_model_bridge/video_worker.py",
    "src/local_model_bridge/video_models.py",
    "src/local_model_bridge/video_memory.py",
    "src/local_model_bridge/video_residency.py",
    "src/local_model_bridge/gpu_telemetry.py",
)


def capability_present(module: str) -> bool:
    """True when ``module``'s source ships in this tree.

    Only absence of the module itself counts; an import error inside a present
    module still raises at its import site.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


#: Ops catalog rows that only drive voice (Moshi, TTS, the voice backend) or the
#: GPU local-model bridge. The public profile runs the bridge for CPU
#: embeddings only, so it keeps the generic restarts but never seeds these.
OMITTED_OPS_SEED_SLUGS: frozenset[str] = frozenset({
    "bawthub.stop-moshi-stt",
    "bawthub.stop-moshi-tts",
    "bawthub.start-moshi-stt",
    "bawthub.start-moshi-tts",
    "bawthub.restart-tts",
    "bawthub.restart-backend",
    "llm-bawt.reconcile-local-model-bridge-config",
    "llm-bawt.deploy-local-model-bridge-image",
    "llm-bawt.rollback-local-model-bridge-image",
})

#: Voice services the public profile never offers as a restart target.
OMITTED_AUX_SERVICES: frozenset[str] = frozenset({"pocket-tts", "voice-backend"})


def voice_media_present() -> bool:
    """Voice and media generation ship (the media route source is present)."""
    return capability_present("llm_bawt.service.routes.media")


def response_audio_present() -> bool:
    """Spoken reply audio ships (its route source is present)."""
    return capability_present("llm_bawt.service.routes.response_audio")


def home_audio_present() -> bool:
    """Speech/home-audio playback ships (its MCP tool source is present)."""
    return capability_present("llm_bawt.mcp_server.home_audio_tools")
