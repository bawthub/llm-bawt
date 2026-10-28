# CPU-only image for the llm-bawt app (FastAPI service + MCP memory server).
#
# TASK-278 Wave B: the app no longer runs ANY in-process GPU/model code.
# Local GGUF/vLLM chat inference (TASK-276) and MiniLM embeddings (TASK-277)
# both live in the `local-model-bridge` service, which keeps the GPU `Dockerfile`.
# So the app installs ONLY its CPU extras (mcp, service, search) and deliberately
# OMITS the heavy GPU/ML stack: torch, sentence-transformers, transformers,
# llama-cpp-python, vllm. Result: a much smaller image where `import torch`,
# `import llama_cpp`, and `import sentence_transformers` all raise
# ModuleNotFoundError — which the app's status/vram/embedding paths already
# degrade-guard for.
#
# `huggingface-hub` (pure-python, no torch) IS kept: gguf_handler / vllm_handler
# guard on `find_spec("huggingface_hub")` for read-only model-catalog metadata
# and GGUF downloads, and we want those to keep working rather than no-op.
FROM python:3.12-slim
ARG SOURCE_SHA=unknown
LABEL org.opencontainers.image.revision="${SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libs only — no CUDA, no build toolchain (all deps ship cp312 wheels):
#   libpq5     psycopg2-binary runtime
#   libgomp1   OpenMP runtime some wheels link against
#   ffmpeg     video poster-frame extraction (media/storage.py)
#   curl       healthcheck + uv installer
#   git        uv VCS metadata
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 ffmpeg curl ca-certificates git tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_HTTP_TIMEOUT=600

WORKDIR /app

# --- Dependency layer (cached independently of source changes) ---
# Resolve + install deps for the CPU extras WITHOUT the project so this layer
# stays cached when only src/ changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
        --extra mcp --extra service --extra search

# --- Project layer ---
# Now copy source and install the project itself (editable). Fast because the
# dependency set above is already cached.
COPY src/ ./src/
COPY server.sh ./
RUN chmod +x server.sh
RUN uv sync --locked --no-dev --extra mcp --extra service --extra search

# Runtime dirs
RUN mkdir -p /app/.run /app/.logs /root/.config/llm-bawt

ENV PATH="/app/.venv/bin:$PATH" \
    TZ=America/New_York \
    PYTHONUNBUFFERED=1

# Set system timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Expose MCP (8001) + service (8642)
EXPOSE 8001 8642

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8642/health || exit 1

# Default command starts both MCP server and LLM service
CMD ["/app/server.sh", "start", "--stdout", "--verbose"]
