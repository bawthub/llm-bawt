# Multi-stage build for llm-bawt
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    curl \
    openssh-client \
    build-essential \
    cmake \
    ninja-build \
    libpq-dev \
    git \
    ffmpeg \
    cuda-compat-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy ONLY files needed for dependency installation first (for layer caching)
# This ensures source code changes don't invalidate the venv cache
COPY pyproject.toml install.sh ./
RUN chmod +x install.sh

# Create virtual environment and install dependencies using install.sh
# install.sh handles CUDA-aware llama-cpp-python installation when available
ARG WITH_CUDA=true
ARG CUDA_ARCHS="120"
ENV CUDA_ARCHS=$CUDA_ARCHS
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
ENV LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64/stubs
RUN echo "/usr/local/cuda/compat" > /etc/ld.so.conf.d/cuda-compat.conf && ldconfig
ENV UV_HTTP_TIMEOUT=600
# Add CUDA stubs for linking during build (no GPU available in docker build)
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
# Install dependencies only (without project) - this layer is cached
RUN ./install.sh --dev --deps-only

# Install llama-cpp-python with CUDA (separate cached layer)
# Install from GitHub to get latest llama.cpp with newest arch support (e.g. mistral3)
ARG LLAMA_CPP_CACHE_BUST=3
RUN if [ "$WITH_CUDA" = "true" ]; then \
        CUDA_ARCHS="${CUDA_ARCHS:-120}" \
        CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
        uv pip install "llama-cpp-python @ git+https://github.com/abetlen/llama-cpp-python.git" --reinstall; \
    else \
        uv pip install "llama-cpp-python @ git+https://github.com/abetlen/llama-cpp-python.git"; \
    fi

# Install vLLM with CUDA (separate cached layer)
RUN if [ "$WITH_CUDA" = "true" ]; then \
        uv pip install vllm; \
    else \
        echo "Skipping vLLM install (requires CUDA)"; \
    fi

# Wan 2.2 video pipeline. The worker imports this lazily; chat and embeddings do
# not load video weights. Keep the dependency in this GPU image, not the app.
RUN uv pip install 'diffusers==0.40.0' 'ftfy>=6.0' 'imageio>=2.30' 'imageio-ffmpeg>=0.5.1'

# NOW copy source code and other files needed for project install
# Changes to src/ won't invalidate the venv/deps layer above
COPY src/ ./src/
COPY server.sh ./
COPY docker/agent-bridge-constraints.txt ./docker/
RUN chmod +x server.sh

# Install the project itself (fast since deps + llama-cpp + vllm are cached)
RUN uv sync --inexact --extra mcp --extra service --extra search --extra memory --extra huggingface --extra vllm

# Runtime stage
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
# NOTE: gcc/g++ are required at RUNTIME by Triton/FlashInfer, which JIT-compile
# CUDA kernels on first use. Gemma 4 on vLLM forces the TRITON_ATTN backend
# (heterogeneous head dims), so without a C compiler the model fails to load
# with "Failed to find C compiler". The CUDA toolkit (nvcc/ptxas) comes from the
# pip nvidia-* wheels; gcc is the missing host compiler.
RUN apt-get update && apt-get install -y \
    python3 \
    libpq5 \
    libgomp1 \
    curl \
    openssh-client \
    tzdata \
    gcc \
    g++ \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy virtual environment and source from base
COPY --from=base /app/.venv /app/.venv
COPY --from=base /app/src /app/src
COPY --from=base /app/pyproject.toml /app/pyproject.toml
COPY --from=base /app/server.sh /app/server.sh
COPY --from=base /app/docker/agent-bridge-constraints.txt /app/docker/agent-bridge-constraints.txt

# Create necessary directories
RUN mkdir -p /app/.run /app/.logs /root/.config/llm-bawt

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    TZ=America/New_York \
    PYTHONUNBUFFERED=1

# Set system timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Expose ports
EXPOSE 8001 8642

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8642/health || exit 1

# Default command starts both MCP server and LLM service
CMD ["/app/server.sh", "start", "--stdout", "--verbose"]
