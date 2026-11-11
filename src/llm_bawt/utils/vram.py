"""VRAM detection and context window auto-sizing.

Detects available GPU VRAM and calculates the maximum safe context window
for a given model based on available resources.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VRAMInfo:
    """GPU VRAM information."""
    total_bytes: int
    free_bytes: int
    gpu_name: str = "Unknown"
    detection_method: str = "none"

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    def __str__(self) -> str:
        return f"{self.gpu_name}: {self.total_gb:.1f}GB total, {self.free_gb:.1f}GB free ({self.detection_method})"


@dataclass
class ContextSizingResult:
    """Result of auto-sizing the context window."""
    context_window: int
    source: str  # e.g., "auto-vram", "model-cap", "native-limit", "global-config", "default"
    vram_info: VRAMInfo | None = None
    model_file_size_gb: float = 0.0
    estimated_kv_budget_gb: float = 0.0

    def __str__(self) -> str:
        return f"context_window={self.context_window} (source: {self.source})"


def detect_vram() -> VRAMInfo | None:
    """Detect available GPU VRAM.

    Tries torch.cuda first (most accurate), falls back to nvidia-smi parsing.
    Returns None if no GPU is detected.
    """
    info = _detect_vram_torch()
    if info:
        return info

    info = _detect_vram_nvidia_smi()
    if info:
        return info

    logger.debug("No GPU VRAM detected via torch.cuda or nvidia-smi")
    return None


def _detect_vram_torch() -> VRAMInfo | None:
    """Detect VRAM using PyTorch CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.debug("torch.cuda is not available")
            return None

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        # Use memory_reserved to get a better picture of what's actually free
        free = total - torch.cuda.memory_reserved(device)

        return VRAMInfo(
            total_bytes=total,
            free_bytes=free,
            gpu_name=props.name,
            detection_method="torch.cuda",
        )
    except ImportError:
        logger.debug("torch not available for VRAM detection")
        return None
    except Exception as e:
        logger.debug(f"torch.cuda VRAM detection failed: {e}")
        return None


def _detect_vram_nvidia_smi() -> VRAMInfo | None:
    """Detect VRAM by parsing nvidia-smi output."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug(f"nvidia-smi failed with return code {result.returncode}")
            return None

        # Parse first GPU line: "NVIDIA GeForce RTX 5090, 32768, 30000"
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            logger.debug(f"Unexpected nvidia-smi output format: {line}")
            return None

        gpu_name = parts[0]
        total_mib = int(parts[1])
        free_mib = int(parts[2])

        return VRAMInfo(
            total_bytes=total_mib * 1024 * 1024,
            free_bytes=free_mib * 1024 * 1024,
            gpu_name=gpu_name,
            detection_method="nvidia-smi",
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found")
        return None
    except Exception as e:
        logger.debug(f"nvidia-smi VRAM detection failed: {e}")
        return None


# ── KV Cache Estimation ──────────────────────────────────────────────────────

# Approximate total VRAM cost per token of context (not just KV cache).
# These include KV cache + compute buffers + attention scratch space.
# Based on empirical measurements with llama-cpp-python + flash attention.
_KV_BYTES_PER_TOKEN_ESTIMATES = {
    # (min_params_billions, max_params_billions): bytes_per_token
    (0, 4): 512,         # ~0.5 KB/token (1-3B models)
    (4, 10): 1024,       # ~1 KB/token (7-8B models)
    (10, 20): 2560,      # ~2.5 KB/token (13-14B models)
    (20, 30): 3584,      # ~3.5 KB/token (24-27B models)
    (30, 50): 5120,      # ~5 KB/token (32-34B models)
    (50, 80): 7168,      # ~7 KB/token (65-70B models)
    (80, 200): 10240,    # ~10 KB/token (72B+ models)
}


def _estimate_kv_bytes_per_token(model_size_hint: str | None = None, file_size_bytes: int = 0) -> int:
    """Estimate KV cache bytes per token based on model size.

    Args:
        model_size_hint: Optional string like "32B", "8B", "7b" from model name
        file_size_bytes: Model file size in bytes (used to estimate param count if no hint)

    Returns:
        Estimated bytes per token for KV cache
    """
    param_billions = 0.0

    # Try to extract from hint string (e.g., "32B", "8b", "7B")
    if model_size_hint:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_size_hint)
        if match:
            param_billions = float(match.group(1))

    # Estimate from file size if no hint
    # Q4_K_M quantization: ~0.56 bytes per parameter
    # Q8_0 quantization: ~1.0 bytes per parameter
    # Use ~0.7 as a middle ground for estimating from file size
    if param_billions == 0 and file_size_bytes > 0:
        param_billions = (file_size_bytes / 0.7) / 1e9

    if param_billions == 0:
        # Default to 32B estimate if we can't determine
        return 2048

    for (min_p, max_p), kv_bytes in _KV_BYTES_PER_TOKEN_ESTIMATES.items():
        if min_p <= param_billions < max_p:
            return kv_bytes

    # Larger than anything in the table
    return 4096


def estimate_max_context_from_vram(
    vram_info: VRAMInfo,
    model_file_size_bytes: int,
    model_size_hint: str | None = None,
    safety_margin_gb: float = 1.5,
) -> int:
    """Estimate maximum safe context window based on available VRAM.

    Args:
        vram_info: Detected VRAM info
        model_file_size_bytes: Size of the model file in bytes (≈ weight VRAM for GGUF)
        model_size_hint: Optional string like "32B" for KV estimation
        safety_margin_gb: VRAM to reserve for OS/other processes

    Returns:
        Estimated maximum context window tokens
    """
    safety_bytes = int(safety_margin_gb * 1024 ** 3)
    weight_bytes = model_file_size_bytes

    # Use free VRAM (not total) to account for other GPU consumers.
    # Weights still need to be loaded, so subtract them from free VRAM.
    vram_for_kv = vram_info.free_bytes - weight_bytes - safety_bytes

    if vram_for_kv <= 0:
        logger.warning(
            f"Model weights ({weight_bytes / 1e9:.1f}GB) + safety margin "
            f"({safety_margin_gb:.1f}GB) exceed free VRAM ({vram_info.free_gb:.1f}GB). "
            f"Cannot auto-size context window from VRAM."
        )
        # Return 0 to signal that auto-sizing cannot determine a safe value.
        # The caller should fall back to global config or a hardcoded default.
        return 0

    kv_bytes_per_token = _estimate_kv_bytes_per_token(model_size_hint, model_file_size_bytes)
    max_tokens = vram_for_kv // kv_bytes_per_token

    # Round down to nearest 1024 for cleaner numbers
    max_tokens = (max_tokens // 1024) * 1024

    # Clamp to reasonable range
    max_tokens = max(2048, min(max_tokens, 131072))  # 2K min, 128K max

    logger.debug(
        f"VRAM auto-sizing: {vram_info.total_gb:.1f}GB total, {vram_info.free_gb:.1f}GB free, "
        f"{weight_bytes / 1e9:.1f}GB weights, {safety_margin_gb:.1f}GB safety, "
        f"{vram_for_kv / 1e9:.1f}GB for KV → {max_tokens} tokens "
        f"(at {kv_bytes_per_token} bytes/token)"
    )

    return max_tokens


def get_model_file_size(model_path: str) -> int:
    """Get the file size of a model in bytes.

    Args:
        model_path: Path to the model file

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(model_path)
    except OSError:
        logger.debug(f"Could not get file size for: {model_path}")
        return 0


def auto_size_context_window(
    model_definition: dict,
    global_n_ctx: int = 32768,
    global_max_tokens: int = 4096,
    model_path: str | None = None,
) -> ContextSizingResult:
    """Determine the optimal context window for a model.

    Resolution order:
    1. Auto-detect from VRAM (for GGUF models only)
    2. Cap at `context_window` from model definition if set (user override)
    3. Cap at `native_context_limit` if set (model architecture limit)
    4. Fall back to global `LLAMA_CPP_N_CTX` if VRAM detection fails
    5. Hardcoded default (32768) as last resort

    For OpenAI-type models: Use `context_window` from model definition directly.

    Args:
        model_definition: Model definition dict from models.yaml
        global_n_ctx: Global LLAMA_CPP_N_CTX config value
        global_max_tokens: Global MAX_OUTPUT_TOKENS config value (unused here, for reference)
        model_path: Path to the GGUF model file (for VRAM auto-sizing)

    Returns:
        ContextSizingResult with the determined context window and source
    """
    model_type = model_definition.get("type", "")
    yaml_context_window = model_definition.get("context_window")
    native_limit = model_definition.get("native_context_limit")

    # ── OpenAI / Ollama / HF: no VRAM auto-sizing ──
    if model_type != "gguf":
        if yaml_context_window:
            return ContextSizingResult(
                context_window=int(yaml_context_window),
                source="model-definition",
            )
        # Default context windows for API models
        return ContextSizingResult(
            context_window=128000 if model_type == "openai" else global_n_ctx,
            source="default-api" if model_type == "openai" else "global-config",
        )

    # ── GGUF: try VRAM auto-sizing ──
    vram_info = detect_vram()

    if vram_info and model_path:
        file_size = get_model_file_size(model_path)
        if file_size > 0:
            # Extract model size hint from filename or repo_id
            size_hint = _extract_model_size_hint(model_definition)

            auto_ctx = estimate_max_context_from_vram(
                vram_info=vram_info,
                model_file_size_bytes=file_size,
                model_size_hint=size_hint,
            )

            # auto_ctx == 0 means VRAM budget is negative (model barely fits).
            # Fall through to global config instead of returning a bad value.
            if auto_ctx > 0:
                result = ContextSizingResult(
                    context_window=auto_ctx,
                    source="auto-vram",
                    vram_info=vram_info,
                    model_file_size_gb=file_size / (1024 ** 3),
                    estimated_kv_budget_gb=(
                        vram_info.free_bytes - file_size - int(1.5 * 1024 ** 3)
                    ) / (1024 ** 3),
                )

                # Apply caps
                if yaml_context_window and auto_ctx > int(yaml_context_window):
                    result.context_window = int(yaml_context_window)
                    result.source = "model-cap"
                if native_limit and result.context_window > int(native_limit):
                    result.context_window = int(native_limit)
                    result.source = "native-limit"

                return result
            else:
                logger.info(
                    f"VRAM auto-sizing returned 0 (insufficient free VRAM for "
                    f"confident sizing). Falling back to global config "
                    f"(n_ctx={global_n_ctx})."
                )

    # ── Fallback: no VRAM info or no model path ──
    if yaml_context_window:
        return ContextSizingResult(
            context_window=int(yaml_context_window),
            source="model-definition",
        )

    return ContextSizingResult(
        context_window=global_n_ctx,
        source="global-config",
    )


def _extract_model_size_hint(model_definition: dict) -> str | None:
    """Extract model size hint (e.g., '32B') from model definition.

    Looks at repo_id, filename, and description fields.
    """
    for field in ("repo_id", "filename", "description"):
        value = model_definition.get(field, "")
        if value:
            match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", str(value))
            if match:
                return match.group(0)
    return None
