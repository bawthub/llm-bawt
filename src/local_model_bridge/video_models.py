"""Inventory and explicit installation for the one supported local video model.

Weights live exclusively in the GPU bridge's mounted Hugging Face hub cache.
Generation never downloads models; installation must be requested separately.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
from pathlib import Path

from fastapi import HTTPException
from huggingface_hub import constants, snapshot_download, try_to_load_from_cache

# Transfer-cache behavior is process-wide: set this in the GPU bridge environment
# before importing huggingface_hub, rather than mutating it after other imports.

from .video_worker import MODEL_ID

logger = logging.getLogger(__name__)

# No local_dir: snapshot_download stores the blobs once in the existing mounted
# HF hub cache. Keep this list synchronized with the model's index files.
REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/spiece.model",
    "tokenizer/special_tokens_map.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    *(f"text_encoder/model-{i:05d}-of-00003.safetensors" for i in range(1, 4)),
    *(f"transformer/diffusion_pytorch_model-{i:05d}-of-00005.safetensors" for i in range(1, 6)),
)
EXPECTED_DOWNLOAD_BYTES = 34_200_000_000


class VideoModelManager:
    def __init__(self, cache: Path | None = None):
        self.cache = cache or Path(constants.HF_HUB_CACHE)
        self.repo_dir = self.cache / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
        self._lock = asyncio.Lock()
        self._download_task: asyncio.Task | None = None
        self._error: str | None = None

    def installed(self) -> bool:
        return all(
            isinstance(path := try_to_load_from_cache(MODEL_ID, name, cache_dir=self.cache), str)
            and Path(path).is_file()
            for name in REQUIRED_FILES
        )

    def status(self, *, active: bool = False) -> dict:
        # Count physical blobs, not snapshot symlinks (multiple revisions may
        # refer to the same file). Include resumable .incomplete files.
        blobs = self.repo_dir / "blobs"
        size = sum(f.stat().st_size for f in blobs.iterdir() if f.is_file() and not f.is_symlink()) if blobs.is_dir() else 0
        installing = self._download_task is not None and not self._download_task.done()
        return {
            "model": "wan2.2-ti2v-5b",
            "worker_ready": importlib.util.find_spec("diffusers") is not None,
            "installed": self.installed(),
            # The current worker exits after each render; disk installation is
            # never evidence of a GPU-resident pipeline.
            "resident": False,
            "installing": installing,
            "active": active,
            "size_bytes": size,
            "expected_download_bytes": EXPECTED_DOWNLOAD_BYTES,
            "cache_path": str(self.repo_dir),
            "error": self._error,
        }

    async def install(self) -> dict:
        async with self._lock:
            if importlib.util.find_spec("diffusers") is None:
                raise HTTPException(status_code=503, detail="GPU video worker is not installed yet")
            if self._download_task and not self._download_task.done():
                raise HTTPException(status_code=409, detail="Video model download already in progress")
            if self.installed():
                return self.status()
            self._error = None
            self._download_task = asyncio.create_task(self._download())
            return self.status()

    async def _download(self) -> None:
        try:
            # Existing shared mounted cache; no local_dir or second copy.
            await asyncio.to_thread(
                snapshot_download, MODEL_ID, revision="main", cache_dir=self.cache,
                allow_patterns=list(REQUIRED_FILES),
            )
            if not self.installed():
                raise RuntimeError("Downloaded snapshot is missing required model files")
        except Exception as exc:
            self._error = str(exc)
            logger.exception("Wan checkpoint download failed; partial files remain visible and resumable")

    async def remove(self, *, active: bool = False) -> dict:
        async with self._lock:
            if active or (self._download_task and not self._download_task.done()):
                raise HTTPException(status_code=409, detail="Video generation or model installation is active")
            # Fixed repo ID, not a user-provided path. No other model's cache
            # entries can be touched. Reject symlink tricks before deletion.
            if self.repo_dir.is_symlink() or self.repo_dir.parent.resolve() != self.cache.resolve():
                raise HTTPException(status_code=400, detail="Unsafe model cache path")
            if self.repo_dir.is_dir():
                await asyncio.to_thread(shutil.rmtree, self.repo_dir)
            self._error = None
            return self.status()
