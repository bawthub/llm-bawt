"""CUDA allocator high-water marks plus sampled device-wide usage per render.

Allocator peaks are exact for PyTorch; device-wide peak is sampled (250 ms),
not an assertion that every transient driver allocation was captured.
"""
from __future__ import annotations

import math
import threading

from .gpu_telemetry import GpuTelemetry


class VideoMemoryMeasurement:
    def __init__(self, *, cuda=None, telemetry=None):
        if cuda is None:
            import torch
            cuda = torch.cuda
        self.cuda = cuda
        self.telemetry = telemetry or GpuTelemetry()
        self._stop = threading.Event()
        self._sample_error = False
        self._peak_used = 0

    def _sample(self):
        try:
            free, total = self.cuda.mem_get_info()
            self._peak_used = max(self._peak_used, total - free)
        except Exception:
            self._sample_error = True

    def _sample_loop(self):
        while not self._stop.wait(0.25):
            self._sample()

    def __enter__(self):
        self.device = self.telemetry.observe()
        if not self.device.get("ready"):
            raise RuntimeError("GPU measurement telemetry unavailable")
        self.cuda.synchronize()
        self.cuda.reset_peak_memory_stats()
        free, self.total = self.cuda.mem_get_info()
        self.baseline = self.total - free
        self._peak_used = self.baseline
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.cuda.synchronize()
                self._sample()
        finally:
            self._stop.set()
            self._thread.join(timeout=2)
        if exc_type is None and (self._sample_error or self._thread.is_alive()):
            raise RuntimeError("GPU memory sampling failed")

    def result(self):
        mib = 1024 * 1024
        # total_mib: CUDA-usable memory (capacity math). device_total_mib: the
        # nvidia-smi figure the handoff consented to (identity only). They differ
        # (e.g. RTX 5080: 15838 vs 16303 MiB), so never compare across sources.
        return {"gpu_uuid": self.device["uuid"], "total_mib": self.total // mib,
                "device_total_mib": self.device["total_mib"],
                "baseline_used_mib": math.ceil(self.baseline / mib),
                "sampled_peak_used_mib": math.ceil(self._peak_used / mib),
                "peak_reserved_mib": math.ceil(self.cuda.max_memory_reserved() / mib),
                "peak_allocated_mib": math.ceil(self.cuda.max_memory_allocated() / mib),
                "worker_reserved_mib": math.ceil(self.cuda.memory_reserved() / mib)}
