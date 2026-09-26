"""Bounded, on-demand NVIDIA driver observations for video preflight.

No GPU-context creation, no cached health verdict: a stale container may report
healthy while newly started CUDA work cannot see the card.
"""
from __future__ import annotations

import csv
import io
import subprocess
from datetime import UTC, datetime


class GpuTelemetry:
    def __init__(self, *, runner=subprocess.run):
        self._runner = runner

    def observe(self) -> dict:
        observed_at = datetime.now(UTC).isoformat()
        try:
            result = self._runner(
                ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=True,
            )
            rows = list(csv.reader(io.StringIO(result.stdout)))
            if len(rows) != 1 or len(rows[0]) != 5:
                raise ValueError("Expected exactly one GPU")
            uuid, name, total, used, free = (value.strip() for value in rows[0])
            total_mib, used_mib, free_mib = int(total), int(used), int(free)
            if (not uuid.startswith("GPU-") or not name or total_mib <= 0 or used_mib < 0
                    or free_mib < 0 or free_mib > total_mib or used_mib > total_mib):
                raise ValueError("Invalid GPU telemetry")
            return {
                "ready": True, "observed_at": observed_at, "uuid": uuid, "name": name,
                "total_mib": total_mib, "used_mib": used_mib, "free_mib": free_mib,
            }
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {"ready": False, "observed_at": observed_at,
                    "error": f"GPU telemetry unavailable: {type(exc).__name__}"}
