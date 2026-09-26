"""Wan 2.2 inference, with a reusable pipeline in a dedicated GPU subprocess.

The bridge's embedding server never owns the video pipeline. A resident worker
can handle consecutive renders, then exit to release its CUDA/CPU allocations.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FPS = 24
# Stay conservative on a 16 GB card; the model is offloaded to system RAM.
DIMENSIONS = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (512, 512)},
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (704, 704)},
}


def dimensions(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    try:
        return DIMENSIONS[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported local video dimensions: {resolution} {aspect_ratio}") from exc


def load_source_image(source_image: str):
    """Only accept embedded image bytes, never fetch user-supplied URLs from the GPU worker."""
    from PIL import Image

    if not source_image.startswith("data:image/") or ";base64," not in source_image:
        raise ValueError("Local video requires an embedded image (data:image/...;base64,...)")
    encoded = source_image.split(";base64,", 1)[1]
    if len(encoded) > 30_000_000:
        raise ValueError("Source image is too large")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        image.load()
        return image.convert("RGB")


class WanPipelineRunner:
    """Reuse an offloaded Wan pipeline for successive renders of one input mode.

    Text and image variants are distinct pipeline classes. Switching variants
    releases the prior pipeline before loading the next; no network downloads.
    """

    def __init__(self, *, loader: Callable[[bool], Any] | None = None,
                 exporter: Callable[..., Any] | None = None):
        self._loader = loader or self._load_pipeline
        self._exporter = exporter
        self._pipeline: Any | None = None
        self._image_mode: bool | None = None

    @staticmethod
    def _load_pipeline(image_mode: bool):
        # Explicit install is the only network path; not even missing tokenizer
        # components may trigger a download during a render.
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available in local-model-bridge; check NVIDIA container runtime")
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
        )
        pipeline_class = WanImageToVideoPipeline if image_mode else WanPipeline
        pipeline = pipeline_class.from_pretrained(
            MODEL_ID, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        if image_mode and not pipeline.config.expand_timesteps:
            raise RuntimeError("Wan TI2V image conditioning requires expand_timesteps; check the checkpoint")
        pipeline.enable_sequential_cpu_offload()
        pipeline.vae.enable_tiling()
        return pipeline

    @property
    def resident(self) -> bool:
        return self._pipeline is not None

    def unload(self) -> None:
        self._pipeline = None
        self._image_mode = None

    def render(self, job: dict, destination: Path) -> dict:
        width, height = dimensions(job["resolution"], job["aspect_ratio"])
        duration = float(job["duration"])
        if not 1 <= duration <= 15:
            raise ValueError("Duration must be between 1 and 15 seconds")
        # Wan temporal latents require 4n+1 frames. Trim the exported result to
        # the requested duration; rounding up would lengthen a 5s clip.
        requested_frames = max(1, round(duration * FPS))
        num_frames = 4 * ((requested_frames - 1 + 3) // 4) + 1
        source_image = job.get("source_image")
        image_mode = bool(source_image)
        if self._pipeline is None or self._image_mode != image_mode:
            self.unload()
            self._pipeline = self._loader(image_mode)
            self._image_mode = image_mode
        kwargs = {
            "prompt": job["prompt"], "height": height, "width": width,
            "num_frames": num_frames, "num_inference_steps": 35, "guidance_scale": 5.0,
        }
        try:
            if source_image:
                kwargs["image"] = load_source_image(source_image)
            output = self._pipeline(**kwargs).frames[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if self._exporter is None:
                from diffusers.utils import export_to_video
                self._exporter = export_to_video
            self._exporter(output[:requested_frames], str(destination), fps=FPS)
            return {"width": width, "height": height, "actual_duration": len(output[:requested_frames]) / FPS}
        except Exception:
            # After an inference/export failure the CUDA state and pipeline
            # validity are unknown. Do not reuse it for another render.
            self.unload()
            raise


def run_job(job: dict, destination: Path) -> dict:
    """Legacy one-shot invocation until the coordinator owns worker lifetime."""
    return WanPipelineRunner().render(job, destination)


def serve_commands(*, reader=None, writer=None, runner: WanPipelineRunner | None = None) -> None:
    """One command at a time, JSON lines on private stdio; EOF unloads Wan.

    A failed render returns an error and exits the worker. The supervising
    bridge must reconcile the durable render claim if this process disappears.
    """
    reader = reader if reader is not None else sys.stdin
    writer = writer if writer is not None else sys.stdout
    runner = runner if runner is not None else WanPipelineRunner()
    try:
        for line in reader:
            try:
                command = json.loads(line)
                if not isinstance(command, dict) or command.get("action") not in ("render", "status", "unload"):
                    raise ValueError("Unsupported worker command")
                action = command["action"]
                if action == "render":
                    job_path = Path(command["job"])
                    output_path = Path(command["output"])
                    job = json.loads(job_path.read_text(encoding="utf-8"))
                    info = runner.render(job, output_path)
                    output_path.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
                    result = {"ok": True, "resident": runner.resident, "metadata": info}
                elif action == "status":
                    result = {"ok": True, "resident": runner.resident}
                else:
                    runner.unload()
                    result = {"ok": True, "resident": False}
                writer.write(json.dumps(result) + "\n")
                writer.flush()
                if action == "unload":
                    break
            except Exception as exc:
                runner.unload()
                writer.write(json.dumps({"ok": False, "resident": False, "error": f"{type(exc).__name__}: {exc}"[-1600:]}) + "\n")
                writer.flush()
                # Even after dereferencing an offloaded pipeline, a CUDA fault
                # can leave device allocations behind. Exiting the subprocess
                # is the only reliable release before another mode acquires it.
                break
    finally:
        runner.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Wan video generation")
    parser.add_argument("job", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--resident", action="store_true")
    args = parser.parse_args()
    if args.resident:
        if args.job or args.output:
            parser.error("Resident mode accepts no job arguments")
        serve_commands()
    else:
        if args.job is None or args.output is None:
            parser.error("One-shot mode requires job and output paths")
        job = json.loads(args.job.read_text(encoding="utf-8"))
        info = run_job(job, args.output)
        args.output.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")


if __name__ == "__main__":
    main()
