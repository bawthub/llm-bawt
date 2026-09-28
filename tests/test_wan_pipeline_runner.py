"""Resident Wan inference retains one pipeline across compatible jobs."""
import io
import json
from types import SimpleNamespace

import pytest

from local_model_bridge.video_worker import WanPipelineRunner, serve_commands


def test_compatible_renders_reuse_pipeline_and_unload(tmp_path):
    loaded = []
    exported = []

    class Pipeline:
        def __call__(self, **kwargs):
            return SimpleNamespace(frames=[[object()] * kwargs["num_frames"]])

    def loader(image_mode):
        loaded.append(image_mode)
        return Pipeline()

    runner = WanPipelineRunner(loader=loader, exporter=lambda frames, path, fps: exported.append((len(frames), path, fps)))
    job = {"prompt": "dog", "resolution": "480p", "aspect_ratio": "16:9", "duration": 1}
    assert runner.render(job, tmp_path / "one.mp4")["actual_duration"] == 1
    assert runner.render(job, tmp_path / "two.mp4")["actual_duration"] == 1
    assert loaded == [False]
    assert len(exported) == 2
    assert runner.resident is True
    runner.unload()
    assert runner.resident is False
    runner.render(job, tmp_path / "three.mp4")
    assert loaded == [False, False]


def test_render_failure_drops_pipeline_before_retry(tmp_path):
    loaded = []

    class Pipeline:
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA failure")

    runner = WanPipelineRunner(loader=lambda mode: loaded.append(mode) or Pipeline())
    job = {"prompt": "dog", "resolution": "480p", "aspect_ratio": "16:9", "duration": 1}
    with pytest.raises(RuntimeError, match="CUDA failure"):
        runner.render(job, tmp_path / "bad.mp4")
    assert runner.resident is False
    with pytest.raises(RuntimeError, match="CUDA failure"):
        runner.render(job, tmp_path / "retry.mp4")
    assert loaded == [False, False]


def test_resident_command_loop_reuses_pipeline_then_unloads(tmp_path):
    loaded = []

    class Pipeline:
        def __call__(self, **kwargs):
            return SimpleNamespace(frames=[[object()] * kwargs["num_frames"]])

    runner = WanPipelineRunner(loader=lambda mode: loaded.append(mode) or Pipeline(),
                               exporter=lambda frames, path, fps: None)
    job = tmp_path / "input.json"
    job.write_text(json.dumps({"prompt": "dog", "resolution": "480p", "aspect_ratio": "16:9", "duration": 1}))
    commands = [
        {"action": "render", "job": str(job), "output": str(tmp_path / "one.mp4")},
        {"action": "render", "job": str(job), "output": str(tmp_path / "two.mp4")},
        {"action": "status"}, {"action": "unload"},
    ]
    writer = io.StringIO()
    serve_commands(reader=io.StringIO("".join(json.dumps(command) + "\n" for command in commands)),
                   writer=writer, runner=runner)
    replies = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert [reply["resident"] for reply in replies] == [True, True, True, False]
    assert loaded == [False]
    assert runner.resident is False
    assert (tmp_path / "two.json").is_file()


def test_input_mode_change_reloads_pipeline(monkeypatch, tmp_path):
    import local_model_bridge.video_worker as worker

    loaded = []

    class Pipeline:
        def __call__(self, **kwargs):
            return SimpleNamespace(frames=[[object()] * kwargs["num_frames"]])

    monkeypatch.setattr(worker, "load_source_image", lambda image, width, height: object())
    runner = WanPipelineRunner(loader=lambda mode: loaded.append(mode) or Pipeline(),
                               exporter=lambda frames, path, fps: None)
    job = {"prompt": "dog", "resolution": "480p", "aspect_ratio": "16:9", "duration": 1}
    runner.render(job, tmp_path / "text.mp4")
    runner.render({**job, "source_image": "data:image/png;base64,unused"}, tmp_path / "image.mp4")
    runner.render(job, tmp_path / "text-again.mp4")
    assert loaded == [False, True, False]


def test_invalid_resident_command_unloads_and_reports_error(tmp_path):
    class Pipeline:
        def __call__(self, **kwargs):
            raise RuntimeError("GPU fault")

    runner = WanPipelineRunner(loader=lambda mode: Pipeline())
    job = tmp_path / "input.json"
    job.write_text(json.dumps({"prompt": "dog", "resolution": "480p", "aspect_ratio": "16:9", "duration": 1}))
    commands = [{"action": "render", "job": str(job), "output": str(tmp_path / "bad.mp4")},
                {"action": "status"}]
    writer = io.StringIO()
    serve_commands(reader=io.StringIO("".join(json.dumps(command) + "\n" for command in commands)),
                   writer=writer, runner=runner)
    replies = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert len(replies) == 1
    assert replies[0]["ok"] is False and "GPU fault" in replies[0]["error"]
    assert runner.resident is False


def test_invalid_job_does_not_load_pipeline(tmp_path):
    loaded = []
    runner = WanPipelineRunner(loader=lambda mode: loaded.append(mode))
    with pytest.raises(ValueError, match="Duration"):
        runner.render({"resolution": "480p", "aspect_ratio": "16:9", "duration": 16}, tmp_path / "bad.mp4")
    assert loaded == []
