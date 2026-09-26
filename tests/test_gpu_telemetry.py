"""Preflight telemetry never guesses capacity after a driver error."""

import subprocess

from local_model_bridge.gpu_telemetry import GpuTelemetry


def test_fresh_single_gpu_observation():
    def runner(argv, **kwargs):
        assert argv[0] == "nvidia-smi"
        assert kwargs["timeout"] == 4
        return subprocess.CompletedProcess(argv, 0, "GPU-example, NVIDIA RTX, 16303, 12021, 3819\n", "")

    state = GpuTelemetry(runner=runner).observe()
    assert (state["ready"], state["free_mib"], state["total_mib"]) == (True, 3819, 16303)
    assert state["observed_at"]


def test_driver_failure_and_unexpected_gpu_count_fail_closed():
    def failing(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 4)

    assert not GpuTelemetry(runner=failing).observe()["ready"]
    def multiple(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "GPU-one, RTX, 16000, 1000, 15000\nGPU-two, RTX, 16000, 1000, 15000\n", "")

    assert not GpuTelemetry(runner=multiple).observe()["ready"]
