"""TASK-1639: the bind dry run prints every refusal and creates nothing."""
import io

from llm_bawt.ops.bind_preflight import main, run
from llm_bawt.ops.service import OpsDispatchError
from llm_bawt.ops.source_state import SourceStateError, SourceStateRefusals


def test_a_prepared_binding_passes():
    out = io.StringIO()
    snapshot = {"config": {"drift": ["env"]}, "gpu_target": None}
    assert run("op.slug", None, lambda op, args: snapshot, out) == 0
    assert out.getvalue().startswith("OK op.slug") and "drift: env" in out.getvalue()


def test_every_refusal_is_listed():
    refusals = SourceStateRefusals([SourceStateError("source_mount_unclassified", ["/run/x"]),
                                    SourceStateError("source_env_unsafe", ["HF_HOME"])])

    def prepare(op, args):
        raise OpsDispatchError(refusals.code, f"config target cannot be bound: {refusals}")

    out = io.StringIO()
    assert run("op.slug", {"a": 1}, prepare, out) == 1
    lines = out.getvalue().splitlines()
    assert lines[0] == "REFUSED op.slug (source_mount_unclassified): 2 refusal(s)"
    assert "/run/x" in lines[1] and lines[2] == "  - source_env_unsafe: HF_HOME"


def test_args_must_be_a_json_object(capsys):
    import pytest

    for bad in ("[1]", "nope"):
        with pytest.raises(SystemExit) as caught:
            main(["op.slug", "--args", bad])
        assert caught.value.code == 2
