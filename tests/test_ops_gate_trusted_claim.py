"""Trusted claim stays an internal PostgreSQL-only, fixed-input seam."""
from types import SimpleNamespace

import pytest

from llm_bawt.ops.gate_trusted_claim import claim_trusted_gate_job


def test_off_postgresql_refuses_before_read_or_transaction():
    def forbidden():
        pytest.fail("non-PG claim opened a transaction")

    store = SimpleNamespace(_require=lambda: None,
        engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite"), begin=forbidden))
    assert claim_trusted_gate_job(store, "missing") is False


@pytest.mark.parametrize("override", ["turn_token", "caller", "required_tag", "environment", "plan"])
def test_no_caller_tag_environment_or_plan_override(override):
    with pytest.raises(TypeError):
        claim_trusted_gate_job(object(), "job", **{override: "not-authority"})
