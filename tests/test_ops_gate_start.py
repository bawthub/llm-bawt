"""Dormant start storage is exclusive, bounded, and never a launch permit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bawt.ops import gate_start as start


def test_nonpostgresql_refuses_without_touching_storage(tmp_path):
    store = SimpleNamespace(_require=lambda: None,
        engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    root = tmp_path / "never-created"
    with pytest.raises(start.GateStartError, match="^gate_start_custody_invalid$"):
        with start.reserve_gate_start(store, "1" * 32, receipt_root=root):
            pytest.fail("non-PG start yielded")
    assert not root.exists()


def test_start_marker_is_private_and_never_overwrites_even_identical_bytes(tmp_path):
    raw = b'fixture-start-intent'
    start._publish(tmp_path, raw)
    path = tmp_path / start._START
    assert path.read_bytes() == raw and path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        start._publish(tmp_path, raw)
    assert path.read_bytes() == raw


def test_start_marker_never_follows_an_existing_link(tmp_path):
    target = tmp_path / "outside"
    target.write_bytes(b"retained")
    (tmp_path / start._START).symlink_to(target)
    with pytest.raises(FileExistsError):
        start._publish(tmp_path, b"not-written")
    assert target.read_bytes() == b"retained"
