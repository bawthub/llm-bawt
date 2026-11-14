"""Unit tests for :mod:`llm_bawt.service.jobs.media_gc` (TASK-231).

These tests exercise the GC orchestration and SQL builder in isolation
— without a live Postgres. The orphan-finding query itself uses
Postgres-only constructs (``jsonb_array_elements``, ``make_interval``)
so its on-DB behavior is covered by the integration smoke test
documented in TASK-231; here we verify:

* the orphan-eviction loop calls ``MediaStore.delete`` once per row
* counts reported in the result dict (``deleted_count`` /
  ``freed_bytes``) match what we deleted
* dry-run mode never touches the store
* the SQL builder degrades safely when there are zero messages tables

The 7-day grace + ``expires_at`` semantics live entirely in the SQL
query, so we verify those via the contract test: when the find-orphans
hook returns row X, the GC loop deletes row X regardless of *why* the
query picked it. Both the "8-days-old & unreferenced" and "expired"
branches map to the same downstream behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from llm_bawt.media.store import MediaStore
from llm_bawt.service.jobs import media_gc
from llm_bawt.service.jobs.media_gc import (
    ReferenceScanIncomplete,
    breaker_limit,
    parse_bawthub_references,
    run_media_gc,
)

_VALID_ID = "ma_" + "A" * 26


# ---------------------------------------------------------------------------
# In-memory fakes — mirror tests/test_media_store.py
# ---------------------------------------------------------------------------


class FakeMediaAssetStore:
    """Stand-in for :class:`MediaAssetStore` used by :class:`MediaStore`.

    Tracks rows in a dict so ``MediaStore.delete`` can resolve the
    sha256 and call back into ``delete``.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.by_sha: dict[str, str] = {}

    def add(self, *, asset_id: str, sha256: str, size_bytes: int) -> None:
        self.rows[asset_id] = {
            "id": asset_id,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        self.by_sha[sha256] = asset_id

    def get_by_id(self, asset_id: str) -> Optional[dict[str, Any]]:
        row = self.rows.get(asset_id)
        return dict(row) if row else None

    def get_by_sha256(self, sha256: str) -> Optional[dict[str, Any]]:
        rid = self.by_sha.get(sha256)
        return dict(self.rows[rid]) if rid else None

    def delete(self, asset_id: str) -> bool:
        row = self.rows.pop(asset_id, None)
        if row is None:
            return False
        self.by_sha.pop(row["sha256"], None)
        return True


def _write_blob_set(root: Path, sha: str) -> list[Path]:
    """Write the three blob variants the deletion path expects.

    Mirrors :data:`llm_bawt.media.store.VARIANT_DIRS` so we can assert
    they're all gone after a GC run.
    """
    from llm_bawt.media.store import VARIANT_DIRS

    paths: list[Path] = []
    for subdir in VARIANT_DIRS.values():
        p = root / subdir / sha[:2] / sha[2:4] / f"{sha}.webp"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-webp-bytes")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_store(tmp_path: Path) -> tuple[MediaStore, FakeMediaAssetStore]:
    """Build a real MediaStore over a tmp-path tree + in-memory DB fake.

    Returns ``(media_store, asset_store)`` so tests can both manipulate
    rows directly and assert what the GC removed.
    """
    asset_store = FakeMediaAssetStore()
    store = MediaStore(root=tmp_path, db=asset_store)
    return store, asset_store


@pytest.fixture
def patched_orphan_finder(monkeypatch: pytest.MonkeyPatch):
    """Stub out the Postgres-only SQL hook.

    Tests configure ``state["orphans"]`` to whatever orphan rows the SQL
    "would have" returned, and ``state["tables"]`` to the discovered
    messages-table count.
    """
    state: dict[str, Any] = {
        "orphans": [],
        "local_ids": {"ma_" + "R" * 26},
        "remote_ids": {"ma_" + "B" * 26},
        "collectable": 100,
    }

    def fake_find_orphans(conn, referenced, grace_days):
        state["referenced_seen"] = set(referenced)
        return [r for r in state["orphans"] if r["id"] not in referenced]

    def fake_collect(conn, *, deadline, clock=None):
        return set(state["local_ids"]), 1

    def fake_fetch():
        return set(state["remote_ids"]), 1

    def fake_engine(config, **kwargs):
        # Any sentinel that supports ``connect()`` / ``begin()`` as context
        # managers; the query callbacks above ignore the conn entirely.
        class _DummyConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def begin(self):
                return self

            def exec_driver_sql(self, sql):
                state.setdefault("driver_sql", []).append(sql)

        class _DummyEngine:
            def connect(self):
                return _DummyConn()

        return _DummyEngine()

    monkeypatch.setattr(media_gc, "_find_orphan_assets", fake_find_orphans)
    monkeypatch.setattr(media_gc, "collect_local_references", fake_collect)
    monkeypatch.setattr(media_gc, "fetch_bawthub_references", fake_fetch)
    monkeypatch.setattr(media_gc, "_count_collectable", lambda conn: state["collectable"])
    monkeypatch.setattr(media_gc, "get_shared_engine", fake_engine, raising=False)
    # ``run_media_gc`` does ``from llm_bawt.utils.db import get_shared_engine``
    # late, so we have to patch the source module too.
    import llm_bawt.utils.db as utils_db

    monkeypatch.setattr(utils_db, "get_shared_engine", fake_engine)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunMediaGc:
    """End-to-end orchestration tests with a faked SQL layer."""

    def test_no_orphans_returns_zero_counts(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        patched_orphan_finder: dict,
    ):
        """No orphan rows → deleted_count=0, freed_bytes=0, errors=[].

        Validates: "running it twice in a row should be a no-op the
        second time" — same shape as zero orphans on the second pass.
        """
        store, _ = fake_store
        patched_orphan_finder["orphans"] = []

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 0
        assert result["deleted_count"] == 0
        assert result["freed_bytes"] == 0
        assert result["dry_run"] is False
        assert result["errors"] == []
        assert result["scanned_columns"] == 1
        assert result["bawthub_columns"] == 1
        assert "aborted_reason" not in result

    def test_orphan_evicted_deletes_row_and_blobs(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Orphan from 8 days ago with no references → row + 3 blobs gone.

        Covers TASK-231 acceptance test "orphan from 8 days ago gets
        GC'd" plus the blob-cleanup requirement.
        """
        store, asset_store = fake_store

        sha = "a" * 64
        asset_id = "ma_OLD_ORPHAN"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=12345)
        blob_paths = _write_blob_set(tmp_path, sha)
        for p in blob_paths:
            assert p.exists()

        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 12345}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 1
        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 12345
        assert result["errors"] == []
        # Row gone
        assert asset_store.get_by_id(asset_id) is None
        # All three blob variants gone
        for p in blob_paths:
            assert not p.exists(), f"variant blob lingered: {p}"

    def test_referenced_asset_not_passed_to_loop(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Asset NOT in the orphan list survives — referenced/young.

        Mirrors "orphan from yesterday survives" + "referenced asset
        survives even after 8 days": both are characterized by the SQL
        not returning the row, which is the only signal the loop sees.
        """
        store, asset_store = fake_store

        sha_keep = "b" * 64
        keep_id = "ma_KEEP_ME"
        asset_store.add(asset_id=keep_id, sha256=sha_keep, size_bytes=999)
        keep_paths = _write_blob_set(tmp_path, sha_keep)

        # SQL hook returns nothing — the GC loop never sees this row.
        patched_orphan_finder["orphans"] = []

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["deleted_count"] == 0
        # Row + blobs all still there
        assert asset_store.get_by_id(keep_id) is not None
        for p in keep_paths:
            assert p.exists()

    def test_expires_at_path_uses_same_deletion(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """An ``expires_at < now()`` row is GC'd the same way as an aged orphan.

        The two SQL branches funnel into the same downstream loop, so
        the test stages the row as if the SQL had picked it for the
        ``expires_at`` reason: shows the deletion path is identical.
        """
        store, asset_store = fake_store

        sha = "c" * 64
        asset_id = "ma_EXPIRED"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=2048)
        expired_paths = _write_blob_set(tmp_path, sha)

        # The SQL would return this asset because expires_at < now() —
        # we simulate that by having the find-orphans hook return it
        # even though it's "fresh" age-wise.
        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 2048}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 2048
        assert asset_store.get_by_id(asset_id) is None
        for p in expired_paths:
            assert not p.exists()

    def test_dry_run_does_not_delete(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """Dry-run reports counts without touching DB or disk."""
        store, asset_store = fake_store

        sha = "d" * 64
        asset_id = "ma_DRYRUN"
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=4096)
        dry_paths = _write_blob_set(tmp_path, sha)

        patched_orphan_finder["orphans"] = [
            {"id": asset_id, "sha256": sha, "size_bytes": 4096}
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
            dry_run=True,
        )

        assert result["orphan_count"] == 1
        assert result["deleted_count"] == 0
        assert result["dry_run"] is True
        # Dry-run reports what would have been freed (so ops can size the run).
        assert result["freed_bytes"] == 4096
        # Nothing actually removed.
        assert asset_store.get_by_id(asset_id) is not None
        for p in dry_paths:
            assert p.exists()

    def test_counts_aggregate_across_multiple_orphans(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
    ):
        """``deleted_count`` and ``freed_bytes`` sum correctly across rows."""
        store, asset_store = fake_store

        sizes = [100, 500, 4000]
        for i, size in enumerate(sizes):
            sha = (f"{i:x}" * 64)[:64]
            asset_store.add(asset_id=f"ma_M_{i}", sha256=sha, size_bytes=size)
            _write_blob_set(tmp_path, sha)

        patched_orphan_finder["orphans"] = [
            {"id": f"ma_M_{i}", "sha256": (f"{i:x}" * 64)[:64], "size_bytes": s}
            for i, s in enumerate(sizes)
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 3
        assert result["deleted_count"] == 3
        assert result["freed_bytes"] == sum(sizes)
        for i in range(3):
            assert asset_store.get_by_id(f"ma_M_{i}") is None

    def test_delete_error_continues_sweep(
        self,
        fake_store: tuple[MediaStore, FakeMediaAssetStore],
        tmp_path: Path,
        patched_orphan_finder: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A single failed delete is recorded but doesn't abort the sweep.

        Re-entrancy guarantee: a flaky delete leaves the remaining
        orphans for the next nightly run, but doesn't stop us from
        cleaning up the rest of *this* run.
        """
        store, asset_store = fake_store

        sha_bad = "e" * 64
        sha_ok = "f" * 64
        asset_store.add(asset_id="ma_BAD", sha256=sha_bad, size_bytes=1)
        asset_store.add(asset_id="ma_OK", sha256=sha_ok, size_bytes=200)
        _write_blob_set(tmp_path, sha_bad)
        _write_blob_set(tmp_path, sha_ok)

        original_delete = store.delete

        def maybe_explode(asset_id: str) -> None:
            if asset_id == "ma_BAD":
                raise RuntimeError("simulated blob unlink failure")
            return original_delete(asset_id)

        monkeypatch.setattr(store, "delete", maybe_explode)

        patched_orphan_finder["orphans"] = [
            {"id": "ma_BAD", "sha256": sha_bad, "size_bytes": 1},
            {"id": "ma_OK", "sha256": sha_ok, "size_bytes": 200},
        ]

        result = run_media_gc(
            config=object(),
            media_store=store,
        )

        assert result["orphan_count"] == 2
        assert result["deleted_count"] == 1
        assert result["freed_bytes"] == 200
        assert len(result["errors"]) == 1
        assert result["errors"][0]["asset_id"] == "ma_BAD"
        assert "simulated blob unlink failure" in result["errors"][0]["error"]
        # Successful asset is gone; failed one stays for the next run.
        assert asset_store.get_by_id("ma_BAD") is not None
        assert asset_store.get_by_id("ma_OK") is None


class TestFailClosed:
    """TASK-2224: an incomplete reference picture deletes nothing."""

    def _stage_orphan(self, fake_store, tmp_path, asset_id="ma_ORPHAN", sha="9" * 64):
        store, asset_store = fake_store
        asset_store.add(asset_id=asset_id, sha256=sha, size_bytes=10)
        paths = _write_blob_set(tmp_path, sha)
        return store, asset_store, paths, {"id": asset_id, "sha256": sha, "size_bytes": 10}

    def test_bawthub_referenced_asset_survives(self, fake_store, tmp_path, patched_orphan_finder):
        """A card-only image (referenced only in BawtHub) is not collected."""
        store, asset_store, paths, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        patched_orphan_finder["remote_ids"] = {row["id"]}

        result = run_media_gc(config=object(), media_store=store)

        assert result["deleted_count"] == 0
        assert row["id"] in patched_orphan_finder["referenced_seen"]
        assert asset_store.get_by_id(row["id"]) is not None
        assert all(p.exists() for p in paths)

    @pytest.mark.parametrize("failure", ["local", "remote"])
    def test_unreadable_source_deletes_nothing(
        self, fake_store, tmp_path, patched_orphan_finder, monkeypatch, failure
    ):
        store, asset_store, paths, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]

        def boom(*a, **k):
            raise ReferenceScanIncomplete(f"{failure} source unreadable")

        target = "collect_local_references" if failure == "local" else "fetch_bawthub_references"
        monkeypatch.setattr(media_gc, target, boom)

        result = run_media_gc(config=object(), media_store=store)

        assert result["deleted_count"] == 0
        assert f"{failure} source unreadable" in result["aborted_reason"]
        assert asset_store.get_by_id(row["id"]) is not None
        assert all(p.exists() for p in paths)

    def test_budget_exceeded_after_remote_deletes_nothing(self, fake_store, tmp_path, patched_orphan_finder):
        store, asset_store, _, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        ticks = iter([0.0, 10_000.0])

        result = run_media_gc(
            config=object(), media_store=store, sweep_budget_s=60, clock=lambda: next(ticks)
        )

        assert "budget" in result["aborted_reason"]
        assert asset_store.get_by_id(row["id"]) is not None

    def test_empty_reference_set_deletes_nothing(self, fake_store, tmp_path, patched_orphan_finder):
        store, asset_store, _, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        patched_orphan_finder["local_ids"] = set()
        patched_orphan_finder["remote_ids"] = set()

        result = run_media_gc(config=object(), media_store=store)

        assert "empty" in result["aborted_reason"]
        assert asset_store.get_by_id(row["id"]) is not None

    @pytest.mark.parametrize("empty", ["local", "remote"])
    def test_one_empty_source_deletes_nothing(
        self, fake_store, tmp_path, patched_orphan_finder, empty
    ):
        """Codex cmwfwt2l1: the other source's references must not mask an empty one."""
        store, asset_store, paths, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        patched_orphan_finder[f"{empty}_ids"] = set()

        result = run_media_gc(config=object(), media_store=store)

        assert result["deleted_count"] == 0
        assert "empty" in result["aborted_reason"]
        assert asset_store.get_by_id(row["id"]) is not None
        assert all(p.exists() for p in paths)

    def test_budget_exceeded_during_candidate_lookup_deletes_nothing(
        self, fake_store, tmp_path, patched_orphan_finder
    ):
        store, asset_store, paths, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        # start, after remote, timeout bound, after candidate lookup
        ticks = iter([0.0, 1.0, 1.0, 10_000.0])

        result = run_media_gc(
            config=object(), media_store=store, sweep_budget_s=60, clock=lambda: next(ticks)
        )

        assert result["deleted_count"] == 0
        assert "candidate lookup" in result["aborted_reason"]
        assert asset_store.get_by_id(row["id"]) is not None
        assert all(p.exists() for p in paths)

    def test_candidate_queries_bounded_by_remaining_budget(
        self, fake_store, tmp_path, patched_orphan_finder
    ):
        store, _, _, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]
        ticks = iter([0.0, 1.0, 50.0, 51.0])

        run_media_gc(config=object(), media_store=store, sweep_budget_s=60, clock=lambda: next(ticks))

        assert patched_orphan_finder["driver_sql"] == ["SET LOCAL statement_timeout = 10000"]

    def test_breaker_trips_and_override_allows(self, fake_store, tmp_path, patched_orphan_finder):
        store, asset_store = fake_store
        rows = []
        for i in range(30):
            sha = f"{i:064x}"
            asset_store.add(asset_id=f"ma_B{i}", sha256=sha, size_bytes=1)
            rows.append({"id": f"ma_B{i}", "sha256": sha, "size_bytes": 1})
        patched_orphan_finder["orphans"] = rows
        patched_orphan_finder["collectable"] = 100
        assert breaker_limit(100) == 25

        tripped = run_media_gc(config=object(), media_store=store, include_candidates=True)
        assert tripped["deleted_count"] == 0
        assert "breaker" in tripped["aborted_reason"]
        assert len(tripped["candidates"]) == 30
        assert all(asset_store.get_by_id(r["id"]) for r in rows)

        purged = run_media_gc(config=object(), media_store=store, max_candidates=30)
        assert purged["deleted_count"] == 30

    def test_scheduled_result_carries_no_ids(self, fake_store, tmp_path, patched_orphan_finder):
        """job_runs is swept for references, so default results hold no ids."""
        store, _, _, row = self._stage_orphan(fake_store, tmp_path)
        patched_orphan_finder["orphans"] = [row]

        result = run_media_gc(config=object(), media_store=store, dry_run=True)

        assert "candidates" not in result
        assert row["id"] not in repr(result)


class TestParseBawthubReferences:
    def test_valid(self):
        ids, columns = parse_bawthub_references({"complete": True, "columns_scanned": 9, "ids": [_VALID_ID]})
        assert ids == {_VALID_ID} and columns == 9

    def test_empty_but_complete_is_valid(self):
        assert parse_bawthub_references({"complete": True, "columns_scanned": 3, "ids": []}) == (set(), 3)

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "<html>app shell</html>",
            [],
            {"columns_scanned": 3, "ids": []},
            {"complete": "true", "columns_scanned": 3, "ids": []},
            {"complete": True, "columns_scanned": 0, "ids": []},
            {"complete": True, "columns_scanned": True, "ids": []},
            {"complete": True, "columns_scanned": 3},
            {"complete": True, "columns_scanned": 3, "ids": "ma_x"},
            {"complete": True, "columns_scanned": 3, "ids": ["not-an-id"]},
            {"complete": True, "columns_scanned": 3, "ids": [_VALID_ID + "Z"]},
        ],
    )
    def test_malformed_rejected(self, payload):
        with pytest.raises(ReferenceScanIncomplete):
            parse_bawthub_references(payload)


# ---------------------------------------------------------------------------
# TASK-847: mixed image / file orphans
# ---------------------------------------------------------------------------


def test_gc_removes_file_kind_blob_and_image_variants(
    fake_store: tuple[MediaStore, FakeMediaAssetStore],
    tmp_path: Path,
    patched_orphan_finder: dict,
) -> None:
    """A ``kind=file`` orphan loses its single ``files/`` blob; an image orphan
    loses its three variants; both rows go."""
    from llm_bawt.media.asset_kinds import FILES_DIR

    store, asset_store = fake_store
    img_sha = "a" * 64
    file_sha = "b" * 64
    img_paths = _write_blob_set(tmp_path, img_sha)
    file_path = tmp_path / FILES_DIR / file_sha[:2] / file_sha[2:4] / file_sha
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"pdf-ish")

    asset_store.add(asset_id="ma_img", sha256=img_sha, size_bytes=300)
    asset_store.add(asset_id="ma_file", sha256=file_sha, size_bytes=7)
    asset_store.rows["ma_file"]["kind"] = "file"
    asset_store.rows["ma_file"]["filename"] = "x.pdf"

    patched_orphan_finder["orphans"] = [
        {"id": "ma_img", "sha256": img_sha, "size_bytes": 300},
        {"id": "ma_file", "sha256": file_sha, "size_bytes": 7},
    ]
    result = run_media_gc(config=object(), media_store=store)
    assert result["deleted_count"] == 2
    assert result["freed_bytes"] == 307
    assert result["errors"] == []
    assert all(not p.exists() for p in img_paths)
    assert not file_path.exists()
    assert asset_store.rows == {}
