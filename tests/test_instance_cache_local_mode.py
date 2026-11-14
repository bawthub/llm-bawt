"""Regression tests for local_mode cache poisoning in InstanceManagerMixin.

Scenario (TASK-29 incident):
    1. POST /v1/admin/reload-bots clears entire _llm_bawt_cache
    2. First request for bot=byte arrives with augment_memory=false → local_mode=True
    3. ServiceLLMBawt created WITHOUT memory, cached under (model, "byte", user)
    4. Subsequent augment_memory=true requests reuse the no-memory instance
    5. All future byte turns are memory-poisoned until next restart

The cache isolates local_mode in both directions. Memory-enabled instances
must not leak augmentation into requests explicitly disabling it.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from llm_bawt.service.background_service import BackgroundService


def _make_service(**overrides: Any) -> BackgroundService:
    """Build a minimal BackgroundService without full __init__."""
    svc = BackgroundService.__new__(BackgroundService)
    svc._llm_bawt_cache = {}
    svc._client_cache = {}
    svc._bg_client_cache = {}
    svc._bg_client_lock = threading.Lock()
    svc._session_model_overrides = {}
    svc._available_models = ["test-model"]
    svc._default_model = "test-model"

    lifecycle = MagicMock()
    lifecycle.current_model = "test-model"
    lifecycle.clear_pending_switch.return_value = None
    svc._model_lifecycle = lifecycle

    config = MagicMock()
    config.defined_models = {"models": {"test-model": {"type": "openai"}}}
    config.model_copy.return_value = config
    svc.config = config

    for k, v in overrides.items():
        setattr(svc, k, v)
    return svc


def _fake_instance(
    local_mode: bool,
    memory: Any = "auto",
    _db_available: bool = True,
) -> SimpleNamespace:
    """Return a lightweight stand-in for ServiceLLMBawt.

    Args:
        local_mode: Whether the instance was created in local mode.
        memory: Memory client (None = no memory, "auto" = MagicMock if not local).
        _db_available: Whether the DB was reachable at init time.
    """
    if memory == "auto":
        memory = None if local_mode else MagicMock()
    inst = SimpleNamespace(
        local_mode=local_mode,
        client=MagicMock(),
        memory=memory,
        _db_available=_db_available,
    )
    return inst


# ---------------------------------------------------------------------------
# 1. Core poisoning scenario: local_mode=True cached → local_mode=False request
# ---------------------------------------------------------------------------

class TestLocalModeCachePoisoning:
    """Verify that a cached local_mode=True instance is NOT reused for a
    local_mode=False (memory-needed) request."""

    def test_local_mode_true_cached_is_evicted_for_false_request(self) -> None:
        """A cached local-mode instance must be discarded when a memory-
        enabled request arrives for the same (model, bot, user)."""
        svc = _make_service()
        poisoned = _fake_instance(local_mode=True)
        cache_key = ("test-model", "byte", "nick", False)
        svc._llm_bawt_cache[(*cache_key[:3], True)] = poisoned

        with patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls:
            fresh = _fake_instance(local_mode=False)
            MockCls.return_value = fresh

            result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)

        # Must NOT return the poisoned instance
        assert result is not poisoned
        # Must have constructed a new ServiceLLMBawt with local_mode=False
        MockCls.assert_called_once()
        call_kwargs = MockCls.call_args
        assert call_kwargs.kwargs.get("local_mode") is False or (
            not call_kwargs.kwargs.get("local_mode", True)
        )
        # New instance is now cached
        assert svc._llm_bawt_cache[cache_key] is fresh

    def test_local_mode_false_cached_reused_for_false_request(self) -> None:
        """A memory-enabled cached instance is reused for memory-enabled requests."""
        svc = _make_service()
        good = _fake_instance(local_mode=False)
        cache_key = ("test-model", "byte", "nick", False)
        svc._llm_bawt_cache[cache_key] = good

        result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)

        assert result is good  # cache hit

    def test_local_mode_false_cached_isolated_from_true_request(self) -> None:
        """Disabling augmentation must never borrow a warm memory-enabled object."""
        svc = _make_service()
        good = _fake_instance(local_mode=False)
        cache_key = ("test-model", "byte", "nick", False)
        svc._llm_bawt_cache[cache_key] = good
        local = _fake_instance(local_mode=True)
        with patch("llm_bawt.service.core.ServiceLLMBawt", return_value=local) as build:
            result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=True)
        assert result is local
        assert result.memory is None
        assert build.call_args.kwargs["local_mode"] is True
        assert svc._llm_bawt_cache[cache_key] is good
        assert svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False) is good
        assert svc._get_llm_bawt("test-model", "byte", "nick", local_mode=True) is local

    def test_local_mode_true_cached_reused_for_true_request(self) -> None:
        """A local-mode cached instance is fine for another local-mode request."""
        svc = _make_service()
        local = _fake_instance(local_mode=True)
        cache_key = ("test-model", "byte", "nick", True)
        svc._llm_bawt_cache[cache_key] = local

        result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=True)

        assert result is local  # cache hit


# ---------------------------------------------------------------------------
# 2. Full incident replay: reload → poison → recovery
# ---------------------------------------------------------------------------

class TestReloadPoisonRecovery:
    """End-to-end replay of the cache-poisoning incident."""

    def test_reload_then_no_memory_then_memory_request(self) -> None:
        """Simulates: reload-bots → augment_memory=false → augment_memory=true."""
        svc = _make_service()

        # Pre-existing good instance
        good_original = _fake_instance(local_mode=False)
        svc._llm_bawt_cache[("test-model", "byte", "nick", False)] = good_original

        # Step 1: reload-bots nukes cache
        svc.invalidate_all_instances()
        assert svc._llm_bawt_cache == {}

        # Step 2: First request arrives with augment_memory=false (local_mode=True)
        with patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls:
            no_mem_instance = _fake_instance(local_mode=True)
            MockCls.return_value = no_mem_instance
            result1 = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=True)

        assert result1 is no_mem_instance
        assert result1.local_mode is True

        # Step 3: Next request with augment_memory=true (local_mode=False) — THE FIX
        with patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls:
            mem_instance = _fake_instance(local_mode=False)
            MockCls.return_value = mem_instance
            result2 = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)

        # Fix must discard the poisoned instance and create a new one
        assert result2 is not no_mem_instance
        assert result2 is mem_instance
        assert result2.local_mode is False

        # Step 4: Subsequent memory-enabled requests should get the good instance
        result3 = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)
        assert result3 is mem_instance  # cache hit


# ---------------------------------------------------------------------------
# 3. Cross-bot isolation: poison for one bot must not affect another
# ---------------------------------------------------------------------------

class TestCrossBotIsolation:
    """Ensure local_mode poisoning in one bot doesn't affect others."""

    def test_poisoned_byte_does_not_affect_nova(self) -> None:
        svc = _make_service()
        poisoned_byte = _fake_instance(local_mode=True)
        good_nova = _fake_instance(local_mode=False)
        svc._llm_bawt_cache[("test-model", "byte", "nick", True)] = poisoned_byte
        svc._llm_bawt_cache[("test-model", "nova", "nick", False)] = good_nova

        # nova request should still get the good instance
        result = svc._get_llm_bawt("test-model", "nova", "nick", local_mode=False)
        assert result is good_nova

    def test_sync_soul_invalidates_only_target_bot(self) -> None:
        """invalidate_bot_instances only evicts the named bot."""
        svc = _make_service()
        svc._llm_bawt_cache[("test-model", "snark", "nick", False)] = _fake_instance(local_mode=False)
        svc._llm_bawt_cache[("test-model", "byte", "nick", False)] = _fake_instance(local_mode=False)

        removed = svc.invalidate_bot_instances("snark")
        assert removed == 1
        assert ("test-model", "byte", "nick", False) in svc._llm_bawt_cache
        assert ("test-model", "snark", "nick", False) not in svc._llm_bawt_cache


# ---------------------------------------------------------------------------
# 4. Edge case: missing local_mode attribute on cached instance
# ---------------------------------------------------------------------------

class TestMissingLocalModeAttribute:
    """Old cached instances may lack a local_mode attribute."""

    def test_missing_attribute_defaults_false(self) -> None:
        """getattr(cached, 'local_mode', False) should treat missing attr as
        non-local (memory-enabled), so no eviction for memory requests."""
        svc = _make_service()
        old = SimpleNamespace(client=MagicMock(), memory=MagicMock(), _db_available=True)  # no local_mode attr
        svc._llm_bawt_cache[("test-model", "byte", "nick", False)] = old

        result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)
        # Should be treated as local_mode=False → reused
        assert result is old


# ---------------------------------------------------------------------------
# 5. Failed memory init: non-local instance with memory=None, _db_available=False
# ---------------------------------------------------------------------------

class TestFailedMemoryInitEviction:
    """A non-local instance whose memory init raised an exception is cached
    with memory=None and _db_available=False.  It must be evicted when a
    non-local request arrives and DB credentials are now available."""

    def test_memory_failed_instance_evicted_when_db_available(self) -> None:
        """Cached non-local instance with memory=None/_db_available=False
        is discarded when has_database_credentials returns True."""
        svc = _make_service()
        broken = _fake_instance(local_mode=False, memory=None, _db_available=False)
        cache_key = ("test-model", "snark", "nick", False)
        svc._llm_bawt_cache[cache_key] = broken

        with (
            patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls,
            patch("llm_bawt.utils.config.has_database_credentials", return_value=True),
        ):
            fresh = _fake_instance(local_mode=False)
            MockCls.return_value = fresh
            result = svc._get_llm_bawt("test-model", "snark", "nick", local_mode=False)

        assert result is not broken
        assert result is fresh
        MockCls.assert_called_once()

    def test_memory_failed_instance_reused_when_no_db_credentials(self) -> None:
        """If DB credentials are still absent, don't churn — reuse the
        cached instance (there's nothing to retry)."""
        svc = _make_service()
        broken = _fake_instance(local_mode=False, memory=None, _db_available=False)
        cache_key = ("test-model", "snark", "nick", False)
        svc._llm_bawt_cache[cache_key] = broken

        with patch("llm_bawt.utils.config.has_database_credentials", return_value=False):
            result = svc._get_llm_bawt("test-model", "snark", "nick", local_mode=False)

        assert result is broken  # no point retrying

    def test_requires_memory_false_not_evicted(self) -> None:
        """An instance with memory=None but _db_available=True means
        requires_memory=False — this is intentional, not a failure."""
        svc = _make_service()
        no_mem_ok = _fake_instance(local_mode=False, memory=None, _db_available=True)
        cache_key = ("test-model", "spark", "nick", False)
        svc._llm_bawt_cache[cache_key] = no_mem_ok

        result = svc._get_llm_bawt("test-model", "spark", "nick", local_mode=False)

        assert result is no_mem_ok  # intentionally no memory — keep it

    def test_healthy_instance_not_evicted(self) -> None:
        """A normal instance with working memory is never evicted."""
        svc = _make_service()
        good = _fake_instance(local_mode=False, memory=MagicMock(), _db_available=True)
        cache_key = ("test-model", "byte", "nick", False)
        svc._llm_bawt_cache[cache_key] = good

        result = svc._get_llm_bawt("test-model", "byte", "nick", local_mode=False)

        assert result is good

    def test_reload_then_memory_failure_then_recovery(self) -> None:
        """Full incident replay: reload → memory init fails → next request recovers."""
        svc = _make_service()

        # Step 1: reload clears cache
        svc.invalidate_all_instances()

        # Step 2: Instance created but memory init fails (DB down)
        with patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls:
            broken = _fake_instance(local_mode=False, memory=None, _db_available=False)
            MockCls.return_value = broken
            result1 = svc._get_llm_bawt("test-model", "snark", "nick", local_mode=False)

        assert result1 is broken
        assert result1.memory is None

        # Step 3: Next request — DB is back, broken instance should be evicted
        with (
            patch("llm_bawt.service.core.ServiceLLMBawt") as MockCls,
            patch("llm_bawt.utils.config.has_database_credentials", return_value=True),
        ):
            recovered = _fake_instance(local_mode=False)
            MockCls.return_value = recovered
            result2 = svc._get_llm_bawt("test-model", "snark", "nick", local_mode=False)

        assert result2 is not broken
        assert result2 is recovered
        assert result2.memory is not None
