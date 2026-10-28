from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from llm_bawt.service import dependencies
from llm_bawt.service.instance_manager import InstanceManagerMixin
from llm_bawt.service.routes import media as media_routes

HANG_GUARD_S = 10.0
WORKERS = 12


class _ContendedLock:
    """Cache lock that holds the first build until every caller is queued on it.

    Each caller counts itself before it blocks on the real lock. The build
    (which runs with the lock held) waits until ``expect(n)`` callers have
    arrived, so the single-flight check runs while every one of them has
    already missed the lock-free fast path, instead of hoping a sleep is
    long enough for them to get there.
    """

    def __init__(self, lock) -> None:
        self._lock = lock
        self._cond = threading.Condition()
        self._arrived = 0
        self._expected = 0

    def expect(self, callers: int) -> None:
        with self._cond:
            self._arrived = 0
            self._expected = callers

    def __enter__(self):
        with self._cond:
            self._arrived += 1
            self._cond.notify_all()
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)

    def hold_until_contended(self) -> None:
        with self._cond:
            if not self._expected:
                return
            expected = self._expected
            self._expected = 0
            assert self._cond.wait_for(lambda: self._arrived >= expected, HANG_GUARD_S), (
                f"only {self._arrived} of {expected} callers reached the cache lock"
            )


@pytest.fixture(autouse=True)
def reset_dependencies():
    dependencies.reset_store_cache()
    dependencies.set_service(None)
    yield
    dependencies.reset_store_cache()
    dependencies.set_service(None)


def test_media_accessors_cache_by_config_identity(monkeypatch) -> None:
    from llm_bawt.media import assets, db

    monkeypatch.setattr(assets, "MediaAssetStore", lambda config: ("asset", config))
    monkeypatch.setattr(
        db, "MediaGenerationStore", lambda config: ("generation", config)
    )
    first_config = object()
    second_config = object()

    first_asset = dependencies.get_media_asset_store(first_config)
    first_generation = dependencies.get_media_generation_store(first_config)

    assert dependencies.get_media_asset_store(first_config) is first_asset
    assert dependencies.get_media_generation_store(first_config) is first_generation
    assert dependencies.get_media_asset_store(second_config) is not first_asset
    assert dependencies.get_media_generation_store(second_config) is not first_generation


def test_turn_log_accessor_returns_service_owned_writer() -> None:
    canonical = object()
    dependencies.set_service(SimpleNamespace(_turn_log_store=canonical))
    assert dependencies.get_turn_log_store() is canonical


def test_startup_warm_and_media_route_resolve_same_store(monkeypatch) -> None:
    from llm_bawt.media import db

    built: list[object] = []

    def factory(config):
        store = SimpleNamespace(config=config)
        built.append(store)
        return store

    monkeypatch.setattr(db, "MediaGenerationStore", factory)
    config = object()
    warmed = dependencies.get_media_generation_store(config)
    dependencies.set_service(SimpleNamespace(config=config))

    assert media_routes._get_store() is warmed
    assert built == [warmed]


def test_concurrent_media_store_access_constructs_once(monkeypatch) -> None:
    from llm_bawt.media import assets

    constructed = 0
    constructed_lock = threading.Lock()
    cache_lock = _ContendedLock(dependencies._store_lock)

    def factory(config):
        nonlocal constructed
        cache_lock.hold_until_contended()
        with constructed_lock:
            constructed += 1
        return SimpleNamespace(config=config)

    monkeypatch.setattr(assets, "MediaAssetStore", factory)
    monkeypatch.setattr(dependencies, "_store_lock", cache_lock)
    config = object()
    cache_lock.expect(WORKERS)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        stores = list(
            pool.map(
                lambda _: dependencies.get_media_asset_store(config),
                range(24),
            )
        )

    assert constructed == 1
    assert all(store is stores[0] for store in stores)


def test_nested_store_factory_does_not_deadlock(monkeypatch) -> None:
    """A factory may resolve a sibling singleton through the same helper.

    ``get_ops_service``'s factory calls ``get_ops_store``, so the build runs
    while the store lock is already held. A non-reentrant lock deadlocked the
    calling thread here — and because the MCP ops tools await these accessors
    on the event loop, that wedged the whole MCP server until restart. A warm
    inner cache hides it, so the cold path is the one worth pinning.
    """
    # Patched where the accessors look them up: dependencies imports both at
    # module level so the continuation outbox never imports on first use.
    monkeypatch.setattr(dependencies, "OpsStore", lambda config: ("store", config))
    monkeypatch.setattr(dependencies, "build_admission_ops_service", lambda config, store: ("service", store))
    config = object()

    done = threading.Event()
    result: list[object] = []

    def build() -> None:
        # Cold cache: the inner get_ops_store cannot take the lock-free path.
        result.append(dependencies.get_ops_service(config))
        done.set()

    worker = threading.Thread(target=build, daemon=True)
    worker.start()
    assert done.wait(timeout=10), "get_ops_service deadlocked on the store lock"
    assert result[0] == ("service", ("store", config))
    # Both singletons must be cached, and the sibling reused rather than rebuilt.
    assert dependencies.get_ops_service(config) is result[0]
    assert dependencies.get_ops_store(config) is result[0][1]


def test_memory_client_cache_is_single_flight_and_keeps_user_dimension(
    monkeypatch,
) -> None:
    from llm_bawt.mcp_server import client as client_module

    constructed: list[tuple[str, str | None]] = []
    constructed_lock = threading.Lock()
    cache_lock = _ContendedLock(threading.Lock())

    def factory(*, config, bot_id, user_id, server_url):
        cache_lock.hold_until_contended()
        with constructed_lock:
            constructed.append((bot_id, user_id))
        return SimpleNamespace(
            config=config,
            bot_id=bot_id,
            user_id=user_id,
            server_url=server_url,
        )

    monkeypatch.setattr(client_module, "get_memory_client", factory)
    manager = InstanceManagerMixin()
    manager.config = SimpleNamespace(MCP_SERVER_URL=None)
    manager._memory_clients = {}
    manager._memory_clients_lock = cache_lock

    cache_lock.expect(WORKERS)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        clients = list(
            pool.map(
                lambda _: manager.get_memory_client("nova", "user-a"),
                range(24),
            )
        )

    other_user = manager.get_memory_client("nova", "user-b")
    assert all(client is clients[0] for client in clients)
    assert other_user is not clients[0]
    assert constructed == [("nova", "user-a"), ("nova", "user-b")]


def test_memory_storage_caches_are_instance_owned_and_single_flight(
    monkeypatch,
) -> None:
    from llm_bawt.mcp_server import storage as storage_module
    from llm_bawt.memory import postgresql

    backend_builds: list[str] = []
    manager_builds: list[str] = []
    cache_lock = _ContendedLock(threading.Lock())

    def backend_factory(*, config, bot_id, provision_schema=False):
        cache_lock.hold_until_contended()
        backend_builds.append(bot_id)
        return SimpleNamespace(
            config=config,
            bot_id=bot_id,
            ensure_schema=lambda: None,
            provision_schema=provision_schema,
        )

    def manager_factory(*, config, bot_id, provision_schema=False):
        cache_lock.hold_until_contended()
        manager_builds.append(bot_id)
        return SimpleNamespace(
            config=config,
            bot_id=bot_id,
            ensure_schema=lambda: None,
            provision_schema=provision_schema,
        )

    monkeypatch.setattr(storage_module, "PostgreSQLMemoryBackend", backend_factory)
    monkeypatch.setattr(postgresql, "PostgreSQLShortTermManager", manager_factory)
    first = storage_module.MemoryStorage(config=SimpleNamespace())
    first._cache_lock = cache_lock

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        cache_lock.expect(WORKERS)
        backends = list(pool.map(lambda _: first.get_backend("nova"), range(16)))
        cache_lock.expect(WORKERS)
        managers = list(
            pool.map(lambda _: first.get_short_term_manager("nova"), range(16))
        )

    second = storage_module.MemoryStorage(config=SimpleNamespace())
    second_backend = second.get_backend("nova")
    second_manager = second.get_short_term_manager("nova")

    assert all(value is backends[0] for value in backends)
    assert all(value is managers[0] for value in managers)
    assert second_backend is not backends[0]
    assert second_manager is not managers[0]
    assert backend_builds == ["nova", "nova"]
    assert manager_builds == ["nova", "nova"]
