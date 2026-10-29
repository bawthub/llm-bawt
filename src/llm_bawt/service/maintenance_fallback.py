"""Backup maintenance model on a provider spending limit (TASK-1704 slice B).

Design: ``docs/MAINTENANCE_MODEL_FALLBACK.md``.

When a settings-resolved maintenance call's provider answers ``spending_limit``
(slice A's classifier), the same call is replayed once on the user-chosen
``maintenance_model_fallback``, and the provider is marked exhausted in Redis so
later calls skip it until a cooldown passes. After the cooldown exactly one call
(the probe) touches the primary again; it never costs its job, because a probe
that hits the limit again falls through to the backup in the same call and
re-arms the cooldown. Nothing else (rate limits, auth, timeouts…) ever switches.

The bell is pull-based: BawtHub polls ``/v1/health/maintenance``. The switched
state is read from the durable Redis record on every poll, so a switch notice
can't be lost to a crash or a failed send; it shows for the whole exhausted
period and turns back to ok (with "back on primary since …") on switch-back.

No model name lives here: the backup is a DB runtime setting, and an unset,
unknown, same-provider or incapable backup fails visibly instead of being
substituted.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from .logging import get_service_logger
from .maintenance_health import (
    SPENDING_LIMIT, MaintenanceHealth, MonitoredClient, classify_failure, maintenance_health,
)

log = get_service_logger(__name__)

FALLBACK_SETTING = "maintenance_model_fallback"
COOLDOWN_SETTING = "maintenance_fallback_cooldown_seconds"
INCOMPATIBLE = "fallback_incompatible"
STATE_UNAVAILABLE = "fallback_state_unavailable"
STATE_JOB = "maintenance_fallback_state"

_KEY = "maintenance:fallback:{provider}"
_PROBE = "maintenance:fallback:{provider}:probe"
_RECOVERED = "maintenance:fallback:{provider}:recovered"
# Per-provider edge counter, no TTL: every switch and recovery takes the next
# value, so two edges can never share a bell incident id however fast. Redis may
# still drop it (allkeys-lru eviction, a restart without persistence), so a
# missing counter is re-seeded from Redis TIME in us (_NEXT_EDGE): it only moves
# forward and never reuses an id the user already dismissed.
_EDGE = "maintenance:fallback:{provider}:edge"
# A probe is one real call: its claim must outlive that call, not the cooldown,
# so a crashed probe frees the primary quickly.
PROBE_CLAIM_SECONDS = 300
# How long the "back on primary" row stays visible after recovery.
RECOVERED_ROW_SECONDS = 24 * 3600


class StateUnavailable(RuntimeError):
    """The shared exhaustion state can't be read or written (Redis down)."""


@dataclass
class Exhaustion:
    provider: str
    since: float
    until: float
    hint_source: str
    backup: str | None = None
    edge: int = 0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


@dataclass
class Episode:
    """Recovered spending-limit episodes kept for the notice window."""

    provider: str
    first_since: float
    recovered_at: float
    episodes: int
    edge: int = 0


class ExhaustionStore(ABC):
    """Per-provider exhausted state shared by every worker and restart.

    ``mark`` creates the record (switched). After that only the current probe
    holder may change it: ``claim_probe`` returns a token, and ``rearm`` /
    ``clear`` act only while the record's ``since`` matches and the probe key
    still holds that token, so a probe whose lease expired changes nothing.
    """

    @abstractmethod
    def get(self, provider: str) -> Exhaustion | None: ...

    @abstractmethod
    def mark(self, record: Exhaustion) -> bool:
        """Create the record if absent, stamped with the next edge number. True = this caller made the switch."""

    @abstractmethod
    def claim_probe(self, provider: str, ttl_seconds: int) -> str | None:
        """Token for the single probe holder; the claim expires if the probe crashes."""

    @abstractmethod
    def rearm(self, provider: str, since: float, token: str, until: float) -> bool:
        """Failed probe: move ``until`` and release the probe, if still the holder."""

    @abstractmethod
    def clear(self, provider: str, since: float, token: str) -> bool:
        """Successful probe: delete the record and record the episode (next edge number), if still the holder."""

    @abstractmethod
    def all(self) -> list[Exhaustion]: ...

    @abstractmethod
    def episodes(self) -> list[Episode]: ...


# Next edge number as a decimal string; a lost counter restarts at the current
# microsecond, above any value it held, since edges are far rarer than one per
# microsecond. Strings throughout: Lua's tostring and cjson keep only 14
# significant digits and would round a 16-digit value.
_NEXT_EDGE = """
local function next_edge(key)
  if redis.call('EXISTS', key) == 0 then
    local t = redis.call('TIME')
    redis.call('SET', key, t[1] .. string.format('%06d', tonumber(t[2])))
  end
  return string.format('%d', redis.call('INCR', key))
end
"""

# Create the record only if absent, stamped with the next edge. KEYS: record, edge.
_MARK = _NEXT_EDGE + """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
local rec = cjson.decode(ARGV[1])
rec['edge'] = next_edge(KEYS[2])
redis.call('SET', KEYS[1], cjson.encode(rec))
return 1
"""

# Fenced on the record's ``since`` (ms) and the probe token. KEYS: record, probe[, episode, edge].
_REARM = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local rec = cjson.decode(raw)
if string.format('%d', rec['since']) ~= ARGV[1] then return 0 end
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
rec['until'] = tonumber(ARGV[3])
redis.call('SET', KEYS[1], cjson.encode(rec))
redis.call('DEL', KEYS[2])
return 1
"""

_CLEAR = _NEXT_EDGE + """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local rec = cjson.decode(raw)
if string.format('%d', rec['since']) ~= ARGV[1] then return 0 end
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
local ep = {first_since = rec['since'], recovered_at = tonumber(ARGV[3]), episodes = 1,
            edge = next_edge(KEYS[4])}
local old = redis.call('GET', KEYS[3])
if old then
  local prev = cjson.decode(old)
  if prev['first_since'] < ep['first_since'] then ep['first_since'] = prev['first_since'] end
  ep['episodes'] = prev['episodes'] + 1
end
redis.call('DEL', KEYS[1], KEYS[2])
redis.call('SET', KEYS[3], cjson.encode(ep), 'EX', tonumber(ARGV[4]))
return 1
"""


def _ms(value: float) -> int:
    return int(round(value * 1000))


class RedisExhaustionStore(ExhaustionStore):
    """Records have no TTL: they outlive the cooldown and only a fenced clear removes them."""

    def __init__(self, url: str) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, decode_responses=True,
                                           socket_timeout=2.0, socket_connect_timeout=2.0)
        self._rearm = self._redis.register_script(_REARM)
        self._clear = self._redis.register_script(_CLEAR)
        self._mark = self._redis.register_script(_MARK)

    def _run(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - any Redis failure means "state unavailable"
            raise StateUnavailable(type(error).__name__) from None

    @staticmethod
    def _load(raw: str | None) -> Exhaustion | None:
        if not raw:
            return None
        data = json.loads(raw)
        return Exhaustion(provider=data["provider"], since=data["since"] / 1000,
                          until=data["until"], hint_source=data["hint_source"],
                          backup=data.get("backup"), edge=int(data.get("edge", 0)))

    def get(self, provider: str) -> Exhaustion | None:
        return self._load(self._run(lambda: self._redis.get(_KEY.format(provider=provider))))

    def mark(self, record: Exhaustion) -> bool:
        payload = asdict(record) | {"since": _ms(record.since)}
        return bool(self._run(lambda: self._mark(
            keys=[_KEY.format(provider=record.provider), _EDGE.format(provider=record.provider)],
            args=[json.dumps(payload)])))

    def claim_probe(self, provider: str, ttl_seconds: int) -> str | None:
        token = uuid.uuid4().hex
        won = self._run(lambda: self._redis.set(
            _PROBE.format(provider=provider), token, nx=True, ex=max(1, int(ttl_seconds))))
        return token if won else None

    def rearm(self, provider: str, since: float, token: str, until: float) -> bool:
        return bool(self._run(lambda: self._rearm(
            keys=[_KEY.format(provider=provider), _PROBE.format(provider=provider)],
            args=[_ms(since), token, until])))

    def clear(self, provider: str, since: float, token: str) -> bool:
        return bool(self._run(lambda: self._clear(
            keys=[_KEY.format(provider=provider), _PROBE.format(provider=provider),
                  _RECOVERED.format(provider=provider), _EDGE.format(provider=provider)],
            args=[_ms(since), token, time.time(), RECOVERED_ROW_SECONDS])))

    def all(self) -> list[Exhaustion]:
        def scan() -> list[Exhaustion]:
            keys = [k for k in self._redis.scan_iter(match=_KEY.format(provider="*"))
                    if k.count(":") == 2]
            return [r for r in (self._load(v) for v in (self._redis.mget(keys) if keys else [])) if r]
        return self._run(scan)

    def episodes(self) -> list[Episode]:
        def scan() -> list[Episode]:
            keys = list(self._redis.scan_iter(match=_RECOVERED.format(provider="*")))
            values = self._redis.mget(keys) if keys else []
            out = []
            for key, raw in zip(keys, values):
                if raw:
                    data = json.loads(raw)
                    out.append(Episode(provider=key.split(":")[2],
                                       first_since=data["first_since"] / 1000,
                                       recovered_at=float(data["recovered_at"]),
                                       episodes=int(data["episodes"]),
                                       edge=int(data.get("edge", 0))))
            return out
        return self._run(scan)


class MemoryExhaustionStore(ExhaustionStore):
    """Same semantics in one process; used by tests."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, Exhaustion] = {}
        self._probes: dict[str, tuple[str, float]] = {}
        self._episodes: dict[str, Episode] = {}
        self._edges: dict[str, int] = {}
        self._clock = clock
        self.down = False

    def _check(self) -> None:
        if self.down:
            raise StateUnavailable("ConnectionError")

    def _holder(self, provider: str, since: float, token: str) -> Exhaustion | None:
        record = self._records.get(provider)
        probe = self._probes.get(provider)
        if record is None or record.since != since or probe is None:
            return None
        if probe[0] != token or probe[1] <= self._clock():
            return None
        return record

    def _next_edge(self, provider: str) -> int:
        # Same as _NEXT_EDGE: a lost counter restarts from the clock in microseconds.
        self._edges[provider] = self._edges.get(provider, int(self._clock() * 1_000_000)) + 1
        return self._edges[provider]

    def get(self, provider: str) -> Exhaustion | None:
        self._check()
        with self._lock:
            record = self._records.get(provider)
            return Exhaustion(**asdict(record)) if record else None

    def mark(self, record: Exhaustion) -> bool:
        self._check()
        with self._lock:
            if record.provider in self._records:
                return False
            stored = Exhaustion(**asdict(record))
            stored.edge = self._next_edge(record.provider)
            self._records[record.provider] = stored
            return True

    def claim_probe(self, provider: str, ttl_seconds: int) -> str | None:
        self._check()
        with self._lock:
            now = self._clock()
            held = self._probes.get(provider)
            if held is not None and held[1] > now:
                return None
            token = uuid.uuid4().hex
            self._probes[provider] = (token, now + max(1, int(ttl_seconds)))
            return token

    def rearm(self, provider: str, since: float, token: str, until: float) -> bool:
        self._check()
        with self._lock:
            record = self._holder(provider, since, token)
            if record is None:
                return False
            record.until = until
            del self._probes[provider]
            return True

    def clear(self, provider: str, since: float, token: str) -> bool:
        self._check()
        with self._lock:
            record = self._holder(provider, since, token)
            if record is None:
                return False
            now = self._clock()
            prev = self._live_episode(provider)
            self._episodes[provider] = Episode(
                provider=provider,
                first_since=min(record.since, prev.first_since) if prev else record.since,
                recovered_at=now, episodes=(prev.episodes + 1) if prev else 1,
                edge=self._next_edge(provider))
            del self._records[provider]
            del self._probes[provider]
            return True

    def _live_episode(self, provider: str) -> Episode | None:
        episode = self._episodes.get(provider)
        if episode and episode.recovered_at + RECOVERED_ROW_SECONDS > self._clock():
            return episode
        return None

    def all(self) -> list[Exhaustion]:
        self._check()
        with self._lock:
            return [Exhaustion(**asdict(r)) for r in self._records.values()]

    def episodes(self) -> list[Episode]:
        self._check()
        with self._lock:
            return [Episode(**asdict(e)) for p in list(self._episodes)
                    if (e := self._live_episode(p))]


def reset_hint_seconds(error: Any, *, now: float | None = None) -> float | None:
    """Provider reset hint (``Retry-After`` seconds or HTTP date) when the error carries one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - odd header containers
        return None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(str(value)).timestamp()
    except (TypeError, ValueError):
        return None
    return max(0.0, when - (now if now is not None else time.time()))


class StateHealth:
    """One alert while the shared exhaustion state can't be reached."""

    def __init__(self, health: MaintenanceHealth) -> None:
        self._health = health
        self._lock = threading.Lock()
        self._down = False

    def unavailable(self, reason: str) -> None:
        with self._lock:
            first = not self._down
            self._down = True
        if first:
            log.error("Maintenance fallback state unavailable (%s): per-call fallback only", reason)
        self._health.record_failure(STATE_JOB, STATE_UNAVAILABLE,
                                    f"exhaustion state unavailable ({reason}); per-call fallback only",
                                    model=None, provider=None, credential=None)

    def available(self) -> None:
        with self._lock:
            was_down, self._down = self._down, False
        if was_down:
            self._health.record_success(STATE_JOB, model=None, provider=None, credential=None)


@dataclass
class Backup:
    """A resolved, validated backup: its monitored client and labels."""

    client: MonitoredClient
    model: str
    provider: str


class FallbackClient:
    """Wraps the primary ``MonitoredClient``; only ``spending_limit`` reaches the backup.

    ``last_served_model`` / ``last_served_provider`` / ``last_via_fallback`` say
    which model answered the most recent call, for callers that report it.
    """

    def __init__(self, primary: MonitoredClient, *, job: str, model: str, provider: str,
                 open_backup: Callable[[str], Backup | None], store: ExhaustionStore,
                 cooldown_seconds: Callable[[], float], state_health: StateHealth,
                 on_incompatible: Callable[[str | None, str], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._primary = primary
        self._job = job
        self._model = model
        self._provider = provider
        self._open_backup = open_backup
        self._store = store
        self._cooldown = cooldown_seconds
        self._state = state_health
        self._on_incompatible = on_incompatible or (lambda _alias, _why: None)
        self._clock = clock
        self.last_served_model: str | None = model
        self.last_served_provider: str | None = provider
        self.last_via_fallback = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)

    @property
    def wrapped(self) -> Any:
        return self._primary.wrapped

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("query", *args, **kwargs)

    def query_bounded(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("query_bounded", *args, **kwargs)

    # -- state ---------------------------------------------------------------
    def _state_op(self, fn: Callable[[], Any], default: Any) -> tuple[Any, bool]:
        """(result, available). Redis down → per-call mode with one alert."""
        try:
            result = fn()
        except StateUnavailable as error:
            self._state.unavailable(str(error))
            return default, False
        self._state.available()
        return result, True

    def _until(self, error: Any) -> tuple[float, str]:
        now = self._clock()
        hint = reset_hint_seconds(error, now=now)
        if hint is not None:
            return now + hint, "provider"
        return now + float(self._cooldown()), "setting"

    # -- call ----------------------------------------------------------------
    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        record, stateful = self._state_op(lambda: self._store.get(self._provider), None)
        probing = None
        if record is not None:
            if self._clock() < record.until:
                return self._via_backup(method, None, args, kwargs)
            token, claimed = self._state_op(
                lambda: self._store.claim_probe(self._provider, PROBE_CLAIM_SECONDS), None)
            if not claimed:
                # Redis failed mid-call: stateless per-call mode (primary, then backup).
                stateful = False
            elif not token:
                return self._via_backup(method, None, args, kwargs)
            else:
                probing = (record, token)

        try:
            result = getattr(self._primary, method)(*args, **kwargs)
        except Exception as error:
            if classify_failure(error) != SPENDING_LIMIT:
                raise
            return self._spending_limit(error, probing, stateful, method, args, kwargs)
        if isinstance(result, str) and result.lstrip().startswith("ERROR:") \
                and classify_failure(result) == SPENDING_LIMIT:
            return self._spending_limit(result, probing, stateful, method, args, kwargs)
        self._served(self._model, self._provider, False)
        if probing is not None:
            switched_back, _ = self._state_op(
                lambda: self._store.clear(self._provider, probing[0].since, probing[1]), False)
            if switched_back:
                log.warning("Maintenance: %s spending limit cleared, back on primary %s",
                            self._provider, self._model)
        return result

    def _spending_limit(self, error: Any, probing: tuple[Exhaustion, str] | None, stateful: bool,
                        method: str, args: tuple, kwargs: dict) -> Any:
        until, source = self._until(error)
        if probing is not None:
            self._state_op(lambda: self._store.rearm(
                self._provider, probing[0].since, probing[1], until), False)
        elif stateful:
            record = Exhaustion(provider=self._provider, since=self._clock(), until=until,
                                hint_source=source)
            switched, _ = self._state_op(lambda: self._store.mark(record), False)
            if switched:
                log.warning("Maintenance: %s spending limit, switching to backup until %s (%s)",
                            self._provider, _iso(until), source)
        return self._via_backup(method, error, args, kwargs)

    def _via_backup(self, method: str, primary_error: Any, args: tuple, kwargs: dict) -> Any:
        backup = self._open_backup(self._provider)
        if backup is not None and not callable(getattr(backup.client.wrapped, method, None)):
            self._on_incompatible(backup.model, f"cannot serve {method}")
            backup = None
        if backup is None:
            # Visible: open_backup recorded why. The primary's outcome stands.
            if primary_error is None:
                raise RuntimeError(f"{self._provider} is exhausted and no usable backup maintenance model is configured")
            if isinstance(primary_error, BaseException):
                raise primary_error
            return primary_error
        self._served(backup.model, backup.provider, True)
        return getattr(backup.client, method)(*args, **kwargs)

    def _served(self, model: str, provider: str, via_fallback: bool) -> None:
        self.last_served_model = model
        self.last_served_provider = provider
        self.last_via_fallback = via_fallback


def served_model(client: Any, default: str | None) -> str | None:
    """Model that answered the client's last call (backup when it served)."""
    value = getattr(client, "last_served_model", None) if isinstance(client, FallbackClient) else None
    return value or default


def fallback_checks(store: ExhaustionStore) -> list[dict[str, Any]]:
    """Bell rows from durable state, one per provider, never ``ok``.

    The bell only shows non-ok rows, so both "on backup" and "was on backup"
    (the episode, kept ``RECOVERED_ROW_SECONDS``) are ``degraded``. BawtHub
    identifies a bell incident by child ``name:status:episode``; ``episode`` is
    the provider's edge number (switch for a live record, latest recovery for an
    episode), taken from a counter that never repeats, so every edge is a new
    notice however close together, and polls of unchanged state keep the same id.
    A live record wins over an episode for the same provider.
    """
    rows: dict[str, dict[str, Any]] = {}
    for episode in sorted(store.episodes(), key=lambda e: e.provider):
        count = f"{episode.episodes} episode{'s' if episode.episodes != 1 else ''}"
        rows[episode.provider] = {
            "name": f"maintenance_fallback_recovered:{episode.provider}", "status": "degraded",
            "episode": str(episode.edge),
            "reason": f"{episode.provider} spending limit: maintenance was on the backup model "
                      f"from {_iso(episode.first_since)} to {_iso(episode.recovered_at)} "
                      f"({count}); back on primary",
        }
    for record in sorted(store.all(), key=lambda r: r.provider):
        rows[record.provider] = {
            "name": f"maintenance_fallback:{record.provider}", "status": "degraded",
            "episode": str(record.edge),
            "reason": f"{record.provider} spending limit since {_iso(record.since)}: "
                      f"maintenance on the backup model; next primary probe after "
                      f"{_iso(record.until)}",
        }
    return [rows[p] for p in sorted(rows)]


_store: ExhaustionStore | None = None
_state_health: StateHealth | None = None
_lock = threading.Lock()


def default_store(config: Any) -> ExhaustionStore:
    global _store
    with _lock:
        if _store is None:
            _store = RedisExhaustionStore(str(config.REDIS_URL))
        return _store


def default_state_health() -> StateHealth:
    global _state_health
    with _lock:
        if _state_health is None:
            _state_health = StateHealth(maintenance_health())
        return _state_health
