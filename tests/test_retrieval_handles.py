"""TASK-1323 §5.3: owner-bound, expiring handles with a compare-and-swap update."""
from __future__ import annotations

import pytest

from llm_bawt.retrieval.handles import HandleExpired, HandleStore

A, B = ("snark", "nick"), ("byte", "nick")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def store(clock, **kwargs):
    return HandleStore(clock=clock, wall=clock, **kwargs)


def test_owner_bound(clock):
    s = store(clock)
    key = s.new_key()
    s.put(key, A, "p")
    assert s.get(key, A) == "p"
    for owner in (B, ("snark", "eve")):
        with pytest.raises(HandleExpired):
            s.get(key, owner)
    with pytest.raises(HandleExpired):
        s.put(key, B, "stolen")
    assert s.get(key, A) == "p"


def test_unknown_and_non_string_keys_read_as_expired(clock):
    s = store(clock)
    for key in ("nope", None, 7):
        with pytest.raises(HandleExpired):
            s.get(key, A)


def test_ttl_and_a_later_page_never_extends_it(clock):
    s = store(clock, ttl=60)
    key = s.new_key()
    s.put(key, A, 1)
    clock.now += 50
    s.put(key, A, 2)
    clock.now += 11
    with pytest.raises(HandleExpired):
        s.get(key, A)


def test_capability_expiry_caps_the_ttl(clock):
    s = store(clock, ttl=900)
    key = s.new_key()
    s.put(key, A, 1, capability_expires_at=clock.now + 30)
    clock.now += 31
    with pytest.raises(HandleExpired):
        s.get(key, A)
    with pytest.raises(HandleExpired):
        s.put(s.new_key(), A, 1, capability_expires_at=clock.now - 1)


def test_per_owner_and_global_capacity_evict_oldest(clock):
    s = store(clock, per_owner=2, capacity=3)
    k1, k2, k3 = (s.new_key() for _ in range(3))
    for key in (k1, k2, k3):
        s.put(key, A, key)
    with pytest.raises(HandleExpired):
        s.get(k1, A)
    assert s.get(k3, A) == k3
    kb1, kb2 = s.new_key(), s.new_key()
    s.put(kb1, B, 1)
    s.put(kb2, B, 2)
    with pytest.raises(HandleExpired):
        s.get(k2, A)
    assert s.get(kb1, B) == 1


def test_replace_is_compare_and_swap(clock):
    s = store(clock, ttl=60)
    key = s.new_key()
    first, second = object(), object()
    s.put(key, A, first)
    assert not s.replace(key, B, first, second)
    assert not s.replace(key, A, object(), second)
    clock.now += 30
    assert s.replace(key, A, first, second)
    assert s.get(key, A) is second
    assert not s.replace(key, A, first, object())  # the loser of a race
    clock.now += 31
    with pytest.raises(HandleExpired):
        s.get(key, A)
    assert not s.replace(key, A, second, first)
