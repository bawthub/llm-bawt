"""TASK-2590: startup snapshot of the strict reader selection."""

import json
from types import SimpleNamespace

import pytest

from llm_bawt.service import strict_reader_config as cfg
from llm_bawt.service.routes.settings import is_protected_setting_key
from llm_bawt.service.strict_reader_execution import (
    StrictReaderRefused,
    is_strict_request,
    strict_house,
    strict_skill_source,
)
from llm_bawt.setting_definitions import is_global_only_runtime_setting, setting_default

_UNSET = object()


class _Store:
    def __init__(self, value=_UNSET, *, text=None, fail=False):
        self.text = text if text is not None else (None if value is _UNSET else json.dumps(value))
        self.fail = fail

    def get_raw_value_json(self, scope_type, scope_id, key):
        assert (scope_type, scope_id, key) == ("global", "*", "strict_reader")
        if self.fail:
            raise RuntimeError("secret-dsn-should-not-leak")
        return self.text


@pytest.fixture
def with_db(monkeypatch):
    monkeypatch.setattr("llm_bawt.utils.config.has_database_credentials", lambda c=None: True)


def _boot(store):
    service = SimpleNamespace()
    cfg.load_snapshot(object(), lambda: store).apply(service)
    return service


GEN = {"enabled": True, "bot": "reader-a", "house": "house-x",
       "skill_generation": "gen-abc", "skill_layers": {"core": "r1"}}
LEGACY = {"enabled": True, "bot": "reader-a", "house": "house-x", "skill_generation": "legacy"}


def test_setting_is_global_only_operator_write_only_and_off_by_default():
    assert is_global_only_runtime_setting("strict_reader")
    assert is_protected_setting_key("strict_reader")
    assert is_protected_setting_key(" Strict_Reader ")
    assert setting_default("strict_reader") == {"enabled": False}


def test_db_less_install_is_off_without_reading(monkeypatch):
    monkeypatch.setattr("llm_bawt.utils.config.has_database_credentials", lambda c=None: False)

    def boom():
        raise AssertionError("store must not be built")

    s = SimpleNamespace()
    cfg.load_snapshot(object(), boom).apply(s)
    assert s.strict_reader_bot is None and s.strict_reader_config_error is None


@pytest.mark.parametrize("store", [_Store(), _Store({"enabled": False}),
                                   _Store({"enabled": False, "bot": "reader-a"})])
def test_absent_or_explicit_false_is_off(with_db, store):
    s = _boot(store)
    assert s.strict_reader_bot is None and s.strict_reader_config_error is None
    assert not is_strict_request(s, "reader-a")


def test_complete_generation_admitted(with_db):
    s = _boot(_Store(GEN))
    assert (s.strict_reader_bot, s.strict_reader_house) == ("reader-a", "house-x")
    assert s.strict_reader_skill_generation == "gen-abc"
    assert s.strict_reader_skill_layers == {"core": "r1"}
    assert s.strict_reader_config_error is None


@pytest.mark.parametrize("patch", [
    {"house": None},
    {"house": "Bad House!"},
    {"skill_generation": None},
    {"skill_generation": "current"},
    {"skill_generation": "gen-../x"},
    {"skill_layers": None},
    {"skill_layers": {}},
    {"skill_layers": {"core": 3}},
    {"extra": 1},
    {"enabled": "yes"},
    {"enabled": None},
])
def test_trusted_slug_with_invalid_selection_is_fenced(with_db, patch):
    s = _boot(_Store({**GEN, **patch}))
    assert s.strict_reader_config_error
    assert s.strict_reader_bot == "reader-a"
    assert is_strict_request(s, "reader-a")  # every strict ingress keeps it off shared readers
    assert s.strict_reader_house is None and s.strict_reader_skill_generation is None
    with pytest.raises(StrictReaderRefused):
        strict_skill_source(s)
    with pytest.raises(StrictReaderRefused):
        strict_house(s, SimpleNamespace(house="house-x"))


def test_legacy_with_layers_is_fenced(with_db):
    s = _boot(_Store({**LEGACY, "skill_layers": {"core": "r1"}}))
    assert s.strict_reader_config_error and s.strict_reader_bot == "reader-a"
    assert s.strict_reader_skill_generation is None


@pytest.mark.parametrize("store", [
    _Store(fail=True),
    _Store(text="{not json"),
    _Store("on"),
    _Store([1]),
    _Store({"enabled": True}),
    _Store({"enabled": True, "bot": "Bad Bot"}),
    _Store({"bot": "reader-a"} | {"bot": None}),
])
def test_untrusted_config_fails_startup_with_redacted_error(with_db, store, caplog):
    with pytest.raises(cfg.StrictReaderConfigError) as exc:
        _boot(store)
    assert "secret-dsn" not in str(exc.value) and "secret-dsn" not in caplog.text
    assert "strict_reader" in caplog.text


def test_store_construction_failure_fails_startup(with_db):
    def broken():
        raise RuntimeError("engine down")

    with pytest.raises(cfg.StrictReaderConfigError):
        cfg.load_snapshot(object(), broken)


def test_restart_picks_up_change(with_db):
    store = _Store(GEN)
    first = _boot(store)
    store.text = json.dumps({**GEN, "skill_generation": "gen-def", "skill_layers": {"core": "r2"}})
    assert first.strict_reader_skill_generation == "gen-abc"  # snapshot is fixed per process
    second = _boot(store)
    assert second.strict_reader_skill_generation == "gen-def"
    assert second.strict_reader_skill_layers == {"core": "r2"}


def test_reader_inverse_is_legacy_on_same_bot_and_ignores_current(with_db, monkeypatch, tmp_path):
    current = tmp_path / "current-target"
    current.mkdir()
    monkeypatch.setenv("SKILL_SOURCES_CURRENT", str(current))
    store = _Store(GEN)
    assert _boot(store).strict_reader_skill_generation == "gen-abc"
    store.text = json.dumps(LEGACY)  # rollback: same bot/house, legacy, no layers, restart
    s = _boot(store)
    assert (s.strict_reader_bot, s.strict_reader_house) == ("reader-a", "house-x")
    assert s.strict_reader_skill_generation == "legacy" and s.strict_reader_skill_layers is None
    assert is_strict_request(s, "reader-a")
    roots = strict_skill_source(s).roots
    assert all(str(current) not in str(r) for r in roots)


def test_dormant_rollback_enabled_false_restores_ordinary(with_db):
    store = _Store(GEN)
    assert is_strict_request(_boot(store), "reader-a")
    store.text = json.dumps({"enabled": False})
    s = _boot(store)
    assert s.strict_reader_bot is None and not is_strict_request(s, "reader-a")


def test_raw_reader_distinguishes_absent_from_unavailable():
    from llm_bawt.runtime_settings import RuntimeSettingsStore

    store = RuntimeSettingsStore.__new__(RuntimeSettingsStore)
    store.engine = None
    with pytest.raises(RuntimeError):
        store.get_raw_value_json("global", "*", "strict_reader")
