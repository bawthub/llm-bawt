"""TASK-2386: the master key fails closed instead of silently resetting."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from llm_bawt.service.api import _encrypted_records_exist
from llm_bawt.service.providers import crypto


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LLM_BAWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(crypto, "_fernet", None)
    monkeypatch.setattr(crypto, "_generation_guard", None)
    return tmp_path


def test_invalid_env_key_raises_and_never_falls_back(isolated, monkeypatch):
    path = isolated / ".config" / "llm-bawt" / "secret.key"
    path.parent.mkdir(parents=True)
    path.write_text(Fernet.generate_key().decode())
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", "x" * 7)  # synthetic, short
    with pytest.raises(ValueError, match="not a valid Fernet key") as caught:
        crypto._load_or_create_key()
    assert "x" * 7 not in str(caught.value)


def test_valid_env_key_wins_over_the_file(isolated, monkeypatch):
    env_key = Fernet.generate_key()
    path = isolated / ".config" / "llm-bawt" / "secret.key"
    path.parent.mkdir(parents=True)
    path.write_text(Fernet.generate_key().decode())
    monkeypatch.setenv("LLM_BAWT_SECRET_KEY", env_key.decode())
    assert crypto._load_or_create_key() == env_key


def test_missing_key_with_records_refuses_to_generate(isolated):
    crypto.refuse_generation_when(lambda: True)
    with pytest.raises(crypto.MissingRootKeyError):
        crypto._load_or_create_key()
    assert not (isolated / ".config" / "llm-bawt" / "secret.key").exists()


def test_guard_that_cannot_tell_fails_closed(isolated):
    def broken():
        raise RuntimeError("db down")

    crypto.refuse_generation_when(broken)
    with pytest.raises(crypto.MissingRootKeyError):
        crypto._load_or_create_key()


def test_fresh_install_still_generates(isolated):
    crypto.refuse_generation_when(lambda: False)
    key = crypto._load_or_create_key()
    Fernet(key)
    assert (isolated / ".config" / "llm-bawt" / "secret.key").read_bytes().strip() == key


def test_existing_file_is_used_even_when_generation_is_refused(isolated):
    key = Fernet.generate_key()
    path = isolated / ".config" / "llm-bawt" / "secret.key"
    path.parent.mkdir(parents=True)
    path.write_text(key.decode())
    crypto.refuse_generation_when(lambda: True)
    assert crypto._load_or_create_key() == key


class _Store:
    def __init__(self, engine):
        self.engine = engine


def test_records_check_finds_an_encrypted_value(tmp_path):
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE runtime_settings (value_json TEXT)"))
        conn.execute(text("INSERT INTO runtime_settings VALUES ('true')"))
    check = _encrypted_records_exist(_Store(engine))
    assert check() is False
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO runtime_settings VALUES ('"enc:v1:abc"')"""))
    assert check() is True
    assert _encrypted_records_exist(_Store(None))() is False
