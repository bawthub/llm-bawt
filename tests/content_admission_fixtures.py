"""Explicit SQLite admission stores for tests of adjacent tool behavior.

Keep the real provenance boundary active without consulting a live
configuration database. Not autouse: outage and held-write tests keep their
own deliberately failing/tainted stores.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.content_provenance import ContentProvenanceStore


@pytest.fixture(name="content_store")
def _content_store():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False},
    )
    yield ContentProvenanceStore(engine)
    engine.dispose()


@pytest.fixture(name="native_admission")
def _native_admission(content_store):
    return {
        "content_store": content_store,
        "bot_id": "synthetic-bot",
        "turn_id": "synthetic-turn",
        "trigger_message_id": "synthetic-trigger",
    }
