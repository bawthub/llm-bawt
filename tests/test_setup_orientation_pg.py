"""First-bot orientation admission against the production create transaction."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from llm_bawt.bot_create_transaction import BotCreateTransaction
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_ledger import delivery_audience
from llm_bawt.memory.model_catalog_migration import migrate_model_catalog
from llm_bawt.message_authorship import SETUP_ORIENTATION_AUTHOR_ID
from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore
from llm_bawt.service.inter_bot_claims import validate_inter_bot_claim
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.service.setup_orientation import SETTING_KEY, record_choice
from llm_bawt.service.turn_admission import lock_target

pytestmark = pytest.mark.integration


def test_two_setup_submissions_commit_only_one_first_chat(private_app_pg):
    profiles = BotProfileStore(private_app_pg)
    settings = RuntimeSettingsStore(private_app_pg)
    deliveries = InterBotDeliveryStore(private_app_pg)
    assert profiles.engine is settings.engine is deliveries.engine
    migrate_model_catalog(profiles.engine)
    with profiles.engine.begin() as conn:
        model = conn.execute(text("""
            INSERT INTO models (key, vendor, display_name, default_context_window)
            VALUES ('orientation-model', 'anthropic', 'Orientation model', 10000) RETURNING id
        """)).scalar_one()
        endpoint = conn.execute(text("""
            INSERT INTO model_endpoints (model_id, access_path_id)
            SELECT :m, id FROM access_paths WHERE key='anthropic-api' RETURNING id
        """), {"m": model}).scalar_one()

    guard = SimpleNamespace(lock=lock_target, check=lambda *_: None)
    start = Barrier(2, timeout=10)

    def create(slug, *, opted_in=True, fail=False, race=False):
        def on_create(conn, profile):
            record_choice(conn, profile, opted_in=opted_in, user_id=f"user-{slug}",
                          prompt="Welcome", store=deliveries)
            if fail:
                raise RuntimeError("rollback after enqueue")

        payload = {"slug": slug, "name": slug, "system_prompt": "Be helpful",
                   "bot_type": "agent", "agent_backend": "claude-code", "harness": "claude-code",
                   "default_model": "orientation-model", "endpoint_id": endpoint}
        if race:
            start.wait()
        return BotCreateTransaction(profiles).create(payload, {}, guard, lambda *_: None, on_create=on_create)

    with pytest.raises(RuntimeError, match="rollback after enqueue"):
        create("rollback", fail=True)
    assert profiles.get("rollback") is None
    assert not deliveries.list()
    assert settings.get_scope_settings("global", "*").get(SETTING_KEY) is None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(create, "introduction-a", race=True)
        second = pool.submit(create, "introduction-b", race=True)
        assert {first.result(timeout=20).slug, second.result(timeout=20).slug} == {"introduction-a", "introduction-b"}
    rows = deliveries.list()
    assert len(rows) == 1
    row = rows[0]
    target = row.target_bot_id
    with pytest.raises(HTTPException, match="409"):
        create(target)
    assert [d.id for d in deliveries.list()] == [row.id]
    state = settings.get_scope_settings("global", "*")[SETTING_KEY]
    assert state == {"bot_id": target, "opted_in": True, "milestone": "first_chat", "delivery_id": row.id}
    assert delivery_audience(row, default_user="unrelated").user_id == f"user-{target}"
    assert "orientation-model, connection anthropic-api" in row.message
    assert deliveries.payload(row.id)["prefer_steer"] is False

    claimed = deliveries.claim_next(target, claim_owner="fixture", steer_capable=False)
    assert claimed is not None and claimed.id == row.id
    request = ChatCompletionRequest.model_validate({
        **deliveries.payload(row.id), "inter_bot_claim_token": claimed.claim_token,
    })
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=deliveries))
    author = validate_inter_bot_claim(service, request)
    assert author.entity_type == "system" and author.entity_id == SETUP_ORIENTATION_AUTHOR_ID

    # A second choice cannot replace the committed milestone, even opt-out.
    create("later-bot", opted_in=False)
    assert settings.get_scope_settings("global", "*")[SETTING_KEY] == state
    assert len(deliveries.list()) == 1
