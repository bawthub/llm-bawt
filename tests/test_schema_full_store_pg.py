"""Full store-owned catalog proof on a disposable, independently named database.

The acceptance fixture supplies a networkless pgvector cluster and Unix socket;
this module NEVER connects to operational PostgreSQL or starts the application.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from tests.pg_isolation import require_test_db_config
from tests.private_postgres import PORT, private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def full_store_db(private_pg, monkeypatch):
    from llm_bawt.utils import db

    name = f"schema_full_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one() == "postgres"
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = private_url(private_pg, name)
    monkeypatch.setattr(db, "build_postgres_url", lambda config: url)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", str(private_pg))
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    engine = create_engine(url)
    try:
        config = require_test_db_config()
        assert config.POSTGRES_DATABASE == engine.url.database == name
        with engine.connect() as conn:
            assert conn.execute(text("SELECT current_database()")).scalar_one() == name
        yield config, engine
    finally:
        engine.dispose()
        db.reset_shared_engines()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_pristine_full_store_catalog(full_store_db, tmp_path):
    from agent_bridge.store import create_agent_event_tables
    from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
    from llm_bawt.integrations.google_home_oauth import ProbeOAuthStore, ProbeSettings
    from llm_bawt.integrations.google_home_probe_store import ProbeCaptureStore
    from llm_bawt.integrations.home_audio_store import HomeAudioStore
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.media.assets import MediaAssetStore
    from llm_bawt.media.db import MediaGenerationStore
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore
    from llm_bawt.memory.model_catalog_migration import migrate_model_catalog
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.ops.release_store import ReleaseStore
    from llm_bawt.ops.store import OpsStore
    from llm_bawt.profiles import ProfileManager
    from llm_bawt.prompt_registry import PromptTemplateStore
    from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore
    from llm_bawt.service.chat_pending_questions import PendingQuestionStore
    from llm_bawt.service.scheduler import create_scheduler_tables
    from llm_bawt.service.talk_reply_outbox import TalkReplyOutbox
    from llm_bawt.service.turn_logs import TurnLogStore

    config, engine = full_store_db
    manifest = []

    def bootstrap(label, action):
        action()
        manifest.append(label)

    # Dependencies matter: memory installs sessions and the two fresh FKs;
    # model migration needs bot_profiles and creates the referenced catalog.
    bootstrap("memory.postgresql:PostgreSQLMemoryBackend", lambda: PostgreSQLMemoryBackend(config, bot_id="schema_full_fixture"))
    bootstrap("runtime_settings:BotProfileStore", lambda: BotProfileStore(config))
    bootstrap("memory.model_catalog_migration:migrate_model_catalog", lambda: migrate_model_catalog(engine))
    bootstrap("runtime_settings:RuntimeSettingsStore", lambda: RuntimeSettingsStore(config))
    bootstrap("profiles:ProfileManager", lambda: ProfileManager(config))
    bootstrap("prompt_registry:PromptTemplateStore", lambda: PromptTemplateStore(config))
    bootstrap("approval_policy_store:ToolApprovalPolicyStore", lambda: ToolApprovalPolicyStore(config, engine=engine))
    bootstrap("service.turn_logs:TurnLogStore", lambda: TurnLogStore(config))
    bootstrap("service.chat_pending_questions:PendingQuestionStore", lambda: PendingQuestionStore(config))
    bootstrap("service.scheduler:create_scheduler_tables", lambda: create_scheduler_tables(engine))
    bootstrap("inter_bot_delivery:InterBotDeliveryStore", lambda: InterBotDeliveryStore(config))
    bootstrap("service.talk_reply_outbox:TalkReplyOutbox", lambda: TalkReplyOutbox(engine))
    bootstrap("media.db:MediaGenerationStore", lambda: MediaGenerationStore(config))
    bootstrap("media.assets:MediaAssetStore", lambda: MediaAssetStore(config))
    bootstrap("media.gpu_handoff_store:GpuHandoffStore", lambda: GpuHandoffStore(engine))
    bootstrap("integrations.home_audio_store:HomeAudioStore", lambda: HomeAudioStore(engine))
    bootstrap("integrations.google_home_probe_store:ProbeCaptureStore", lambda: ProbeCaptureStore(engine))
    bootstrap("integrations.google_home_oauth:ProbeOAuthStore", lambda: ProbeOAuthStore(engine, ProbeSettings()))
    bootstrap("ops.store:OpsStore", lambda: OpsStore(config, engine=engine))
    bootstrap("ops.release_store:ReleaseStore", lambda: ReleaseStore(config, engine=engine))
    bootstrap("agent_bridge.store:create_agent_event_tables", lambda: create_agent_event_tables(engine))

    # Memory constructor provisions exactly one three-partition synthetic bot.
    # Remove only those test-owned empty children, with RESTRICT; the reference
    # consists of shared source schema, not fixture-created runtime partitions.
    with engine.begin() as conn:
        for parent in ("messages", "memories", "forgotten_messages"):
            child = f"{parent}_p_schema_full_fixture"
            assert conn.execute(text(f"SELECT count(*) FROM ONLY public.{child}")).scalar_one() == 0
            conn.execute(text(f"DROP TABLE public.{child} RESTRICT"))

    catalog_tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))
    lines, metadata = catalog_tool["capture_test"](config.POSTGRES_DATABASE)
    catalog = catalog_tool["parse_catalog"](lines)
    relations = {key for kind, key in catalog if kind == "relation" and catalog[(kind, key)]["kind"] in ("r", "p")}
    # 53 = 49 + TASK-1233's system_gates and gate_members (TurnLogStore bootstraps them)
    # + TASK-1329's bot_tag_catalog and bot_profile_tags (BotProfileStore bootstraps them)
    # + TASK-1461's api_balance_ledger and api_balances (TurnLogStore bootstraps them)
    # + TASK-1521's bot_profile_changes audit (BotProfileStore bootstraps it).
    assert len(relations) == 56, sorted(relations)
    assert {"system_gates", "gate_members", "bot_tag_catalog", "bot_profile_tags",
            "api_balance_ledger", "api_balances",
            "bot_profile_changes"} <= {key.split(".")[-1] for key in relations}
    assert not any("schema_full_fixture" in key for _, key in catalog)
    assert catalog_tool["compare"](catalog, catalog) == []
    metadata["bootstrap_manifest"] = manifest
    output_dir = Path(os.environ.get("TASK1149_EVIDENCE_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_tool["write_snapshot"](output_dir / "pristine-full-store.jsonl", lines, metadata)
    assert len(json.loads((output_dir / "pristine-full-store.jsonl.meta.json").read_text())["bootstrap_manifest"]) == 21


def test_upgrade_synthetic_legacy_converges(full_store_db, tmp_path, monkeypatch):
    """Rehearse physical DDL on this test's private DB, never on the saved live DB."""
    config, engine = full_store_db
    external_dir = os.environ.get("TASK1149_EVIDENCE_DIR")
    monkeypatch.setenv("TASK1149_EVIDENCE_DIR", str(tmp_path))
    test_pristine_full_store_catalog(full_store_db, tmp_path)
    evidence_dir = Path(external_dir) if external_dir else tmp_path
    evidence_dir.mkdir(parents=True, exist_ok=True)
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))
    reference_lines = (tmp_path / "pristine-full-store.jsonl").read_text().splitlines()
    reference = tool["parse_catalog"](reference_lines)
    if external_dir:
        tool["write_snapshot"](evidence_dir / "synthetic-reference.jsonl", reference_lines,
                               {"identity": config.POSTGRES_DATABASE, "mode": "synthetic-reference"})
    assert engine.url.database == config.POSTGRES_DATABASE
    assert config.POSTGRES_DATABASE.endswith("_test")

    # Recreate ONLY the observed shared-schema legacy signatures (not live data,
    # runtime partitions, or a divergent extension version).
    with engine.begin() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one() == config.POSTGRES_DATABASE
        for flag in ("tts_mode", "include_summaries", "include_in_global_search"):
            conn.exec_driver_sql(f"ALTER TABLE public.bot_profiles ALTER COLUMN {flag} DROP NOT NULL")
        conn.exec_driver_sql("ALTER TABLE public.chat_pending_questions ALTER COLUMN origin_harness DROP NOT NULL")
        for column, width in (("tool_use_id", 128), ("parent_tool_use_id", 128),
                              ("approval_request_id", 128), ("approval_status", 24)):
            conn.exec_driver_sql(f"ALTER TABLE public.tool_call_records ALTER COLUMN {column} TYPE varchar({width})")
        for table, old, new in (
            ("agent_runs", "agent_runs_pkey", "openclaw_runs_pkey"),
            ("agent_events", "agent_events_pkey", "openclaw_events_pkey"),
            ("agent_events", "agent_events_event_dedupe_key_key", "openclaw_events_event_dedupe_key_key"),
            ("agent_session_state", "agent_session_state_pkey", "openclaw_session_state_pkey"),
        ):
            conn.exec_driver_sql(f"ALTER TABLE public.{table} RENAME CONSTRAINT {old} TO {new}")
        conn.exec_driver_sql("ALTER SEQUENCE public.agent_events_id_seq RENAME TO openclaw_events_id_seq")
        for index in ("ix_bot_profiles_endpoint_id", "ix_bot_profiles_prompt_override_id",
                      "ix_entity_profiles_email", "ix_tool_call_records_tool_use_id", "ix_turn_logs_ended_at"):
            conn.exec_driver_sql(f"DROP INDEX public.{index}")
        conn.exec_driver_sql("DROP INDEX public.ix_inter_bot_delivery_dispatch")
        conn.exec_driver_sql("""CREATE INDEX ix_inter_bot_delivery_dispatch
            ON public.inter_bot_deliveries (target_bot_id, ordinal)
            WHERE status IN ('QUEUED', 'DISPATCHING')""")

    def capture(name):
        lines, metadata = tool["capture_test"](config.POSTGRES_DATABASE)
        tool["write_snapshot"](evidence_dir / f"{name}.jsonl", lines, metadata)
        return tool["parse_catalog"](lines)

    before = capture("synthetic-legacy-before")
    drift = tool["compare"](reference, before)
    assert len(drift) > 0
    assert {row["kind"] for row in drift} <= {"column", "constraint", "index", "sequence", "owner_acl", "dependency"}

    from scripts.schema_shared_upgrade import upgrade_shared_schema

    from llm_bawt.runtime_settings import BotProfile

    # The pristine store has no catalog endpoint to bind, and every harness
    # requires one, so the legacy NULL row is planted with the endpoint
    # trigger paused for this one insert. Re-enabling needs its own transaction:
    # the insert leaves deferred trigger events that block ALTER TABLE.
    with engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE public.bot_profiles DISABLE TRIGGER bot_profiles_model_endpoint_check")
    try:
        with engine.begin() as conn:
            conn.execute(BotProfile.__table__.insert().values(
                slug="synthetic-null-preflight", name="Synthetic", description="",
                system_prompt="Synthetic", harness="chat", tts_mode=None,
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
            ))
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE public.bot_profiles ENABLE TRIGGER bot_profiles_model_endpoint_check")
    with pytest.raises(RuntimeError, match="tts_mode contains NULL"):
        upgrade_shared_schema(engine, expected_db=config.POSTGRES_DATABASE)
    assert tool["compare"](before, capture("rejected-null-before")) == []
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM public.bot_profiles WHERE slug='synthetic-null-preflight'")

    upgrade_shared_schema(engine, expected_db=config.POSTGRES_DATABASE)
    after = capture("synthetic-legacy-after")
    remaining = tool["compare"](reference, after)
    assert remaining == [], remaining[:20]
    upgrade_source = Path(__file__).resolve().parents[1] / "scripts/schema_shared_upgrade.py"
    (evidence_dir / "upgrade-diff.json").write_text(json.dumps({
        "before": len(drift), "after": len(remaining), "database": config.POSTGRES_DATABASE,
        "source": str(upgrade_source.relative_to(Path(__file__).resolve().parents[1])),
        "upgrade_sha256": hashlib.sha256(upgrade_source.read_bytes()).hexdigest(),
        "before_sha256": hashlib.sha256((evidence_dir / "synthetic-legacy-before.jsonl").read_bytes()).hexdigest(),
        "after_sha256": hashlib.sha256((evidence_dir / "synthetic-legacy-after.jsonl").read_bytes()).hexdigest(),
    }, sort_keys=True) + "\n")
