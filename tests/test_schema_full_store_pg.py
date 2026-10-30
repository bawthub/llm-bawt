"""Full store-owned catalog proof on a disposable, independently named database.

The acceptance fixture supplies a networkless pgvector cluster and Unix socket;
this module NEVER connects to operational PostgreSQL or starts the application.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.full_store_fixture import full_store_db  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.integration


def test_pristine_full_store_catalog(full_store_db, tmp_path):
    from llm_bawt.schema_bootstrap import bootstrap_full_store

    config, engine = full_store_db
    manifest = bootstrap_full_store(config, engine)

    catalog_tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))
    lines, metadata = catalog_tool["capture_test"](config.POSTGRES_DATABASE)
    catalog = catalog_tool["parse_catalog"](lines)
    relations = {key for kind, key in catalog if kind == "relation" and catalog[(kind, key)]["kind"] in ("r", "p")}
    # 53 = 49 + TASK-1233's system_gates and gate_members (TurnLogStore bootstraps them)
    # + TASK-1329's bot_tag_catalog and bot_profile_tags (BotProfileStore bootstraps them)
    # + TASK-1461's api_balance_ledger and api_balances (TurnLogStore bootstraps them)
    # + TASK-1521's bot_profile_changes audit (BotProfileStore bootstraps it)
    # + TASK-1661's approval_approver_fanout (InterBotDeliveryStore bootstraps it)
    # + TASK-1749's gate_escalations audit (TurnLogStore bootstraps it)
    # + TASK-1745's system_events ledger (OpsStore bootstraps it).
    assert len(relations) == 60, sorted(relations)
    assert {"system_events", "system_gates", "gate_members", "gate_escalations", "bot_tag_catalog", "bot_profile_tags",
            "api_balance_ledger", "api_balances",
            "bot_profile_changes", "approval_approver_fanout"} <= {key.split(".")[-1] for key in relations}
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
