"""The one ordered list of store-owned schema bootstraps (TASK-1802).

Every store creates its own tables on construction. This module is the single
place that names all of them, in dependency order, so a disposable database
(the schema-catalog acceptance test, the isolated preview stack) gets exactly
the schema the running service owns. It creates schema only: no rows beyond
what each store's bootstrap itself writes.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import text
from sqlalchemy.engine import Engine

# The memory backend provisions per-bot partitions; bootstrapping needs a bot
# id, so a synthetic one is used and its (empty) partitions removed afterwards.
SCRATCH_BOT_ID = "schema_full_fixture"
PARTITIONED_PARENTS = ("messages", "memories", "forgotten_messages")


def _steps(config, engine: Engine) -> list[tuple[str, Callable[[], object]]]:
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

    # Dependencies matter: memory installs sessions and the two fresh FKs;
    # model migration needs bot_profiles and creates the referenced catalog.
    return [
        ("memory.postgresql:PostgreSQLMemoryBackend", lambda: PostgreSQLMemoryBackend(config, bot_id=SCRATCH_BOT_ID)),
        ("runtime_settings:BotProfileStore", lambda: BotProfileStore(config)),
        ("memory.model_catalog_migration:migrate_model_catalog", lambda: migrate_model_catalog(engine)),
        ("runtime_settings:RuntimeSettingsStore", lambda: RuntimeSettingsStore(config)),
        ("profiles:ProfileManager", lambda: ProfileManager(config)),
        ("prompt_registry:PromptTemplateStore", lambda: PromptTemplateStore(config)),
        ("approval_policy_store:ToolApprovalPolicyStore", lambda: ToolApprovalPolicyStore(config, engine=engine)),
        ("service.turn_logs:TurnLogStore", lambda: TurnLogStore(config)),
        ("service.chat_pending_questions:PendingQuestionStore", lambda: PendingQuestionStore(config)),
        ("service.scheduler:create_scheduler_tables", lambda: create_scheduler_tables(engine)),
        ("inter_bot_delivery:InterBotDeliveryStore", lambda: InterBotDeliveryStore(config)),
        ("service.talk_reply_outbox:TalkReplyOutbox", lambda: TalkReplyOutbox(engine)),
        ("media.db:MediaGenerationStore", lambda: MediaGenerationStore(config)),
        ("media.assets:MediaAssetStore", lambda: MediaAssetStore(config)),
        ("media.gpu_handoff_store:GpuHandoffStore", lambda: GpuHandoffStore(engine)),
        ("integrations.home_audio_store:HomeAudioStore", lambda: HomeAudioStore(engine)),
        ("integrations.google_home_probe_store:ProbeCaptureStore", lambda: ProbeCaptureStore(engine)),
        ("integrations.google_home_oauth:ProbeOAuthStore", lambda: ProbeOAuthStore(engine, ProbeSettings())),
        ("ops.store:OpsStore", lambda: OpsStore(config, engine=engine)),
        ("ops.release_store:ReleaseStore", lambda: ReleaseStore(config, engine=engine)),
        ("agent_bridge.store:create_agent_event_tables", lambda: create_agent_event_tables(engine)),
    ]


def bootstrap_full_store(config, engine: Engine) -> list[str]:
    """Create every store-owned table on ``engine``; return the ordered manifest.

    The caller owns target safety: ``config`` and ``engine`` must point at the
    same disposable database. The scratch bot's partitions are dropped with
    RESTRICT and must be empty, so the result is shared schema only.
    """
    manifest = []
    for label, action in _steps(config, engine):
        action()
        manifest.append(label)
    with engine.begin() as conn:
        for parent in PARTITIONED_PARENTS:
            child = f"{parent}_p_{SCRATCH_BOT_ID}"
            rows = conn.execute(text(f"SELECT count(*) FROM ONLY public.{child}")).scalar_one()
            if rows:
                raise RuntimeError(f"{child} holds {rows} rows; refusing to drop it")
            conn.execute(text(f"DROP TABLE public.{child} RESTRICT"))
    return manifest
