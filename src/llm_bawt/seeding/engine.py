"""Idempotent tenant seed engine (TASK-350 / TASK-355).

What ``apply()`` provisions on a virgin database — the *infrastructure* a
tenant needs before any bot can exist:

- the normalized model-catalog schema + standard ``access_paths`` rows + the
  ``bot_profiles`` endpoint trigger (:func:`migrate_model_catalog`);
- the ``bot_profiles`` table itself (via :class:`BotProfileStore`);
- built-in prompt-registry defaults.
- disabled ops starter specifications and starter approval policies.

It deliberately seeds **no bots and no model rows**: a bot without a provider
credential is broken on arrival. The setup wizard connects a provider first,
then calls :meth:`TenantSeeder.register_model` with a model discovered live
from that provider, and creates the first bot against the returned endpoint.

Guarantees:

- Catalog/policy insert-if-missing on natural keys; re-running preserves
  operator-edited rows. Prompt defaults additionally advance proven
  seed/sync-managed rows, while explicit and ambiguous overrides stay intact.
- Nothing here reads or writes API keys — those live in the encrypted
  CredentialStore behind the provider adapters (TASK-825).

Both the installer (``python -m llm_bawt.seeding``) and the wizard route
(``/v1/setup/*``) call this engine; seed policy lives here, not in HTTP.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from sqlalchemy import Engine, text

from .data import EndpointSeed, ModelSeed

logger = logging.getLogger(__name__)


class TenantSeeder:
    """Apply the Phase-1 seed set to a tenant database, idempotently."""

    def __init__(self, config: Any = None, engine: Engine | None = None):
        if config is None:
            from ..utils.config import Config

            config = Config()
        self.config = config
        self._engine = engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            from ..utils.db import get_shared_engine

            self._engine = get_shared_engine(self.config)
        if self._engine is None:
            raise RuntimeError(
                "Tenant seed requires database credentials (LLM_BAWT_POSTGRES_*)"
            )
        return self._engine

    # ── apply ──────────────────────────────────────────────────────────

    def apply(self) -> dict[str, Any]:
        """Provision tenant infrastructure and return a created/existing report.

        Order matters on a truly virgin database: the ``bot_profiles`` table
        must exist BEFORE :func:`migrate_model_catalog` runs, because the
        migration only installs the endpoint-check trigger (which derives the
        ``default_model`` / ``agent_backend`` mirror columns) when the table is
        present. Instantiating :class:`BotProfileStore` first guarantees that.
        """
        report: dict[str, Any] = {}
        self._bot_profile_store()
        report["catalog_migration"] = self._ensure_catalog_schema()
        report["prompts"] = self._seed_prompt_defaults()
        report["ops_operations"] = self._seed_ops_catalog()
        report["approval_policies"] = self._seed_approval_policies()
        logger.info("Tenant seed complete: %s", report)
        return report

    def register_model(
        self,
        *,
        vendor: str,
        model_id: str,
        access_path_key: str,
        display_name: str | None = None,
        context_window: int,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Insert-if-missing one model + endpoint and resolve its endpoint id.

        ``model_id`` is the provider's upstream id and doubles as the catalog
        key. Existing rows are left untouched (the operator may have edited
        them); the endpoint id is resolved either way.
        """
        key = (model_id or "").strip()
        if not key:
            raise ValueError("model_id is required")
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        model = ModelSeed(
            key=key,
            vendor=vendor,
            display_name=(display_name or key).strip() or key,
            default_context_window=context_window,
            description=description,
        )
        endpoint = EndpointSeed(
            model_key=key,
            access_path_key=access_path_key,
            upstream_model_id=key,
        )
        with self.engine.begin() as conn:
            models = self._seed_models(conn, (model,))
            endpoints = self._seed_endpoints(conn, (endpoint,))
            endpoint_id = self._resolve_endpoint_id(conn, key, access_path_key)
        return {
            "models": models,
            "endpoints": endpoints,
            "endpoint_id": endpoint_id,
        }

    # ── sections ───────────────────────────────────────────────────────

    def _ensure_catalog_schema(self) -> dict[str, Any]:
        """Normalized catalog schema, standard access paths, bot trigger."""
        from ..memory.model_catalog_migration import migrate_model_catalog

        return migrate_model_catalog(self.engine)

    def _seed_models(self, conn, seeds: tuple[ModelSeed, ...]) -> dict[str, list[str]]:
        sql = text("""
            INSERT INTO models
                (key, vendor, display_name, description, default_context_window,
                 default_tool_support, created_at, updated_at)
            VALUES
                (:key, :vendor, :display_name, :description,
                 :default_context_window, :default_tool_support, NOW(), NOW())
            ON CONFLICT (key) DO NOTHING
            RETURNING id
        """)
        created: list[str] = []
        existing: list[str] = []
        for seed in seeds:
            row = conn.execute(
                sql,
                {
                    "key": seed.key,
                    "vendor": seed.vendor,
                    "display_name": seed.display_name,
                    "description": seed.description,
                    "default_context_window": seed.default_context_window,
                    "default_tool_support": seed.default_tool_support,
                },
            ).first()
            (created if row is not None else existing).append(seed.key)
        return {"created": created, "existing": existing}

    def _seed_endpoints(self, conn, seeds: tuple[EndpointSeed, ...]) -> dict[str, list[str]]:
        sql = text("""
            INSERT INTO model_endpoints
                (model_id, access_path_id, upstream_model_id, serving_config,
                 context_window_override, tool_support_override, pricing,
                 created_at, updated_at)
            SELECT m.id, a.id, :upstream_model_id,
                   CAST(:serving_config AS jsonb), :context_window_override,
                   :tool_support_override, CAST(:pricing AS jsonb), NOW(), NOW()
              FROM models m, access_paths a
             WHERE m.key = :model_key AND a.key = :access_path_key
            ON CONFLICT (model_id, access_path_id) DO NOTHING
            RETURNING id
        """)
        created: list[str] = []
        existing: list[str] = []
        for seed in seeds:
            ref = f"{seed.model_key}@{seed.access_path_key}"
            row = conn.execute(
                sql,
                {
                    "model_key": seed.model_key,
                    "access_path_key": seed.access_path_key,
                    "upstream_model_id": seed.upstream_model_id,
                    "serving_config": json.dumps(seed.serving_config),
                    "context_window_override": seed.context_window_override,
                    "tool_support_override": seed.tool_support_override,
                    "pricing": json.dumps(seed.pricing) if seed.pricing is not None else None,
                },
            ).first()
            (created if row is not None else existing).append(ref)
        return {"created": created, "existing": existing}

    def _seed_prompt_defaults(self) -> dict[str, Any]:
        """Seed missing prompts and advance managed defaults, preserving overrides."""
        from ..prompt_registry import PromptTemplateStore

        store = PromptTemplateStore(self.config)
        if store.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        result = store.seed_defaults()
        return {"created": result["created"], "existing": result["skipped"]}

    def _seed_ops_catalog(self) -> dict[str, Any]:
        """Canonical ops catalog rows (TASK-639), insert-if-missing per slug.

        Seeds ship ``enabled=False``: an operator has to turn each privileged
        operation on deliberately in the UI. Existing rows are never overwritten.
        """
        from ..ops.seeds import seed_all
        from ..ops.store import OpsStore

        store = OpsStore(self.config, engine=self.engine)
        inserted, skipped = seed_all(store)
        return {"created": inserted, "existing": skipped}

    def _seed_approval_policies(self) -> dict[str, Any]:
        """Starter approval rules, including the TASK-639 ``ops_run`` gates.

        Without these an operator sees no ops rules to edit in the admin UI.
        The gate itself does not depend on them — ``ops_run`` fails closed in
        :func:`agent_bridge.approval.evaluate` when nothing matches — but the
        seeded rows are what make the gating visible and carve-out-able.
        """
        from ..approval_policies import ToolApprovalPolicyStore

        store = ToolApprovalPolicyStore(self.config, engine=self.engine)
        if store.engine is None:
            raise RuntimeError("Tool approval policy DB unavailable")
        seeded = store.seed_defaults()
        return {"created": seeded, "total": len(store.list_all())}

    def _resolve_endpoint_id(self, conn, model_key: str, access_path_key: str) -> int:
        endpoint_id = conn.execute(
            text("""
                SELECT e.id
                  FROM model_endpoints e
                  JOIN models m ON m.id = e.model_id
                  JOIN access_paths a ON a.id = e.access_path_id
                 WHERE m.key = :model_key AND a.key = :access_path_key
            """),
            {"model_key": model_key, "access_path_key": access_path_key},
        ).scalar()
        if endpoint_id is None:
            raise RuntimeError(
                f"Seed endpoint {model_key}@{access_path_key} not found after catalog seed"
            )
        return int(endpoint_id)

    def _bot_profile_store(self):
        """BotProfileStore, which also bootstraps the ``bot_profiles`` table."""
        from ..runtime_settings import BotProfileStore

        store = BotProfileStore(self.config)
        if store.engine is None:
            raise RuntimeError("Bot profiles DB unavailable")
        return store


def seed_tenant(config: Any = None) -> dict[str, Any]:
    """Provision tenant infrastructure (idempotent).

    The one entrypoint for the installer and the setup wizard (TASK-355).
    Seeds schema, standard access paths, and prompt defaults — never bots or
    model rows (see the module docstring).
    """
    return TenantSeeder(config).apply()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a virgin llm-bawt tenant's infrastructure "
        "(catalog schema, standard access paths, prompt-registry defaults). "
        "Idempotent: existing rows are never modified. Bots are created by "
        "the setup wizard after a provider is connected."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    print(json.dumps(seed_tenant(), indent=2))


if __name__ == "__main__":
    main()
