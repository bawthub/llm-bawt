"""Normalized model catalog bootstrap, repair, and legacy-table removal.

The normalized catalog (``models`` -> ``model_endpoints`` -> ``access_paths``)
is the only model authority.  This migration is intentionally idempotent:

* create the normalized schema and standard access paths on a fresh install;
* preserve every valid bot endpoint assignment exactly as stored;
* deterministically repair only invalid/unresolved bot assignments from the
  normalized tables;
* install the canonical bot endpoint trigger; and
* remove the obsolete flat model table/view and endpoint ``legacy_type`` shim.

No normalized row is ever reconstructed from the removed flat catalog.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessPathSeed:
    key: str
    vendor: str
    protocol: str
    base_url: str | None
    auth_mechanism: str
    engine_kind: str | None = None


STANDARD_ACCESS_PATHS = (
    AccessPathSeed("openai-api", "openai", "chat-completions", "https://api.openai.com/v1", "api-key"),
    AccessPathSeed("openai-responses", "openai", "responses", "https://api.openai.com/v1", "api-key"),
    AccessPathSeed("openai-oauth", "openai", "responses", "https://chatgpt.com/backend-api/codex", "oauth"),
    AccessPathSeed("anthropic-api", "anthropic", "anthropic-messages", "https://api.anthropic.com", "api-key"),
    AccessPathSeed("anthropic-oauth", "anthropic", "anthropic-messages", "https://api.anthropic.com", "oauth"),
    AccessPathSeed("xai-chat", "xai", "chat-completions", "https://api.x.ai/v1", "api-key"),
    AccessPathSeed("xai-responses", "xai", "responses", "https://api.x.ai/v1", "api-key"),
    AccessPathSeed("zai-anthropic", "zai", "anthropic-messages", "https://api.z.ai/api/anthropic", "api-key"),
    AccessPathSeed("moonshot-anthropic", "moonshot", "anthropic-messages", "https://api.moonshot.ai/anthropic", "api-key"),
    AccessPathSeed("kimi-coding-chat", "kimi", "chat-completions", "https://api.kimi.com/coding/v1", "api-key"),
    AccessPathSeed("local-llamacpp", "local", "chat-completions", None, "none", "llama-cpp"),
    AccessPathSeed("local-vllm", "local", "chat-completions", None, "none", "vllm"),
    AccessPathSeed("ollama", "local", "chat-completions", None, "none", "ollama"),
    AccessPathSeed("openai-compatible", "custom", "chat-completions", None, "configured"),
)


_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(128) NOT NULL UNIQUE,
    vendor VARCHAR(64) NOT NULL,
    display_name VARCHAR(256) NOT NULL,
    description TEXT,
    default_context_window INTEGER NOT NULL,
    default_tool_support VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT models_context_window_positive CHECK (default_context_window > 0)
);

CREATE TABLE IF NOT EXISTS access_paths (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(128) NOT NULL UNIQUE,
    vendor VARCHAR(64) NOT NULL,
    protocol VARCHAR(32) NOT NULL,
    base_url TEXT,
    auth_mechanism VARCHAR(64) NOT NULL,
    engine_kind VARCHAR(64),
    system_prompt_overrides JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT access_paths_protocol_check
        CHECK (protocol IN ('chat-completions', 'responses', 'anthropic-messages'))
);

CREATE TABLE IF NOT EXISTS model_endpoints (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    access_path_id BIGINT NOT NULL REFERENCES access_paths(id) ON DELETE RESTRICT,
    upstream_model_id VARCHAR(512),
    serving_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_window_override INTEGER,
    tool_support_override VARCHAR(64),
    pricing JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT model_endpoints_model_access_key UNIQUE (model_id, access_path_id)
);

CREATE INDEX IF NOT EXISTS idx_model_endpoints_model_id ON model_endpoints(model_id);
CREATE INDEX IF NOT EXISTS idx_model_endpoints_access_path_id ON model_endpoints(access_path_id);

ALTER TABLE access_paths
    ADD COLUMN IF NOT EXISTS system_prompt_overrides JSONB NOT NULL DEFAULT '{}'::jsonb;
UPDATE access_paths SET system_prompt_overrides = '{}'::jsonb
 WHERE system_prompt_overrides IS NULL;
ALTER TABLE access_paths
    ALTER COLUMN system_prompt_overrides SET DEFAULT '{}'::jsonb,
    ALTER COLUMN system_prompt_overrides SET NOT NULL;

-- Per-model Claude Code auto-compact window; NULL = global agent_compact_threshold.
ALTER TABLE models ADD COLUMN IF NOT EXISTS compact_threshold INTEGER;
ALTER TABLE models DROP CONSTRAINT IF EXISTS models_compact_threshold_positive;
ALTER TABLE models ADD CONSTRAINT models_compact_threshold_positive
    CHECK (compact_threshold IS NULL OR compact_threshold > 0);
"""


_BOT_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION bot_profiles_check_model_endpoint()
RETURNS TRIGGER AS $$
DECLARE
    endpoint_protocol TEXT;
    endpoint_vendor TEXT;
    endpoint_model_key TEXT;
BEGIN
    IF NEW.harness = 'openclaw' THEN
        IF NEW.endpoint_id IS NOT NULL THEN
            RAISE EXCEPTION 'bot %: openclaw owns its model and cannot pin endpoint_id', NEW.slug
            USING ERRCODE = 'check_violation';
        END IF;
        NEW.default_model := NULL;
        NEW.agent_backend := 'openclaw';
        RETURN NEW;
    END IF;

    IF NEW.endpoint_id IS NULL THEN
        RAISE EXCEPTION 'bot %: harness % requires endpoint_id', NEW.slug, NEW.harness
        USING ERRCODE = 'not_null_violation';
    END IF;

    SELECT a.protocol, a.vendor, m.key
      INTO endpoint_protocol, endpoint_vendor, endpoint_model_key
      FROM model_endpoints e
      JOIN access_paths a ON a.id = e.access_path_id
      JOIN models m ON m.id = e.model_id
     WHERE e.id = NEW.endpoint_id;

    IF endpoint_protocol IS NULL THEN
        RAISE EXCEPTION 'bot %: endpoint_id % does not exist', NEW.slug, NEW.endpoint_id
        USING ERRCODE = 'foreign_key_violation';
    END IF;

    IF NEW.harness = 'chat' AND endpoint_protocol <> 'chat-completions' THEN
        RAISE EXCEPTION 'bot %: chat harness requires chat-completions endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'codex' AND endpoint_protocol <> 'responses' THEN
        RAISE EXCEPTION 'bot %: codex harness requires responses endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-code'
          AND (endpoint_protocol <> 'anthropic-messages' OR endpoint_vendor <> 'anthropic') THEN
        RAISE EXCEPTION 'bot %: claude-code requires an Anthropic Messages endpoint',
            NEW.slug USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-proxy'
          AND (endpoint_vendor = 'anthropic'
               OR endpoint_protocol NOT IN ('anthropic-messages', 'responses', 'chat-completions')) THEN
        RAISE EXCEPTION 'bot %: claude-proxy requires a non-Anthropic proxy endpoint',
            NEW.slug USING ERRCODE = 'check_violation';
    ELSIF NEW.harness NOT IN ('chat', 'codex', 'claude-code', 'claude-proxy') THEN
        RAISE EXCEPTION 'bot %: unknown harness %', NEW.slug, NEW.harness
        USING ERRCODE = 'check_violation';
    END IF;

    -- The normalized pair is canonical. These two columns are derived mirrors
    -- retained for existing bot/runtime interfaces, never independent inputs.
    NEW.default_model := endpoint_model_key;
    NEW.agent_backend := CASE NEW.harness
        WHEN 'chat' THEN NULL
        WHEN 'claude-proxy' THEN 'claude-code'
        ELSE NEW.harness
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bot_profiles_model_backend_check ON bot_profiles;
DROP TRIGGER IF EXISTS bot_profiles_model_endpoint_check ON bot_profiles;
CREATE TRIGGER bot_profiles_model_endpoint_check
BEFORE INSERT OR UPDATE ON bot_profiles
FOR EACH ROW EXECUTE FUNCTION bot_profiles_check_model_endpoint();

DROP FUNCTION IF EXISTS bot_profiles_check_model_backend();
"""


def _compatible(harness: str, vendor: str, protocol: str) -> bool:
    if harness == "openclaw":
        return True
    if harness == "chat":
        return protocol == "chat-completions"
    if harness == "codex":
        return protocol == "responses"
    if harness == "claude-code":
        return vendor == "anthropic" and protocol == "anthropic-messages"
    if harness == "claude-proxy":
        return vendor != "anthropic" and protocol in {
            "anthropic-messages",
            "responses",
            "chat-completions",
        }
    return False


def _derive_harness(agent_backend: str | None, vendor: str, protocol: str) -> str:
    backend = (agent_backend or "").strip().lower()
    if backend == "openclaw":
        return "openclaw"
    if backend == "codex":
        return "codex"
    if backend == "claude-code":
        if vendor == "anthropic" and protocol == "anthropic-messages":
            return "claude-code"
        return "claude-proxy"
    return "chat"


def _endpoint_preference(agent_backend: str | None, vendor: str, protocol: str) -> int:
    backend = (agent_backend or "").strip().lower()
    if backend == "codex":
        return 0 if protocol == "responses" else 10
    if backend == "claude-code":
        if vendor == "anthropic" and protocol == "anthropic-messages":
            return 0
        if vendor != "anthropic" and protocol in {
            "anthropic-messages",
            "responses",
            "chat-completions",
        }:
            return 1
        return 10
    if backend == "openclaw":
        return 0
    return 0 if protocol == "chat-completions" else 10


def _seed_access_paths(conn) -> None:
    sql = text("""
        INSERT INTO access_paths
            (key, vendor, protocol, base_url, auth_mechanism, engine_kind)
        VALUES (:key, :vendor, :protocol, :base_url, :auth_mechanism, :engine_kind)
        ON CONFLICT (key) DO NOTHING
    """)
    for path in STANDARD_ACCESS_PATHS:
        conn.execute(
            sql,
            {
                "key": path.key,
                "vendor": path.vendor,
                "protocol": path.protocol,
                "base_url": path.base_url,
                "auth_mechanism": path.auth_mechanism,
                "engine_kind": path.engine_kind,
            },
        )


def _repair_bot_assignments(conn) -> list[str]:
    table_exists = conn.execute(
        text("""
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = 'bot_profiles'
        """)
    ).fetchone()
    if not table_exists:
        return []

    conn.execute(text("ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS harness VARCHAR(32)"))
    conn.execute(text("ALTER TABLE bot_profiles ADD COLUMN IF NOT EXISTS endpoint_id BIGINT"))
    conn.execute(text("ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_default_model_fk"))
    conn.execute(text("ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_harness_check"))
    conn.execute(text("ALTER TABLE bot_profiles DROP CONSTRAINT IF EXISTS bot_profiles_endpoint_fk"))
    conn.execute(text("DROP TRIGGER IF EXISTS bot_profiles_model_backend_check ON bot_profiles"))
    conn.execute(text("DROP TRIGGER IF EXISTS bot_profiles_model_endpoint_check ON bot_profiles"))

    endpoint_rows = conn.execute(text("""
        SELECT e.id, m.key, a.vendor, a.protocol
          FROM model_endpoints e
          JOIN models m ON m.id = e.model_id
          JOIN access_paths a ON a.id = e.access_path_id
         ORDER BY m.key, e.id
    """)).mappings().all()
    endpoints_by_id = {int(row["id"]): row for row in endpoint_rows}
    endpoints_by_model: dict[str, list[Any]] = {}
    for row in endpoint_rows:
        endpoints_by_model.setdefault(row["key"], []).append(row)

    repaired: list[str] = []
    bots = conn.execute(text("""
        SELECT slug, agent_backend, default_model, harness, endpoint_id
          FROM bot_profiles ORDER BY slug FOR UPDATE
    """)).mappings().all()
    for bot in bots:
        slug = str(bot["slug"])
        backend = (bot["agent_backend"] or "").strip().lower()
        harness = (bot["harness"] or "").strip().lower()
        endpoint = endpoints_by_id.get(int(bot["endpoint_id"])) if bot["endpoint_id"] is not None else None

        if backend == "openclaw" or harness == "openclaw":
            if harness == "openclaw" and endpoint is None and bot["default_model"] is None:
                continue
            conn.execute(
                text("""
                    UPDATE bot_profiles
                       SET harness = 'openclaw', endpoint_id = NULL, default_model = NULL,
                           agent_backend = 'openclaw'
                     WHERE slug = :slug
                """),
                {"slug": slug},
            )
            repaired.append(slug)
            continue

        valid = (
            endpoint is not None
            and _compatible(harness, endpoint["vendor"], endpoint["protocol"])
            and bot["default_model"] == endpoint["key"]
        )
        if valid:
            continue

        candidates = endpoints_by_model.get(str(bot["default_model"] or ""), [])
        if not candidates:
            raise RuntimeError(
                f"bot {slug}: model {bot['default_model']!r} has no normalized endpoint"
            )
        endpoint = min(
            candidates,
            key=lambda row: (
                _endpoint_preference(backend or None, row["vendor"], row["protocol"]),
                int(row["id"]),
            ),
        )
        repaired_harness = _derive_harness(
            backend or None,
            endpoint["vendor"],
            endpoint["protocol"],
        )
        if not _compatible(repaired_harness, endpoint["vendor"], endpoint["protocol"]):
            raise RuntimeError(
                f"bot {slug}: no endpoint compatible with backend={backend or 'chat'}"
            )
        conn.execute(
            text("""
                UPDATE bot_profiles
                   SET harness = :harness, endpoint_id = :endpoint_id,
                       default_model = :model_key,
                       agent_backend = CASE :harness
                           WHEN 'chat' THEN NULL
                           WHEN 'claude-proxy' THEN 'claude-code'
                           ELSE :harness
                       END
                 WHERE slug = :slug
            """),
            {
                "slug": slug,
                "harness": repaired_harness,
                "endpoint_id": int(endpoint["id"]),
                "model_key": endpoint["key"],
            },
        )
        repaired.append(slug)

    conn.execute(text("""
        ALTER TABLE bot_profiles
            ALTER COLUMN harness SET DEFAULT 'chat',
            ALTER COLUMN harness SET NOT NULL,
            ADD CONSTRAINT bot_profiles_harness_check
                CHECK (harness IN ('chat','claude-code','codex','claude-proxy','openclaw')),
            ADD CONSTRAINT bot_profiles_endpoint_fk
                FOREIGN KEY (endpoint_id) REFERENCES model_endpoints(id)
                ON UPDATE CASCADE ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
    """))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_bot_profiles_endpoint_id ON bot_profiles(endpoint_id)"))
    conn.execute(text(_BOT_TRIGGER_SQL))

    invalid = conn.execute(text("""
        SELECT b.slug
          FROM bot_profiles b
          LEFT JOIN model_endpoints e ON e.id = b.endpoint_id
          LEFT JOIN access_paths a ON a.id = e.access_path_id
         WHERE (b.harness <> 'openclaw' AND b.endpoint_id IS NULL)
            OR (b.harness = 'openclaw' AND b.endpoint_id IS NOT NULL)
            OR (b.harness = 'chat' AND a.protocol IS DISTINCT FROM 'chat-completions')
            OR (b.harness = 'codex' AND a.protocol IS DISTINCT FROM 'responses')
            OR (b.harness = 'claude-code'
                AND (a.vendor IS DISTINCT FROM 'anthropic'
                     OR a.protocol IS DISTINCT FROM 'anthropic-messages'))
            OR (b.harness = 'claude-proxy'
                AND (a.vendor IS NULL OR a.vendor = 'anthropic'
                     OR a.protocol NOT IN ('anthropic-messages','responses','chat-completions')))
    """)).scalars().all()
    if invalid:
        raise RuntimeError(f"unresolved or incompatible bot endpoint assignments: {', '.join(invalid)}")
    return repaired


def migrate_model_catalog(engine: Engine, dry_run: bool = False) -> dict[str, Any]:
    """Install/repair the normalized catalog and delete obsolete flat storage."""
    if dry_run:
        with engine.connect() as conn:
            tables = {
                row[0]
                for row in conn.execute(text("""
                    SELECT table_name FROM information_schema.tables
                     WHERE table_schema = 'public'
                       AND table_name IN ('models','model_endpoints','access_paths','bot_profiles')
                """))
            }
        return {"dry_run": True, "tables": sorted(tables)}

    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('normalized-model-catalog'))"))
        conn.execute(text(_CREATE_SCHEMA_SQL))
        _seed_access_paths(conn)
        repaired_bots = _repair_bot_assignments(conn)

        # Remove compatibility objects only after normalized repair and strict
        # validation succeed; any exception above rolls the whole transaction back.
        conn.execute(text("DROP VIEW IF EXISTS model_definitions_compat"))
        conn.execute(text("DROP TABLE IF EXISTS model_definitions CASCADE"))
        conn.execute(text("ALTER TABLE model_endpoints DROP COLUMN IF EXISTS legacy_type"))

        model_count = conn.execute(text("SELECT COUNT(*) FROM models")).scalar_one()
        endpoint_count = conn.execute(text("SELECT COUNT(*) FROM model_endpoints")).scalar_one()
        access_count = conn.execute(text("SELECT COUNT(*) FROM access_paths")).scalar_one()

    return {
        "models": model_count,
        "endpoints": endpoint_count,
        "access_paths": access_count,
        "repaired_bots": repaired_bots,
        "legacy_catalog_removed": True,
    }


def main() -> None:
    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    parser = argparse.ArgumentParser(description="Install normalized model catalog")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    engine = get_shared_engine(Config())
    if engine is None:
        raise RuntimeError("Database unavailable")
    print(json.dumps(migrate_model_catalog(engine, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
