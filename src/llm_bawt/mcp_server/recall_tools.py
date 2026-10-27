"""Thin trusted-capability adapter for isolated, read-only recall."""
from __future__ import annotations

import asyncio
from functools import lru_cache

from pydantic import ValidationError

from llm_bawt.recall.budgets import compact_json
from llm_bawt.recall.completion import UtilitySynthesizer
from llm_bawt.recall.contracts import MAX_SECONDS, RecallError, RecallRequest
from llm_bawt.recall.ledger import PostgresLedger
from llm_bawt.recall.postgres import MessageSource, MemorySource, RecallDatabase, ToolEvidenceSource
from llm_bawt.recall.service import RecallService, failure
from llm_bawt.recall.task_source import TaskSource
from llm_bawt.task_turn_context import TaskTurnContextError

from .registry import mcp
from .task_association import current_task_turn_context


@lru_cache(maxsize=1)
def _runtime() -> tuple[RecallDatabase, RecallService]:
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    config = Config()
    database = RecallDatabase(get_shared_engine(config))
    host = config.SERVICE_HOST
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    service = RecallService(
        [MessageSource(database), MemorySource(database), ToolEvidenceSource(database), TaskSource()],
        UtilitySynthesizer(f"http://{host}:{config.SERVICE_PORT}"),
        # UI drill-down references only (TASK-1519); never read back by bots.
        ledger=PostgresLedger(database.engine),
    )
    return database, service


@mcp.tool(name="context_recall")
async def context_recall(
    request: RecallRequest,
) -> str:
    """Research caller-owned history in isolated utility context; return bounded cited quotes.

    Selecting tasks explicitly opts into shared task search; task_ids optionally
    narrows it. Follow-ups must repeat sources/task_ids, retain intent, and refresh
    evidence. All identity is inherited from the signed current-turn capability.
    No raw results or model errors ever return through this adapter.
    """
    try:
        context = current_task_turn_context()
        async with asyncio.timeout(MAX_SECONDS):
            database, service = _runtime()
            scope = await database.scope(context)
            result = await service.recall(request, scope)
    except TaskTurnContextError:
        result = failure("unauthorized")
    except ValidationError:
        result = failure("invalid_request")
    except RecallError as error:
        result = failure(error.code)
    except TimeoutError:
        result = failure("timeout")
    except Exception:
        result = failure("source_unavailable")
    # A string intentionally avoids FastMCP's second structured-output copy.
    return compact_json(result)
