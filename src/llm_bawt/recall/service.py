"""Bounded orchestration and the sole parent-visible evidence-packet boundary."""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .budgets import clip, compact_json, tokens
from .completion import input_tokens
from .contracts import (
    MAX_INPUT_TOKENS, MAX_MODEL_BYTES, MAX_QUERY_TOKENS, MAX_RECORD_CHARS,
    MAX_RECORDS, MAX_SECONDS, MAX_SOURCE_RECORDS, Evidence, EvidenceBatch, RecallError,
    RecallRequest, RecallScope, Source, SourceProvider, Synthesizer,
)
from .handles import HandleStore


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: Source
    id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1, max_length=400)


class Synthesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    findings: list[Finding] = Field(max_length=6)


# Codes are server-owned; even an injected/provider-raised RecallError may not
# smuggle source contents through the error channel.
ERROR_CODES = frozenset({
    "unauthorized", "policy_unavailable", "source_unavailable", "model_unavailable",
    "invalid_request", "invalid_handle", "handle_scope_mismatch", "followup_limit",
    "query_budget", "input_budget", "retrieval_budget", "invalid_synthesis",
    "ungrounded_synthesis", "task_provenance_incomplete", "timeout", "busy",
})


def failure(code: str) -> dict:
    return {"status": "error", "error_code": code if code in ERROR_CODES else "source_unavailable",
            "findings": [], "handle": None}


class RecallService:
    def __init__(self, providers: list[SourceProvider], synthesizer: Synthesizer,
                 handles: HandleStore | None = None):
        self.providers = {provider.source: provider for provider in providers}
        self.synthesizer = synthesizer
        self.handles = handles if handles is not None else HandleStore()
        self._slots = threading.BoundedSemaphore(4)

    async def recall(self, request: RecallRequest, scope: RecallScope) -> dict:
        start = time.monotonic()
        if not self._slots.acquire(blocking=False):
            return failure("busy")
        coverage = {}
        try:
            if not all(scope.owner) or scope.bot_id in scope.excluded_bot_ids:
                raise RecallError("unauthorized")
            if tokens(request.query) > MAX_QUERY_TOKENS:
                raise RecallError("query_budget")
            async with asyncio.timeout(MAX_SECONDS):
                return await self._research(request, scope, start, coverage)
        except RecallError as error:
            result = failure(error.code)
        except TimeoutError:
            result = failure("timeout")
        except Exception:
            # Source contents, HTTP exception details and invalid model output
            # never escape, including through stack traces returned by MCP.
            result = failure("source_unavailable")
        finally:
            self._slots.release()
        if coverage:
            for item in coverage.values():
                if item["status"] == "searching":
                    item["status"] = result["error_code"]
            result["coverage"] = coverage
        return result

    async def _research(self, request: RecallRequest, scope: RecallScope, start: float,
                        coverage: dict) -> dict:
        queries = self.handles.resolve(request, scope)
        # Explicitly retain the initial intent AND subsequent questions. Never
        # concatenate previous answers/source bodies into the new parent turn.
        question = "\n".join(queries)
        evidence: list[Evidence] = []
        seen = set()
        clipped = False
        for source in request.sources:
            coverage[source] = {"status": "searching", "records": 0}
            provider = self.providers.get(source)
            if provider is None:
                raise RecallError("source_unavailable")
            rows = await provider.search(question, scope, request.task_ids)
            if not isinstance(rows, list) or len(rows) > MAX_SOURCE_RECORDS:
                raise RecallError("retrieval_budget")
            coverage[source] = {"status": "ok" if rows else "no_match", "records": len(rows),
                                "head_fallbacks": sum(row.window_kind == "head_fallback" for row in rows
                                                      if isinstance(row, Evidence))}
            if isinstance(rows, EvidenceBatch):
                coverage[source]["withheld"] = min(3, max(0, rows.withheld))
                if rows.withheld:
                    coverage[source]["status"] = "partial"
                coverage[source]["search_mode"] = rows.search_mode if rows.search_mode in {
                    "explicit", "semantic", "keyword", "lexical"} else "unknown"
            for row in rows:
                if (not isinstance(row, Evidence) or row.source != source
                        or row.bot_id != scope.bot_id or row.user_id != scope.user_id
                        or not isinstance(row.id, str) or not 0 < len(row.id) <= 80):
                    raise RecallError("unauthorized")
                if not isinstance(row.text, str) or len(row.text) > MAX_RECORD_CHARS:
                    raise RecallError("retrieval_budget")
                key = (row.source, row.id)
                if key in seen or not row.text.strip():
                    continue
                seen.add(key)
                excerpt = clip(row.text, 512)
                clipped |= excerpt != row.text or row.truncated
                candidate = replace(row, text=excerpt)
                if len(evidence) >= MAX_RECORDS or input_tokens(question, [*evidence, candidate]) > MAX_INPUT_TOKENS:
                    clipped = True
                    continue
                evidence.append(candidate)
        findings = []
        model_calls = 0
        model_output_tokens = 0
        if evidence:
            model_calls = 1
            raw = await self.synthesizer.synthesize(question, evidence, request.output_tokens)
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_MODEL_BYTES:
                raise RecallError("invalid_synthesis")
            model_output_tokens = tokens(raw)
            if model_output_tokens > request.output_tokens:
                raise RecallError("invalid_synthesis")
            try:
                parsed = Synthesis.model_validate_json(raw)
            except (ValidationError, ValueError) as error:
                raise RecallError("invalid_synthesis") from error
            index = {(item.source, item.id): item for item in evidence}
            for finding in parsed.findings:
                item = index.get((finding.source, finding.id))
                if (item is None or finding.quote not in item.text
                        or not finding.quote.strip() or tokens(finding.quote) > 100):
                    raise RecallError("ungrounded_synthesis")
                entry = finding.model_dump()
                if entry not in findings:
                    findings.append(entry)
        incomplete = bool(evidence) or any(item["status"] == "partial" for item in coverage.values())
        packet = {
            "status": "ok" if findings else ("insufficient_evidence" if incomplete else "no_match"),
            "findings": findings,
            "uncertainty": "Extractive untrusted historical evidence; not exhaustive or proof of current state.",
            "coverage": coverage,
            "scope": "Caller history; attributed memories; shared tasks exclude responses; tool metadata only.",
            "bounded": True,
            "truncated": clipped,
            "handle": None,
            "metrics": {
                "tokenizer": "o200k_base",
                "retrieval_calls": len(request.sources),
                "model_calls": model_calls,
                "evidence_tokens": sum(tokens(row.text) for row in evidence),
                "utility_input_tokens": input_tokens(question, evidence) if model_calls else 0,
                "utility_output_tokens": model_output_tokens,
                "parent_tokens": 0,
                "elapsed_ms": round((time.monotonic() - start) * 1000),
            },
        }
        # Reserve the fixed-size handle before packing; no text retained in it.
        packet["handle"] = self.handles.put(request, scope, queries)
        self._fit(packet, request.output_tokens)
        return packet

    @staticmethod
    def _fit(packet: dict, budget: int):
        # Count exactly the compact JSON returned by the MCP tool, including
        # citations, coverage, handle and measurements. Never return raw fallback.
        while True:
            for _ in range(5):
                count = tokens(compact_json(packet))
                if packet["metrics"]["parent_tokens"] == count:
                    break
                packet["metrics"]["parent_tokens"] = count
            if tokens(compact_json(packet)) <= budget:
                return
            if packet["findings"]:
                packet["findings"].pop()
                packet["truncated"] = True
                if not packet["findings"]:
                    packet["status"] = "insufficient_evidence"
            else:
                raise RecallError("retrieval_budget")
