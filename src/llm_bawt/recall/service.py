"""Bounded orchestration and the sole parent-visible evidence-packet boundary."""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter
import time
from dataclasses import replace

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .budgets import clip, compact_json, tokens
from .completion import input_tokens
from .contracts import (
    MAX_FOLLOW_ONS_PER_HIT, MAX_INPUT_TOKENS, MAX_MODEL_BYTES, MAX_QUERY_TOKENS, MAX_QUOTE_CHARS,
    MAX_QUOTE_TOKENS, MAX_RECORD_CHARS,
    MAX_RECORDS, MAX_SECONDS, MAX_SOURCE_RECORDS, Evidence, EvidenceBatch, RecallError,
    RecallRequest, RecallScope, Source, SourceProvider, Synthesizer,
)
from .grounding import cap_span, ground_quote
from .handles import HandleStore


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: Source
    id: str = Field(min_length=1, max_length=80)
    # Not length-capped here: grounding maps it to source bytes, then cap_span
    # shortens that span. The whole reply is already bounded by MAX_MODEL_BYTES.
    quote: str = Field(min_length=1)


class Synthesis(BaseModel):
    # Envelope is strict; each finding is validated and grounded on its own so
    # one malformed or misquoted pick cannot discard the grounded ones.
    model_config = ConfigDict(extra="forbid", strict=True)
    findings: list[Any] = Field(max_length=6)


# Codes are server-owned; even an injected/provider-raised RecallError may not
# smuggle source contents through the error channel.
ERROR_CODES = frozenset({
    "unauthorized", "policy_unavailable", "source_unavailable", "model_unavailable",
    "invalid_request", "invalid_handle", "handle_scope_mismatch", "followup_limit",
    "query_budget", "input_budget", "retrieval_budget", "invalid_synthesis",
    "ungrounded_synthesis", "task_provenance_incomplete", "timeout", "busy",
})


logger = logging.getLogger(__name__)
UNCERTAINTY = "Extractive untrusted historical evidence; not exhaustive or proof of current state."
INCOMPLETE_NOTE = " Source coverage incomplete; an unavailable source may contain a later correction."
REJECTED_NOTE = (" {} selected excerpt(s) could not be verified against the source and were omitted;"
                 " a later correction may be missing.")
SHORTENED_NOTE = (" {} answer span(s) could not fit without losing a trailing qualifier or correction;"
                  " expand with the returned handle.")


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
        # Registry-resolved once per recall (blocking DB read off the loop); the
        # same text is budgeted below and sent, so a live edit can't skew either.
        system = await asyncio.to_thread(self.synthesizer.system_prompt, scope.bot_id)
        if not isinstance(system, str) or not system.strip():
            raise RecallError("model_unavailable")
        evidence: list[Evidence] = []
        follow_ons: list[Evidence] = []
        seen = set()
        clipped = False

        def admit(candidate: Evidence) -> None:
            nonlocal clipped
            if (len(evidence) >= MAX_RECORDS
                    or input_tokens(system, question, [*evidence, candidate]) > MAX_INPUT_TOKENS):
                clipped = True
                return
            evidence.append(candidate)

        for source in request.sources:
            coverage[source] = {"status": "searching", "records": 0}
            provider = self.providers.get(source)
            source_start = time.monotonic()
            try:
                if provider is None:
                    raise RecallError("source_unavailable")
                rows = await provider.search(question, scope, request.task_ids)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Only retrieval failures degrade. Policy/authority/contract
                # violations remain fatal even if another source has evidence.
                if isinstance(error, RecallError) and error.code not in {
                    "source_unavailable", "timeout", "busy"}:
                    raise
                code = error.code if isinstance(error, RecallError) and error.code in {
                    "timeout", "busy"} else "source_unavailable"
                root = error.__cause__ or error
                logger.warning("recall source failed source=%s class=%s elapsed_ms=%d",
                               source, type(root).__name__, round((time.monotonic() - source_start) * 1000))
                coverage[source] = {"status": "unavailable", "records": 0, "error_code": code}
                continue
            if not isinstance(rows, list):
                raise RecallError("retrieval_budget")
            hits = {row.id for row in rows if isinstance(row, Evidence) and row.follows is None}
            row_limit = min(getattr(provider, "max_records", MAX_SOURCE_RECORDS),
                            MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT))
            per_hit = Counter(row.follows for row in rows if isinstance(row, Evidence) and row.follows)
            if (len(hits) > MAX_SOURCE_RECORDS or len(rows) > row_limit
                    or any(count > MAX_FOLLOW_ONS_PER_HIT for count in per_hit.values())):
                raise RecallError("retrieval_budget")
            coverage[source] = {"status": "ok" if rows else "no_match", "records": len(rows),
                                "head_fallbacks": sum(row.window_kind == "head_fallback" for row in rows
                                                      if isinstance(row, Evidence) and row.follows is None)}
            if isinstance(rows, EvidenceBatch):
                coverage[source]["withheld"] = min(3, max(0, rows.withheld))
                if rows.withheld:
                    coverage[source]["status"] = "partial"
                coverage[source]["search_mode"] = rows.search_mode if rows.search_mode in {
                    "explicit", "semantic", "keyword", "lexical"} else "unknown"
            for row in rows:
                if (not isinstance(row, Evidence) or row.source != source
                        or row.bot_id != scope.bot_id or row.user_id != scope.user_id
                        or not isinstance(row.id, str) or not 0 < len(row.id) <= 80
                        or (row.follows is not None and row.follows not in hits)):
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
                if row.follows is None:
                    admit(candidate)
                else:
                    follow_ons.append(candidate)
        # A total source outage cannot be interpreted as an empty search.
        unavailable = any(item["status"] == "unavailable" for item in coverage.values())
        if unavailable and all(item["status"] == "unavailable" for item in coverage.values()):
            raise RecallError("source_unavailable")
        # Follow-ons never displace any source's direct hits; they fill whatever
        # remains of the same global record/token caps, behind an admitted hit.
        admitted = {(row.source, row.id) for row in evidence}
        for candidate in follow_ons:
            if (candidate.source, candidate.follows) in admitted:
                admit(candidate)
        # Oldest-first so later corrections read as later; undated rows last.
        evidence.sort(key=lambda row: (row.at is None, row.at or ""))
        findings = []
        model_calls = 0
        model_output_tokens = 0
        rejected = 0
        shortened = 0
        if evidence:
            model_calls = 1
            raw = await self.synthesizer.synthesize(system, question, evidence, request.output_tokens)
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
            for proposed in parsed.findings:
                try:
                    finding = Finding.model_validate(proposed)
                except ValidationError:
                    rejected += 1
                    continue
                item = index.get((finding.source, finding.id))
                # Only exact source bytes leave; the model's own text never does.
                span = ground_quote(finding.quote, item.text) if item is not None else None
                if not span:
                    rejected += 1
                    continue
                offset = item.text.find(span)
                parts = self._quote_parts(span)
                if not parts or len(findings) + len(parts) > 6:
                    # A prefix alone could turn a reversal into a false claim.
                    shortened += 1
                    clipped = True
                    continue
                provenance = {"source_bot": item.source_bot, "session_id": item.session_id,
                              "at": item.at, "follows": item.follows, "review_ref": item.review_ref}
                provenance = {key: value for key, value in provenance.items() if value is not None}
                for position, quote in parts:
                    entry = {"source": finding.source, "id": finding.id, "quote": quote}
                    if provenance:
                        entry["provenance"] = provenance
                    if len(parts) > 1 or item.start_char or offset:
                        entry["start_char"] = item.start_char + offset + position
                    if entry not in findings:
                        findings.append(entry)
            if rejected and not findings:
                raise RecallError("ungrounded_synthesis")
        incomplete = bool(evidence) or unavailable or any(item["status"] == "partial" for item in coverage.values())
        packet = {
            "status": "ok" if findings and not shortened else ("insufficient_evidence" if incomplete else "no_match"),
            "findings": findings if not shortened else [],
            "uncertainty": (UNCERTAINTY + (INCOMPLETE_NOTE if unavailable else "")
                            + (REJECTED_NOTE.format(rejected) if rejected else "")
                            + (SHORTENED_NOTE.format(shortened) if shortened else "")),
            # Selected excerpts that failed schema or grounding and were omitted;
            # caller-visible so a dropped later correction is never silent.
            "rejected_findings": rejected,
            "coverage": coverage,
            "scope": "Public searchable history for trusted user; caller memories; shared tasks exclude responses; tool metadata only.",
            "bounded": True,
            "truncated": clipped,
            "handle": None,
            "metrics": {
                "tokenizer": "o200k_base",
                "retrieval_calls": len(request.sources),
                "model_calls": model_calls,
                "evidence_tokens": sum(tokens(row.text) for row in evidence),
                "utility_input_tokens": input_tokens(system, question, evidence) if model_calls else 0,
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
    def _quote_parts(span: str) -> list[tuple[int, str]] | None:
        """Split one grounded answer into ordered source-exact quoted spans."""
        parts = []
        offset = 0
        while offset < len(span):
            chunk = cap_span(span[offset:], MAX_QUOTE_CHARS, MAX_QUOTE_TOKENS)
            if not chunk or not chunk.strip() or len(parts) == 6:
                return None
            parts.append((offset, chunk))
            offset += len(chunk)
        return parts

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
                # A partial quoted answer is worse than an explicit insufficiency.
                # The handle retains search intent so the caller can expand.
                packet["findings"].clear()
                packet["truncated"] = True
                packet["status"] = "insufficient_evidence"
            else:
                raise RecallError("retrieval_budget")
