"""Bounded orchestration and the sole parent-visible evidence-packet boundary."""
from __future__ import annotations

import asyncio
import copy
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
    MAX_RECORDS, MAX_SECONDS, MAX_SOURCE_RECORDS, PARTIAL_REASONS, Evidence, EvidenceBatch, RecallError,
    RecallRequest, RecallScope, Source, SourceProvider, Synthesizer,
)
from .grounding import cap_span, ground_quote
from .handles import HandleStore
from .ledger import LEDGER_WORKER, LEDGER_WRITE_SECONDS, MAX_LEDGER_ENTRIES, LedgerSink, ledger_entry


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
SHORTENED_NOTE = (" {} answer span(s) cut to fit; text omitted after a truncated quote may hold a"
                  " qualifier or correction; expand with the handle.")
OMITTED_NOTE = " {} further finding(s) omitted to fit the output budget; expand with the returned handle."
MAX_FINDINGS = 6
# The shortest prefix a lone quote is clipped to before a found answer may be
# reported as insufficient_evidence: enough to name what was found.
MIN_CLIPPED_QUOTE_CHARS = 80


class GroundedAnswer:
    """One grounded span as ordered quote parts, shrinkable without lying (TASK-1557).

    A span too long for the remaining slots keeps its head and its LAST part:
    a trailing qualifier or reversal ("... that approval was revoked") is the
    part a prefix would lose. Every kept part followed by omitted text is
    marked ``truncated``, so a cut is never silent.
    """

    def __init__(self, parts: list[dict]):
        self.parts = parts
        self.kept = list(range(len(parts)))
        self.clipped: int | None = None  # chars kept of a lone part's quote
        self.bare = False  # provenance left to the handle; the citation stays

    @property
    def cut(self) -> bool:
        return len(self.kept) < len(self.parts) or self.clipped is not None

    def shrink_to(self, slots: int) -> None:
        if slots >= len(self.parts):
            self.kept = list(range(len(self.parts)))
        elif slots == 1:
            self.kept = [0]
        else:
            self.kept = list(range(slots - 1)) + [len(self.parts) - 1]

    def clip(self) -> bool:
        """Last resort for a lone kept part: a shorter prefix of the same quote,
        then, at the floor, the citation without its provenance block.

        The prefix starts where the quote does, so it stays an exact source
        substring at the same ``start_char``. It ends on a word boundary when
        one is near and is marked ``truncated``. Provenance (speaker, session,
        time) is metadata of the cited record, so the handle still expands it.
        Returns False when neither is left to give up.
        """
        if len(self.kept) != 1:
            return False
        part = self.parts[self.kept[0]]
        quote = part["quote"]
        current = len(quote) if self.clipped is None else self.clipped
        if current <= MIN_CLIPPED_QUOTE_CHARS:
            if self.bare or "provenance" not in part:
                return False
            self.bare = True
            return True
        target = max(MIN_CLIPPED_QUOTE_CHARS, current * 2 // 3)
        head = quote[:target]
        space = head.rfind(" ")
        if space >= target // 2:
            head = head[:space]
        head = head.rstrip()
        if not head:
            return False
        self.clipped = len(head)
        return True

    def entries(self) -> list[dict]:
        entries = []
        for position, index in enumerate(self.kept):
            entry = dict(self.parts[index])
            if len(self.parts) > 1:
                # Parts of one selection read together; "k/n" shows the group.
                entry["part"] = f"{index + 1}/{len(self.parts)}"
            following = self.kept[position + 1] if position + 1 < len(self.kept) else len(self.parts)
            if following != index + 1:
                entry["truncated"] = True
            if self.clipped is not None:
                entry["quote"] = entry["quote"][:self.clipped]
                entry["truncated"] = True
            if self.bare:
                entry.pop("provenance", None)
            entries.append(entry)
        return entries

    def keys(self) -> set[tuple]:
        return {(part["source"], part["id"], part.get("start_char"), part["quote"]) for part in self.parts}


class AnswerSet:
    """The packet's findings, plus the honest notes for anything cut or omitted."""

    def __init__(self, unavailable: bool, rejected: int):
        self.answers: list[GroundedAnswer] = []
        self.unavailable = unavailable
        self.rejected = rejected
        self.omitted = 0

    def add(self, answer: GroundedAnswer) -> None:
        if any(answer.keys() <= item.keys() for item in self.answers):
            return  # the same source span selected twice
        slots = MAX_FINDINGS - sum(len(item.kept) for item in self.answers)
        if slots <= 0:
            self.omitted += 1
            return
        answer.shrink_to(slots)
        self.answers.append(answer)

    @property
    def shortened(self) -> int:
        return sum(answer.cut for answer in self.answers)

    def findings(self) -> list[dict]:
        return [entry for answer in self.answers for entry in answer.entries()]

    def drop_all(self) -> None:
        self.omitted += len(self.answers)
        self.answers.clear()

    def uncertainty(self) -> str:
        return (UNCERTAINTY + (INCOMPLETE_NOTE if self.unavailable else "")
                + (REJECTED_NOTE.format(self.rejected) if self.rejected else "")
                + (SHORTENED_NOTE.format(self.shortened) if self.shortened else "")
                + (OMITTED_NOTE.format(self.omitted) if self.omitted else ""))

    def reduce(self) -> bool:
        """Give up the least material: the last finding, then the tail-most kept
        part, then the end of the one quote left (down to a floor)."""
        if len(self.answers) > 1:
            self.answers.pop()
            self.omitted += 1
            return True
        if self.answers and len(self.answers[0].kept) > 1:
            self.answers[0].shrink_to(len(self.answers[0].kept) - 1)
            return True
        return bool(self.answers) and self.answers[0].clip()


def failure(code: str) -> dict:
    return {"status": "error", "error_code": code if code in ERROR_CODES else "source_unavailable",
            "findings": [], "handle": None}


class RecallService:
    def __init__(self, providers: list[SourceProvider], synthesizer: Synthesizer,
                 handles: HandleStore | None = None, ledger: LedgerSink | None = None):
        self.providers = {provider.source: provider for provider in providers}
        self.synthesizer = synthesizer
        self.handles = handles if handles is not None else HandleStore()
        self.ledger = ledger
        self._slots = threading.BoundedSemaphore(4)
        self._ledger_writes: set[asyncio.Task] = set()

    async def recall(self, request: RecallRequest, scope: RecallScope) -> dict:
        start = time.monotonic()
        if not self._slots.acquire(blocking=False):
            return failure("busy")
        coverage = {}
        read: list[dict] = []
        try:
            if not all(scope.owner) or scope.bot_id in scope.excluded_bot_ids:
                raise RecallError("unauthorized")
            if tokens(request.query) > MAX_QUERY_TOKENS:
                raise RecallError("query_budget")
            async with asyncio.timeout(MAX_SECONDS):
                packet = await self._research(request, scope, start, coverage, read)
            self._record(packet, scope, read)
            return packet
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

    def _record(self, packet: dict, scope: RecallScope, read: list[dict]) -> None:
        """Hand the read list to the ledger in the background (TASK-1519).

        The packet is already final: the sink gets copies, runs after this
        returns, and its failure or slowness is logged, never returned.
        """
        if self.ledger is None or not packet.get("handle"):
            return
        try:
            cited = {(item["source"], item["id"]) for item in packet["findings"]}
            entries = [{**entry, "cited": (entry["source"], entry["id"]) in cited}
                       for entry in read[:MAX_LEDGER_ENTRIES]]
            args = (packet["handle"], scope, copy.deepcopy(packet["coverage"]), entries,
                    max(0, len(read) - MAX_LEDGER_ENTRIES))
            task = asyncio.create_task(self._write_ledger(args))
        except Exception as error:
            logger.warning("recall ledger write skipped class=%s", type(error).__name__)
            return
        self._ledger_writes.add(task)
        task.add_done_callback(self._ledger_writes.discard)

    async def _write_ledger(self, args: tuple) -> None:
        try:
            await asyncio.wait_for(LEDGER_WORKER.run(self.ledger.write, *args), LEDGER_WRITE_SECONDS)
        except Exception as error:
            logger.warning("recall ledger write skipped class=%s", type(error).__name__)

    async def drain_ledger(self) -> None:
        """Wait for background ledger writes (tests and orderly shutdown)."""
        if self._ledger_writes:
            await asyncio.gather(*self._ledger_writes, return_exceptions=True)

    async def _research(self, request: RecallRequest, scope: RecallScope, start: float,
                        coverage: dict, read: list[dict] | None = None) -> dict:
        read = read if read is not None else []
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
                # The recall's own deadline still cancels; only the provider's
                # expiring here becomes TimeoutError.
                async with asyncio.timeout(getattr(provider, "max_seconds", None)):
                    rows = await provider.search(question, scope, request.task_ids)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Only retrieval failures degrade. Policy/authority/contract
                # violations remain fatal even if another source has evidence.
                if isinstance(error, RecallError) and error.code not in {
                    "source_unavailable", "timeout", "busy"}:
                    raise
                if isinstance(error, TimeoutError):
                    code = "timeout"
                else:
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
            coverage[source] = {"status": "ok" if rows else "no_match", "records": len(rows)}
            head_fallbacks = sum(row.window_kind == "head_fallback" for row in rows
                                 if isinstance(row, Evidence) and row.follows is None)
            if head_fallbacks:
                # Reported only when nonzero: coverage is paid per source in
                # every packet's output budget.
                coverage[source]["head_fallbacks"] = head_fallbacks
            depth = getattr(provider, "depth", None)
            if depth in {"key_fields", "details"}:
                coverage[source]["depth"] = depth
            if isinstance(rows, EvidenceBatch):
                # Withheld counts saturate so coverage never enumerates protected rows.
                coverage[source]["withheld"] = min(3, max(0, rows.withheld))
                reasons = sorted(set(getattr(rows, "partial", ())) & PARTIAL_REASONS)
                if reasons:
                    coverage[source]["partial_reasons"] = reasons
                if rows.withheld or reasons:
                    coverage[source]["status"] = "partial"
                coverage[source]["search_mode"] = rows.search_mode if rows.search_mode in {
                    "explicit", "semantic", "keyword", "lexical"} else "unknown"
            for row in rows:
                if (not isinstance(row, Evidence) or row.source != source
                        or row.bot_id != scope.bot_id or row.user_id != scope.user_id
                        or (row.source_bot is not None
                            and row.source_bot not in scope.allowed_bot_ids - scope.excluded_bot_ids)
                        or not isinstance(row.id, str) or not 0 < len(row.id) <= 80
                        or (row.follows is not None and row.follows not in hits)):
                    raise RecallError("unauthorized")
                if not isinstance(row.text, str) or len(row.text) > MAX_RECORD_CHARS:
                    raise RecallError("retrieval_budget")
                # Every accepted row, so the ledger count equals coverage.records.
                read.append(ledger_entry(row))
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
        answers = AnswerSet(bool(unavailable), 0)
        model_calls = 0
        model_output_tokens = 0
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
                    answers.rejected += 1
                    continue
                item = index.get((finding.source, finding.id))
                # Only exact source bytes leave; the model's own text never does.
                span = ground_quote(finding.quote, item.text) if item is not None else None
                if not span:
                    answers.rejected += 1
                    continue
                offset = item.text.find(span)
                parts = self._quote_parts(span)
                if not parts:
                    answers.omitted += 1
                    continue
                provenance = {"source_bot": item.source_bot, "session_id": item.session_id,
                              "at": item.at, "follows": item.follows, "review_ref": item.review_ref}
                provenance = {key: value for key, value in provenance.items() if value is not None}
                entries = []
                for position, quote in parts:
                    entry = {"source": finding.source, "id": finding.id, "quote": quote}
                    if provenance:
                        entry["provenance"] = provenance
                    if len(parts) > 1 or item.start_char or offset:
                        entry["start_char"] = item.start_char + offset + position
                    entries.append(entry)
                answers.add(GroundedAnswer(entries))
            if answers.rejected and not answers.answers and not answers.omitted:
                raise RecallError("ungrounded_synthesis")
        findings = answers.findings()
        incomplete = bool(evidence) or unavailable or any(item["status"] == "partial" for item in coverage.values())
        packet = {
            # A found answer stays ok when cut to fit; the cut is marked, never hidden.
            "status": "ok" if findings else ("insufficient_evidence" if incomplete else "no_match"),
            "findings": findings,
            "uncertainty": answers.uncertainty(),
            # Selected excerpts that failed schema or grounding and were omitted;
            # caller-visible so a dropped later correction is never silent.
            "rejected_findings": answers.rejected,
            "coverage": coverage,
            # Static and short: it is paid in every packet's output budget.
            "scope": ("Public searchable history for trusted user; caller memories; shared tasks; "
                      "tool metadata and clone facts, never commands."),
            "bounded": True,
            "truncated": clipped or answers.shortened > 0 or answers.omitted > 0,
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
        admitted_keys = {(row.source, row.id) for row in evidence}
        for entry in read:
            entry["admitted"] = (entry["source"], entry["id"]) in admitted_keys
        # Reserve the fixed-size handle before packing; no text retained in it.
        packet["handle"] = self.handles.put(request, scope, queries)
        self._fit(packet, request.output_tokens, answers)
        return packet

    @staticmethod
    def _quote_parts(span: str) -> list[tuple[int, str]] | None:
        """Split one grounded answer into ordered source-exact quoted spans."""
        parts = []
        offset = 0
        while offset < len(span):
            chunk = cap_span(span[offset:], MAX_QUOTE_CHARS, MAX_QUOTE_TOKENS)
            if not chunk or not chunk.strip():
                return None
            parts.append((offset, chunk))
            offset += len(chunk)
        return parts

    @staticmethod
    def _fit(packet: dict, budget: int, answers: AnswerSet):
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
            if answers.reduce():
                # Fewer findings or a marked cut, never a cleared answer: what
                # was found stays found (TASK-1557).
                packet["findings"] = answers.findings()
                packet["uncertainty"] = answers.uncertainty()
                packet["truncated"] = True
            elif packet["findings"]:
                # The one limit on "found stays found": not even one quote fits
                # beside the coverage report. The handle keeps the search intent.
                answers.drop_all()
                packet["findings"] = []
                packet["uncertainty"] = answers.uncertainty()
                packet["truncated"] = True
                packet["status"] = "insufficient_evidence"
            else:
                raise RecallError("retrieval_budget")
