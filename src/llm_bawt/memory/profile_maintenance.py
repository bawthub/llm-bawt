"""Profile curation (TASK-1720): turn candidate facts into a cited diff.

Extraction records profile facts as ``candidate`` rows. A curation run hands
the model the live profile plus the candidates and applies the model's diff:

- ``add``: promote a candidate to live, optionally cleaning its key/value.
- ``update``: rewrite a live entry from one or more candidates (they retire as merged).
- ``retire``: retire a candidate, or a live entry when a cited candidate supersedes it.

Every op is validated against the rows actually given to the model. Invalid ops
are dropped and reported. A candidate the model doesn't address is retired with
that reason, so it can't trigger another paid run. Nothing is rewritten wholesale,
and the run never touches the legacy stored summary.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from llm_bawt.clients.base import LLMClient
from llm_bawt.memory.extraction.prompts import get_profile_curation_prompt
from llm_bawt.models.message import Message
from llm_bawt.profiles import (
    STATUS_CANDIDATE, STATUS_LIVE, STATUS_RETIRED, AttributeCategory, CurationConflict, EntityType,
    ProfileManager,
)
from llm_bawt.utils.error_text import safe_error_text

logger = logging.getLogger(__name__)

_KEY = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_CATEGORIES = {str(getattr(AttributeCategory, name)) for name in dir(AttributeCategory)
               if name.isupper()}
UNADDRESSED = "not addressed by curation"


@dataclass
class ProfileCurationResult:
    entity_id: str
    candidates: int
    applied: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _row(attr: Any) -> dict[str, Any]:
    return {"id": attr.id, "category": attr.category, "key": attr.key, "value": attr.value,
            "source": attr.source, "source_message_id": attr.source_message_id}


def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class ProfileCurationService:
    """Curates one entity's candidate profile facts into its live profile."""

    def __init__(self, profile_manager: ProfileManager, llm_client: LLMClient):
        self.profile_manager = profile_manager
        self.llm_client = llm_client

    def run(self, entity_id: str, entity_type: str = "user",
            dry_run: bool = False) -> ProfileCurationResult:
        kind = EntityType.USER if entity_type == "user" else EntityType.BOT
        rows = self.profile_manager.attributes_with_status(kind, entity_id,
                                                           (STATUS_LIVE, STATUS_CANDIDATE))
        live = {a.id: a for a in rows if a.status == STATUS_LIVE}
        candidates = {a.id: a for a in rows if a.status == STATUS_CANDIDATE}
        result = ProfileCurationResult(entity_id=entity_id, candidates=len(candidates))
        if not candidates:
            return result

        prompt = get_profile_curation_prompt(
            live=json.dumps([_row(a) for a in live.values()], ensure_ascii=False, indent=1),
            candidates=json.dumps([_row(a) | {"confidence": a.confidence}
                                   for a in candidates.values()], ensure_ascii=False, indent=1),
        )
        try:
            response = self.llm_client.query([Message(role="user", content=prompt)],
                                             plaintext_output=True)
        except Exception as error:  # noqa: BLE001 - reported on the task result
            logger.exception("Profile curation call failed for %s", entity_id)
            result.error = safe_error_text(error)
            return result
        if not isinstance(response, str) or response.lstrip().startswith("ERROR:"):
            result.error = "model call failed"
            return result
        ops = parse_ops(response)
        if ops is None:
            result.error = "model reply is not a JSON object with an ops list"
            return result

        changes, result.applied, result.rejected = plan_changes(ops, live, candidates)
        if not dry_run and changes:
            seen = {**live, **candidates}
            expected = {c["id"]: ProfileManager.curation_snapshot(seen[c["id"]]) for c in changes}
            try:
                self.profile_manager.apply_curation(kind, entity_id, changes, expected)
            except CurationConflict as conflict:
                # Nothing applied; remaining candidates stay pending for the next run.
                logger.warning("Profile curation %s skipped: %s", entity_id, conflict)
                result.applied = []
                result.error = "profile changed during curation; nothing applied"
                return result
        for op in result.applied:
            logger.info("Profile curation %s: %s", entity_id, json.dumps(op, ensure_ascii=False))
        for op in result.rejected:
            logger.warning("Profile curation %s rejected op: %s", entity_id,
                           json.dumps(op, ensure_ascii=False))
        return result


def parse_ops(response: str) -> list[dict[str, Any]] | None:
    """The ``ops`` list from a reply (raw JSON or a fenced block)."""
    for candidate in (re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL),
                      re.search(r"\{.*\}", response, re.DOTALL)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate.group(1) if candidate.re.groups else candidate.group(0))
        except json.JSONDecodeError:
            continue
        ops = data.get("ops") if isinstance(data, dict) else None
        if isinstance(ops, list):
            return [op for op in ops if isinstance(op, dict)]
    return None


def plan_changes(ops: list[dict[str, Any]], live: dict[int, Any], candidates: dict[int, Any]
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate ops against the given rows → (row changes, applied ops, rejected ops).

    A row is changed by at most one op. A live entry can change only with a cited
    candidate, so curation can't drop or rewrite a fact on the model's say-so alone.
    """
    changes: dict[int, dict[str, Any]] = {}
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    live_keys = {(a.category, a.key) for a in live.values()}

    def reject(op: dict[str, Any], why: str) -> None:
        rejected.append({**op, "rejected": why})

    def free(*ids: int) -> bool:
        return all(i not in changes for i in ids)

    for op in ops:
        kind = op.get("op")
        reason = _text(op.get("reason"))
        if not reason:
            reject(op, "no reason")
            continue
        cited = op.get("candidate_ids")
        if not all(_is_id(op.get(f)) for f in ("id", "candidate_id") if f in op) or (
                "candidate_ids" in op
                and not (isinstance(cited, list) and all(_is_id(c) for c in cited))):
            reject(op, "ids must be integers")
            continue
        if kind == "add":
            cid = op.get("candidate_id")
            if cid not in candidates or not free(cid):
                reject(op, "unknown or already used candidate")
                continue
            cand = candidates[cid]
            category = _text(op.get("category")) or cand.category
            key = _text(op.get("key")) or cand.key
            value = _text(op.get("value")) or _text(cand.value)
            if category not in _CATEGORIES or not _KEY.match(key) or not value:
                reject(op, "invalid category, key or value")
                continue
            if (category, key) in live_keys:
                reject(op, "a live entry already has this key; use update")
                continue
            live_keys.add((category, key))
            changes[cid] = {"id": cid, "status": STATUS_LIVE, "category": category, "key": key,
                            "value": value}
            applied.append({"op": "add", "id": cid, "category": category, "key": key,
                            "value": value, "reason": reason,
                            "sources": [cand.source_message_id]})
        elif kind == "update":
            lid = op.get("id")
            cids = op.get("candidate_ids")
            value = _text(op.get("value"))
            if lid not in live or not isinstance(cids, list) or not cids or not value:
                reject(op, "needs a live id, a value and cited candidate_ids")
                continue
            if any(c not in candidates for c in cids) or not free(lid, *cids):
                reject(op, "unknown or already used candidate")
                continue
            sources = [candidates[c].source_message_id for c in cids]
            changes[lid] = {"id": lid, "value": value,
                            "source_message_id": next((s for s in sources if s), None)}
            for c in cids:
                changes[c] = {"id": c, "status": STATUS_RETIRED,
                              "retire_reason": f"merged into {lid}: {reason}"[:500]}
            applied.append({"op": "update", "id": lid, "old": live[lid].value, "value": value,
                            "reason": reason, "candidates": cids, "sources": sources})
        elif kind == "retire":
            rid = op.get("id")
            if rid in candidates and free(rid):
                changes[rid] = {"id": rid, "status": STATUS_RETIRED, "retire_reason": reason[:500]}
                applied.append({"op": "retire", "id": rid, "reason": reason,
                                "sources": [candidates[rid].source_message_id]})
                continue
            cids = op.get("candidate_ids")
            if (rid in live and isinstance(cids, list) and cids
                    and all(c in candidates for c in cids) and free(rid)):
                changes[rid] = {"id": rid, "status": STATUS_RETIRED, "retire_reason": reason[:500]}
                applied.append({"op": "retire", "id": rid, "reason": reason, "candidates": cids,
                                "sources": [candidates[c].source_message_id for c in cids]})
                continue
            reject(op, "unknown id, or a live entry retired without a cited candidate")
        else:
            reject(op, "unknown op")

    for cid in candidates:
        if cid not in changes:
            changes[cid] = {"id": cid, "status": STATUS_RETIRED, "retire_reason": UNADDRESSED}
            applied.append({"op": "retire", "id": cid, "reason": UNADDRESSED,
                            "sources": [candidates[cid].source_message_id]})
    return list(changes.values()), applied, rejected
