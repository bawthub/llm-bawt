"""Deterministic attribution of live workspace changes to durable turn snapshots.

TASK-729.  This module is intentionally transport- and Git-agnostic: callers
provide current before/after text plus durable turn snapshots.  The same engine
can therefore serve the workspace API and commit-recon without confidence math
drifting between two implementations.
"""

from __future__ import annotations

import difflib
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

Confidence = Literal["exact", "strong", "partial", "probable"]
Ownership = Literal["owned", "shared", "unattributed"]


@dataclass(frozen=True)
class LineChange:
    kind: Literal["+", "-"]
    text: str
    hunk: int
    line: int


@dataclass(frozen=True)
class CurrentWorkspaceFile:
    repo_id: str
    repo_key: str
    repo_aliases: tuple[str, ...]
    path: str
    staged: bool
    status: str
    old_text: str
    new_text: str
    binary: bool = False
    truncated: bool = False
    changes: tuple[LineChange, ...] | None = None


@dataclass(frozen=True)
class TurnFileSnapshot:
    turn_id: str
    trigger_message_id: str | None
    bot_id: str | None
    user_id: str | None
    created_at: datetime
    repo_key: str
    repo_label: str | None
    path: str
    old_path: str | None
    change_kind: str
    before_sha256: str | None
    after_sha256: str | None
    before_text: str
    after_text: str
    binary: bool
    truncated: bool
    source_tool_call_ids: tuple[str, ...] = ()
    prompt: str | None = None


@dataclass
class AttributionCandidate:
    turn_id: str
    trigger_message_id: str | None
    bot_id: str | None
    created_at: datetime
    confidence: Confidence
    matched_changes: int
    dirty_changes: int
    candidate_changes: int
    dirty_coverage: float
    candidate_coverage: float
    dirty_hunks: list[int]
    exact_transition: bool
    final_snapshot: bool
    source_tool_call_ids: tuple[str, ...]
    prompt: str | None
    truncated: bool
    _matched_indices: list[int] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class FileAttribution:
    repo_id: str
    repo_key: str
    path: str
    staged: bool
    ownership: Ownership
    candidates: tuple[AttributionCandidate, ...]
    unmatched_changes: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/").lower()


def line_changes(before: str, after: str) -> list[LineChange]:
    old = before.splitlines(keepends=True)
    new = after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    changes: list[LineChange] = []
    hunk = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunk += 1
        if tag in {"delete", "replace"}:
            changes.extend(
                LineChange("-", text, hunk, i1 + offset + 1)
                for offset, text in enumerate(old[i1:i2])
            )
        if tag in {"insert", "replace"}:
            changes.extend(
                LineChange("+", text, hunk, j1 + offset + 1)
                for offset, text in enumerate(new[j1:j2])
            )
    return changes


def _overlap(candidate: list[LineChange], dirty: list[LineChange]) -> tuple[list[int], list[int]]:
    available: dict[tuple[str, str], list[tuple[int, LineChange]]] = defaultdict(list)
    for index, change in enumerate(dirty):
        available[(change.kind, change.text)].append((index, change))

    matched_indices: list[int] = []
    matched_hunks: set[int] = set()
    for change in candidate:
        choices = available.get((change.kind, change.text), [])
        if not choices:
            continue
        choice_index = min(
            range(len(choices)),
            key=lambda index: abs(choices[index][1].line - change.line),
        )
        if abs(choices[choice_index][1].line - change.line) > 12:
            continue
        dirty_index, dirty_change = choices.pop(choice_index)
        matched_indices.append(dirty_index)
        matched_hunks.add(dirty_change.hunk)
    return sorted(set(matched_indices)), sorted(matched_hunks)


class SnapshotLocation(Protocol):
    """The metadata needed to screen rows before resolving their diff bytes."""

    repo_key: str
    repo_label: str | None
    path: str


def _repo_matches(current: CurrentWorkspaceFile, snapshot: SnapshotLocation) -> bool:
    aliases = {normalize_path(current.repo_key), *(normalize_path(value) for value in current.repo_aliases)}
    stored = {normalize_path(snapshot.repo_key), normalize_path(snapshot.repo_label or "")}
    stored.discard("")
    if aliases & stored:
        return True
    snapshot_path = normalize_path(snapshot.path)
    return any(alias and f"/{alias}/" in f"/{snapshot_path}/" for alias in aliases)


def _path_matches(current: CurrentWorkspaceFile, snapshot: SnapshotLocation) -> bool:
    target = normalize_path(current.path)
    stored = normalize_path(snapshot.path)
    if stored == target or stored.endswith(f"/{target}"):
        return True
    return any(
        stored == normalize_path(f"{alias}/{current.path}")
        for alias in current.repo_aliases
        if alias
    )


def matches_snapshot_location(current: CurrentWorkspaceFile, snapshot: SnapshotLocation) -> bool:
    """Use the attribution engine's exact repo/path rules before loading blobs.

    Capture repo_key may be ``workspace`` rather than the tracked repo, while
    repo_label or an absolute path still identifies it. Keep both checks shared
    with attribution so prefiltering can never discard a candidate it would use.
    """
    return _repo_matches(current, snapshot) and _path_matches(current, snapshot)


def _candidate(current: CurrentWorkspaceFile, snapshot: TurnFileSnapshot) -> AttributionCandidate | None:
    if not matches_snapshot_location(current, snapshot):
        return None

    dirty = [] if current.binary else list(current.changes or line_changes(current.old_text, current.new_text))
    historical = [] if snapshot.binary else line_changes(snapshot.before_text, snapshot.after_text)
    current_before_sha = sha256_text(current.old_text)
    current_after_sha = sha256_text(current.new_text)
    exact_transition = (
        snapshot.before_sha256 == current_before_sha
        and snapshot.after_sha256 == current_after_sha
        and not current.truncated
        and not snapshot.truncated
    )
    final_snapshot = (
        snapshot.after_sha256 == current_after_sha
        and not current.truncated
        and not snapshot.truncated
    )

    matched_indices, dirty_hunks = _overlap(historical, dirty)
    matched = len(matched_indices)
    dirty_count = len(dirty)
    candidate_count = len(historical)
    dirty_coverage = matched / dirty_count if dirty_count else 0.0
    candidate_coverage = matched / candidate_count if candidate_count else 0.0

    if exact_transition or (dirty_count > 0 and dirty_coverage == 1 and candidate_coverage == 1):
        confidence: Confidence = "exact"
        if dirty_count and not matched_indices:
            matched_indices = list(range(dirty_count))
            dirty_hunks = sorted({change.hunk for change in dirty})
            matched = dirty_count
            dirty_coverage = 1.0
            candidate_coverage = 1.0
    elif final_snapshot or dirty_coverage >= 0.8:
        confidence = "strong"
    elif matched > 0:
        confidence = "partial"
    else:
        # A path/tool record is still useful provenance, but never sufficient for
        # a direct safe action without matching current bytes.
        confidence = "probable"

    return AttributionCandidate(
        turn_id=snapshot.turn_id,
        trigger_message_id=snapshot.trigger_message_id,
        bot_id=snapshot.bot_id,
        created_at=snapshot.created_at,
        confidence=confidence,
        matched_changes=matched,
        dirty_changes=dirty_count,
        candidate_changes=candidate_count,
        dirty_coverage=round(dirty_coverage, 4),
        candidate_coverage=round(candidate_coverage, 4),
        dirty_hunks=dirty_hunks,
        exact_transition=exact_transition,
        final_snapshot=final_snapshot,
        source_tool_call_ids=snapshot.source_tool_call_ids,
        prompt=snapshot.prompt,
        truncated=current.truncated or snapshot.truncated,
        _matched_indices=matched_indices,
    )


def attribute_workspace(
    current_files: list[CurrentWorkspaceFile],
    snapshots: list[TurnFileSnapshot],
) -> list[FileAttribution]:
    """Attribute every current file deterministically; never drop dirty scope."""
    output: list[FileAttribution] = []
    ordered_snapshots = sorted(snapshots, key=lambda item: item.created_at, reverse=True)

    for current in current_files:
        candidates = [
            candidate
            for snapshot in ordered_snapshots
            if (candidate := _candidate(current, snapshot)) is not None
        ]
        dirty_changes = [] if current.binary else list(current.changes or line_changes(current.old_text, current.new_text))
        dirty_count = len(dirty_changes)
        remaining = set(range(dirty_count))
        selected: list[AttributionCandidate] = []

        for candidate in candidates:
            new_matches = sorted(set(candidate._matched_indices) & remaining)
            if new_matches:
                candidate.matched_changes = len(new_matches)
                candidate.dirty_coverage = round(len(new_matches) / dirty_count, 4) if dirty_count else 0.0
                candidate.dirty_hunks = sorted(
                    {dirty_changes[index].hunk for index in new_matches}
                )
                selected.append(candidate)
                remaining -= set(new_matches)
            elif not selected and candidate.confidence == "probable":
                selected.append(candidate)

        if not selected and candidates:
            selected = [candidates[0]]

        proven = [candidate for candidate in selected if candidate.confidence != "probable"]
        if not selected:
            ownership: Ownership = "unattributed"
        elif len(proven) == 1 and not remaining and proven[0].confidence in {"exact", "strong"}:
            ownership = "owned"
        elif proven:
            ownership = "shared"
        else:
            ownership = "unattributed"

        output.append(FileAttribution(
            repo_id=current.repo_id,
            repo_key=current.repo_key,
            path=current.path,
            staged=current.staged,
            ownership=ownership,
            candidates=tuple(selected),
            unmatched_changes=len(remaining),
        ))

    return output
