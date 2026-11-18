"""Versioned, evidence-backed user guidance (TASK-893 slice 3). Dormant.

The active guidance for a user is one prompt registry row
(``guidance.preferences.<user_id>``); its body versions are the registry's
append-only ``prompt_template_versions``. This module adds only what the
registry lacks:

* ``guidance_proposals``: a proposed body, its diff, reason and server-resolved
  evidence, and a one-shot user decision. Approval is one transaction that
  locks the proposal and the guidance row, checks the exact base version row,
  stages the registry write and records the result row.
* ``guidance_run_receipts``: an append-only binding from one history run to
  the exact version row it used, committed together with the cursor advance.

Evidence must be a user-role row in a session owned by the proposal's user,
classified HUMAN at high or medium confidence and not a typed command.
Automation, bot traffic, legacy unknowns and other users' rows never count.
The mandate is the user's; a proposal may not change it or add keys. Only the
proposal's own user may decide it, never a bot. No caller uses this module.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from typing import Any, Mapping, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Session

from ..prompt_registry import stage_prompt_upsert
from .history_classification import HIGH, HUMAN, MEDIUM, _partitions, classify

PROPOSALS = "guidance_proposals"
RECEIPTS = "guidance_run_receipts"
SCHEMA_VERSION = 1
_DOC_KEYS = {"v", "mandate", "preferences"}
_PREF_KEYS = {"id", "statement", "status", "supersedes", "superseded_by",
              "first_seen_ts", "evidence"}
_STATUSES = {"active", "superseded"}

_DDL = (f"""
CREATE TABLE IF NOT EXISTS {PROPOSALS} (
  id                     BIGSERIAL PRIMARY KEY,
  user_id                VARCHAR(128) NOT NULL,
  guidance_key           VARCHAR(255) NOT NULL,
  base_version           INTEGER      NULL,
  base_version_row_id    BIGINT       NULL,
  proposed_body          TEXT         NOT NULL,
  diff                   TEXT         NOT NULL,
  reason                 TEXT         NOT NULL,
  evidence_json          JSONB        NOT NULL,
  status                 VARCHAR(16)  NOT NULL DEFAULT 'pending',
  proposed_by            VARCHAR(128) NOT NULL,
  decided_by             VARCHAR(128) NULL,
  decided_at             TIMESTAMPTZ  NULL,
  result_version_row_id  BIGINT       NULL,
  created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
  CHECK (status IN ('pending', 'approved', 'rejected', 'stale'))
)""", f"""
CREATE TABLE IF NOT EXISTS {RECEIPTS} (
  id                       BIGSERIAL PRIMARY KEY,
  bot_id                   VARCHAR(128)     NOT NULL,
  user_id                  VARCHAR(128)     NOT NULL,
  guidance_key             VARCHAR(255)     NOT NULL,
  guidance_version         INTEGER          NULL,
  guidance_version_row_id  BIGINT           NULL,
  body_sha256              CHAR(64)         NULL,
  cursor_before_ts         DOUBLE PRECISION NULL,
  cursor_before_id         TEXT             NULL,
  cursor_after_ts          DOUBLE PRECISION NULL,
  cursor_after_id          TEXT             NULL,
  consumed_count           INTEGER          NULL,
  status                   VARCHAR(16)      NOT NULL DEFAULT 'started',
  started_at               TIMESTAMPTZ      NOT NULL DEFAULT now(),
  finished_at              TIMESTAMPTZ      NULL,
  CHECK (status IN ('started', 'committed', 'failed'))
)""")


def ensure_schema(conn) -> None:
    """Additive and idempotent, like the other module-owned tables."""
    for ddl in _DDL:
        conn.execute(text(ddl))


def guidance_key(user_id: str) -> str:
    return f"guidance.preferences.{_user(user_id)}"


def _user(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip() or user_id == "system":
        raise ValueError("user_id must name one real user")
    return user_id


def _sha(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def canonical(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False)


# -- versions -----------------------------------------------------------------

_LATEST = text("""
    SELECT id, version, body FROM prompt_template_versions
    WHERE key = :key AND scope_type = 'global' AND scope_id = '*'
    ORDER BY version DESC, id DESC LIMIT 1
""")
_BY_ROW = text("""
    SELECT id, key, version, body FROM prompt_template_versions
    WHERE id = :id AND scope_type = 'global' AND scope_id = '*'
""")


def current_version(conn, user_id: str) -> dict[str, Any] | None:
    """The exact current version row: {row_id, version, body, sha256} or None."""
    r = conn.execute(_LATEST, {"key": guidance_key(user_id)}).mappings().first()
    return None if r is None else {"row_id": r["id"], "version": r["version"],
                                   "body": r["body"], "sha256": _sha(r["body"])}


# -- document validation ------------------------------------------------------

def _load(body: str | None) -> dict[str, Any]:
    if body is None:
        return {"v": SCHEMA_VERSION, "mandate": "", "preferences": []}
    return json.loads(body)


def validate_doc(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict) or set(doc) != _DOC_KEYS or doc["v"] != SCHEMA_VERSION:
        raise ValueError("guidance must be exactly {v, mandate, preferences}")
    if not isinstance(doc["mandate"], str) or not isinstance(doc["preferences"], list):
        raise ValueError("mandate must be text and preferences a list")
    ids = set()
    for p in doc["preferences"]:
        if not isinstance(p, dict) or not set(p) <= _PREF_KEYS or not {
                "id", "statement", "status", "evidence"} <= set(p):
            raise ValueError("preference has missing or unknown keys")
        if p["status"] not in _STATUSES or p["id"] in ids:
            raise ValueError("bad preference status or duplicate id")
        if not isinstance(p["evidence"], list):
            raise ValueError("evidence must be a list")
        for e in p["evidence"]:
            if not isinstance(e, dict) or set(e) != {"bot_id", "message_id"}:
                raise ValueError("evidence reference is exactly {bot_id, message_id}")
        ids.add(p["id"])
    by_id = {p["id"]: p for p in doc["preferences"]}
    for p in doc["preferences"]:
        if p.get("superseded_by") is not None:
            if p["status"] != "superseded" or p["superseded_by"] not in by_id:
                raise ValueError("superseded_by must name a preference and mark this superseded")
        if p.get("supersedes") is not None:
            old = by_id.get(p["supersedes"])
            if old is None or old.get("superseded_by") != p["id"]:
                raise ValueError("supersedes must link both ways")
    return doc


def _changed_prefs(base: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old = {p["id"]: p for p in base["preferences"]}
    new_ids = {p["id"] for p in new["preferences"]}
    if not set(old) <= new_ids:
        raise ValueError("preferences are superseded, never removed")
    out = []
    for p in new["preferences"]:
        prev = old.get(p["id"])
        if prev is None or (prev["statement"], prev["evidence"]) != (p["statement"], p["evidence"]):
            out.append(p)
        elif prev["status"] == "superseded" and p["status"] == "active":
            raise ValueError("a superseded preference cannot be reactivated; supersede it again")
    return out


# -- evidence -----------------------------------------------------------------

_EVIDENCE = text("""
    SELECT m.id, m.bot_id, m.content, m.timestamp, m.author_entity_type,
           m.author_entity_id, s.user_id AS owner,
           d.id AS delivery_id, d.sender_bot_id, d.target_bot_id,
           d.metadata_json AS delivery_metadata
    FROM messages m
    LEFT JOIN sessions s ON s.id = m.session_id AND s.bot_id = :slug
    LEFT JOIN inter_bot_deliveries d
      ON d.user_message_id = m.id AND d.target_bot_id = :slug
    WHERE m.bot_id = :part AND m.id = :mid AND m.role = 'user'
    ORDER BY d.id
""")


def resolve_evidence(conn, *, user_id: str, refs: Sequence[Mapping[str, str]],
                     excluded_bot_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Server-resolved facts for each reference; refuses anything not the user's own words.

    Every reference is checked before SQL against the protected-bot partition
    rule, then read from its single partition with its session owner.
    """
    user_id = _user(user_id)
    pairs = []
    for ref in refs:
        bot, mid = ref.get("bot_id"), ref.get("message_id")
        if not isinstance(bot, str) or not isinstance(mid, str) or not mid:
            raise ValueError("evidence reference needs bot_id and message_id")
        (part, slug), = _partitions([bot], excluded_bot_ids).items()
        pairs.append((part, slug, mid))
    out = []
    for part, slug, mid in pairs:
        rows = [dict(r) for r in conn.execute(
            _EVIDENCE, {"part": part, "slug": slug, "mid": mid}).mappings()]
        if not rows:
            raise ValueError(f"evidence {slug}/{mid}: no user-role row")
        owner = rows[0]["owner"]
        if owner is None:
            raise ValueError(f"evidence {slug}/{mid}: session not owned by this bot")
        if owner != user_id:
            raise ValueError(f"evidence {slug}/{mid}: belongs to another user")
        row = dict(rows[0], bot_id=slug)
        if isinstance(row.get("delivery_metadata"), str):
            row["delivery_metadata"] = json.loads(row["delivery_metadata"])
        c = classify(row)
        if c.cls != HUMAN or c.confidence not in (HIGH, MEDIUM) or c.command:
            raise ValueError(f"evidence {slug}/{mid}: {c.cls}/{c.confidence}"
                             f"{'/command' if c.command else ''} is not human preference evidence")
        out.append({"bot_id": slug, "message_id": mid, "ts": row["timestamp"],
                    "class": c.cls, "confidence": c.confidence, "occurrence": c.occurrence})
    return out


# -- proposals ----------------------------------------------------------------

_INSERT_PROPOSAL = text(f"""
    INSERT INTO {PROPOSALS} (user_id, guidance_key, base_version, base_version_row_id,
                             proposed_body, diff, reason, evidence_json, proposed_by)
    VALUES (:user_id, :key, :bv, :brow, :body, :diff, :reason, :ev, :by)
    RETURNING id
""").bindparams(bindparam("ev", type_=JSONB))

_ANY_BASE = object()


class BaseMoved(Exception):
    """The guidance is no longer the version the draft was written against."""


def propose(engine, *, user_id: str, new_doc: Mapping[str, Any], reason: str,
            proposed_by: str, excluded_bot_ids: Sequence[str],
            expected_base_row_id: Any = _ANY_BASE,
            _rollback_row: int | None = None) -> int:
    """Validate and record a pending proposal against the current exact version.

    ``expected_base_row_id`` is the version row a draft was written against
    (None = no version yet). When given, the proposal is refused with
    :class:`BaseMoved` unless the current version is exactly that row, checked
    in the inserting transaction, so a draft is never recorded on a newer base.

    ``_rollback_row`` is set only by :func:`rollback_to`: the body is an exact
    earlier version the user already approved, so the forward-only preference
    rule and evidence re-resolution do not apply; the mandate rule still does.
    """
    user_id = _user(user_id)
    if not reason or not reason.strip():
        raise ValueError("a proposal needs a reason")
    new = validate_doc(json.loads(json.dumps(new_doc)))
    with engine.begin() as conn:
        cur = current_version(conn, user_id)
        if expected_base_row_id is not _ANY_BASE and (cur and cur["row_id"]) != expected_base_row_id:
            raise BaseMoved(f"guidance moved from row {expected_base_row_id} "
                            f"to {cur and cur['row_id']}")
        base = validate_doc(_load(cur and cur["body"]))
        if new["mandate"] != base["mandate"]:
            raise ValueError("the mandate is the user's; a proposal cannot change it")
        if _rollback_row is not None:
            if canonical(new) == canonical(base):
                raise ValueError("rollback target equals the current guidance")
            evidence = [{"rollback_of_version_row_id": _rollback_row}]
        else:
            changed = _changed_prefs(base, new)
            if not changed:
                raise ValueError("proposal changes no preference")
            refs = []
            for p in changed:
                if p["status"] == "active" and not p["evidence"]:
                    raise ValueError(f"preference {p['id']} cites no evidence")
                refs.extend(p["evidence"])
            evidence = resolve_evidence(conn, user_id=user_id, refs=refs,
                                        excluded_bot_ids=excluded_bot_ids)
        body = canonical(new)
        diff = "".join(difflib.unified_diff(
            canonical(base).splitlines(True), body.splitlines(True), "base", "proposed"))
        return conn.execute(_INSERT_PROPOSAL, {
            "user_id": user_id, "key": guidance_key(user_id),
            "bv": cur and cur["version"], "brow": cur and cur["row_id"],
            "body": body, "diff": diff, "reason": reason.strip(),
            "ev": evidence, "by": proposed_by}).scalar_one()


class Stale(Exception):
    """The guidance moved since the proposal's base; the proposal is now stale."""


def _decider(conn, decided_by: str, owner: str) -> None:
    if decided_by != owner:
        raise PermissionError("only the proposal's own user may decide it")
    if conn.execute(text("SELECT to_regclass('bot_profiles')")).scalar() is None:
        raise PermissionError("bot identities unavailable; refusing to decide")
    bot = conn.execute(text("SELECT 1 FROM bot_profiles WHERE lower(slug) = lower(:u)"),
                       {"u": decided_by}).first()
    if bot is None:
        from ..bot_slug_aliases import _exists
        if _exists(conn):
            # Retired slugs are stored as their storage IDs (old_slug).
            bot = conn.execute(text(
                "SELECT 1 FROM bot_slug_aliases WHERE old_slug = COALESCE(NULLIF("
                "regexp_replace(lower(:u), '[^a-z0-9_]', '', 'g'), ''), 'default')"),
                {"u": decided_by}).first()
    if bot is not None:
        raise PermissionError("a bot cannot decide guidance")


def decide(engine, *, proposal_id: int, decided_by: str, approve: bool) -> dict[str, Any]:
    """One-shot user decision. Approval writes the next registry version atomically."""
    try:
        with engine.begin() as conn:
            p = conn.execute(text(f"SELECT * FROM {PROPOSALS} WHERE id = :id FOR UPDATE"),
                             {"id": proposal_id}).mappings().first()
            if p is None:
                raise LookupError("no such proposal")
            _decider(conn, decided_by, p["user_id"])
            if p["status"] != "pending":
                raise ValueError(f"proposal already {p['status']}")
            if not approve:
                conn.execute(text(f"""UPDATE {PROPOSALS} SET status = 'rejected',
                                     decided_by = :by, decided_at = now() WHERE id = :id"""),
                             {"id": proposal_id, "by": decided_by})
                return {"status": "rejected"}
            # Serialize writers of this guidance: lock the template row, or for
            # the first version take a transaction advisory lock on the key.
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": p["guidance_key"]})
            conn.execute(text("""SELECT id FROM prompt_templates WHERE key = :k
                                 AND scope_type = 'global' AND scope_id = '*' FOR UPDATE"""),
                         {"k": p["guidance_key"]})
            cur = current_version(conn, p["user_id"])
            if (cur and cur["row_id"]) != p["base_version_row_id"]:
                raise Stale
            with Session(bind=conn) as s:
                stage_prompt_upsert(
                    s, key=p["guidance_key"], body=p["proposed_body"], title="User guidance",
                    category="guidance", format="json", updated_by=decided_by,
                    change_note=f"proposal:{proposal_id}")
                s.flush()
            new = current_version(conn, p["user_id"])
            conn.execute(text(f"""UPDATE {PROPOSALS} SET status = 'approved', decided_by = :by,
                                  decided_at = now(), result_version_row_id = :row
                                  WHERE id = :id AND status = 'pending'"""),
                         {"id": proposal_id, "by": decided_by, "row": new["row_id"]})
            return {"status": "approved", "version": new["version"], "row_id": new["row_id"]}
    except Stale:
        with engine.begin() as conn:
            conn.execute(text(f"UPDATE {PROPOSALS} SET status = 'stale' "
                              "WHERE id = :id AND status = 'pending'"), {"id": proposal_id})
        return {"status": "stale"}


def rollback_to(engine, *, user_id: str, version_row_id: int, proposed_by: str,
                reason: str, excluded_bot_ids: Sequence[str]) -> int:
    """Propose an older exact version's body as the next version (append-only)."""
    with engine.connect() as conn:
        r = conn.execute(_BY_ROW, {"id": version_row_id}).mappings().first()
    if r is None or r["key"] != guidance_key(user_id):
        raise LookupError("no such version of this user's guidance")
    return propose(engine, user_id=user_id, new_doc=json.loads(r["body"]), reason=reason,
                   proposed_by=proposed_by, excluded_bot_ids=excluded_bot_ids,
                   _rollback_row=version_row_id)


# -- run receipts ---------------------------------------------------------------

def begin_run(engine, *, bot_id: str, user_id: str,
              excluded_bot_ids: Sequence[str]) -> dict[str, Any]:
    """Pin the exact current guidance version for one run; returns its handle."""
    (_part, slug), = _partitions([bot_id], excluded_bot_ids).items()
    user_id = _user(user_id)
    with engine.begin() as conn:
        cur = current_version(conn, user_id)
        rid = conn.execute(text(f"""
            INSERT INTO {RECEIPTS} (bot_id, user_id, guidance_key, guidance_version,
                                    guidance_version_row_id, body_sha256)
            VALUES (:b, :u, :k, :v, :row, :sha) RETURNING id
        """), {"b": slug, "u": user_id, "k": guidance_key(user_id),
               "v": cur and cur["version"], "row": cur and cur["row_id"],
               "sha": cur and cur["sha256"]}).scalar_one()
    return {"receipt_id": rid, "bot_id": slug, "user_id": user_id,
            "guidance_version": cur and cur["version"],
            "guidance_version_row_id": cur and cur["row_id"],
            "body": cur and cur["body"], "sha256": cur and cur["sha256"]}


_FINISH = text(f"""
    UPDATE {RECEIPTS}
    SET status = :status, finished_at = now(), cursor_before_ts = :bts,
        cursor_before_id = :bid, cursor_after_ts = :ats, cursor_after_id = :aid,
        consumed_count = :n
    WHERE id = :id AND bot_id = :b AND user_id = :u AND status = 'started'
    RETURNING id
""")


def finish_run(conn, *, handle: Mapping[str, Any], status: str,
               before: Sequence | None, after: Sequence | None, consumed: int | None) -> None:
    """Terminal, once: a finished receipt is immutable. Raises if not 'started'."""
    done = conn.execute(_FINISH, {
        "id": handle["receipt_id"], "b": handle["bot_id"], "u": handle["user_id"],
        "status": status, "bts": before and before[0], "bid": before and before[1],
        "ats": after and after[0], "aid": after and after[1], "n": consumed}).first()
    if done is None:
        raise ValueError("run receipt is missing or already finished")


def resolve_run(engine, receipt_id: int) -> dict[str, Any]:
    """Exactly the guidance body a past run used, verified by its sha256."""
    with engine.connect() as conn:
        r = conn.execute(text(f"SELECT * FROM {RECEIPTS} WHERE id = :id"),
                         {"id": receipt_id}).mappings().first()
        if r is None:
            raise LookupError("no such run receipt")
        body = None
        if r["guidance_version_row_id"] is not None:
            v = conn.execute(_BY_ROW, {"id": r["guidance_version_row_id"]}).mappings().first()
            if v is None or v["key"] != r["guidance_key"] or _sha(v["body"]) != r["body_sha256"]:
                raise ValueError("pinned guidance version is missing or altered")
            body = v["body"]
    return {**dict(r), "body": body}
