"""Read side of user guidance for the approval page (TASK-893 slice 5).

Read-only: when the guidance tables do not exist yet it returns an empty view
and never runs DDL. Evidence excerpts are read at display time from the active
messages table only, after the protected-partition check, so a forgotten or
newly protected source shows as unavailable rather than as verified text.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from sqlalchemy import text

from . import guidance
from .history_classification import _partitions

EXCERPT_CHARS = 280
HISTORY_LIMIT = 20

_EXCERPT = text("""
    SELECT m.content, m.timestamp, s.user_id AS owner
    FROM messages m
    LEFT JOIN sessions s ON s.id = m.session_id AND s.bot_id = :slug
    WHERE m.bot_id = :part AND m.id = :mid AND m.role = 'user'
    LIMIT 1
""")


def _exists(conn, table: str) -> bool:
    return conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar() is not None


def _prefs(body: str | None) -> dict[str, dict[str, Any]]:
    doc = guidance._load(body)
    return {p["id"]: p for p in doc.get("preferences", [])}


def changes(base_body: str | None, proposed_body: str) -> list[dict[str, Any]]:
    """Every per-preference difference from base to proposed; none is omitted.

    ``evidence`` is the proposed preference's complete citation list, each
    marked ``new`` when the base did not cite it; ``removed_evidence`` lists
    citations the base had and the proposal drops. A difference that fits no
    named kind is reported as ``changed`` so approval never hides a mutation.
    """
    base, new = _prefs(base_body), _prefs(proposed_body)
    out = []
    for pid, p in new.items():
        old = base.get(pid)
        if old == p:
            continue
        old_refs = old.get("evidence", []) if old else []
        refs = p.get("evidence", [])
        removed = [r for r in old_refs if r not in refs]
        if old is None:
            kind = "added"
        elif old.get("status") == "active" and p.get("status") == "superseded":
            kind = "retired"
        elif old.get("statement") != p.get("statement"):
            kind = "reworded"
        elif {k: v for k, v in old.items() if k != "evidence"} == \
                {k: v for k, v in p.items() if k != "evidence"}:
            kind = "evidence_changed"
        else:
            kind = "changed"
        out.append({"id": pid, "kind": kind,
                    "before": old and old.get("statement"), "after": p.get("statement"),
                    "status_before": old and old.get("status"), "status_after": p.get("status"),
                    "superseded_by": p.get("superseded_by"),
                    "evidence": [dict(r, new=r not in old_refs) for r in refs],
                    "removed_evidence": removed})
    for pid, old in base.items():
        if pid not in new:
            out.append({"id": pid, "kind": "removed", "before": old.get("statement"),
                        "after": None, "status_before": old.get("status"), "status_after": None,
                        "superseded_by": None, "evidence": [],
                        "removed_evidence": old.get("evidence", [])})
    return out


def excerpt(conn, *, user_id: str, ref: Mapping[str, Any],
            protected: Sequence[str]) -> dict[str, Any]:
    """One cited message as plain data, or unavailable. Protection is checked before SQL."""
    bot, mid = ref.get("bot_id"), ref.get("message_id")
    base = {"bot_id": bot, "message_id": mid}
    if not isinstance(bot, str) or not isinstance(mid, str) or not mid:
        return {**base, "unavailable": True}
    try:
        (part, slug), = _partitions([bot], protected).items()
    except ValueError:
        return {**base, "unavailable": True}
    r = conn.execute(_EXCERPT, {"part": part, "slug": slug, "mid": mid}).mappings().first()
    if r is None or r["owner"] != user_id:
        return {**base, "unavailable": True}
    content = r["content"] or ""
    if len(content) > EXCERPT_CHARS:
        content = content[:EXCERPT_CHARS - 1] + "…"
    return {**base, "unavailable": False, "ts": r["timestamp"], "text": content}


def _proposal(conn, p: Mapping[str, Any], *, user_id: str, protected: Sequence[str],
              with_excerpts: bool) -> dict[str, Any]:
    base_body = None
    if p["base_version_row_id"] is not None:
        r = conn.execute(guidance._BY_ROW, {"id": p["base_version_row_id"]}).mappings().first()
        base_body = r and r["body"]
    ch = changes(base_body, p["proposed_body"])
    if with_excerpts:
        for c in ch:
            c["evidence"] = [dict(excerpt(conn, user_id=user_id, ref=r, protected=protected),
                                  new=r.get("new", False)) for r in c["evidence"]]
            c["removed_evidence"] = [excerpt(conn, user_id=user_id, ref=r, protected=protected)
                                     for r in c["removed_evidence"]]
    return {"id": p["id"], "status": p["status"], "reason": p["reason"],
            "proposed_by": p["proposed_by"], "created_at": p["created_at"],
            "decided_by": p["decided_by"], "decided_at": p["decided_at"],
            "base_version": p["base_version"], "changes": ch, "diff": p["diff"]}


def view(conn, *, user_id: str, protected: Sequence[str]) -> dict[str, Any]:
    """Current guidance, pending and recent proposals, and version list for one user."""
    user_id = guidance._user(user_id)
    out: dict[str, Any] = {"user_id": user_id, "current": None, "pending": [],
                           "history": [], "versions": []}
    if _exists(conn, "prompt_template_versions"):
        cur = guidance.current_version(conn, user_id)
        if cur is not None:
            doc = guidance._load(cur["body"])
            out["current"] = {"version": cur["version"], "row_id": cur["row_id"],
                              "mandate": doc.get("mandate", ""),
                              "preferences": doc.get("preferences", [])}
        out["versions"] = [dict(r) for r in conn.execute(text("""
            SELECT id AS row_id, version, created_at, change_note FROM prompt_template_versions
            WHERE key = :k AND scope_type = 'global' AND scope_id = '*'
            ORDER BY version DESC, id DESC LIMIT :n"""),
            {"k": guidance.guidance_key(user_id), "n": HISTORY_LIMIT}).mappings()]
    if not _exists(conn, guidance.PROPOSALS):
        return out
    rows = conn.execute(text(f"""
        (SELECT * FROM {guidance.PROPOSALS} WHERE user_id = :u AND status = 'pending'
         ORDER BY id)
        UNION ALL
        (SELECT * FROM {guidance.PROPOSALS} WHERE user_id = :u AND status <> 'pending'
         ORDER BY id DESC LIMIT :n)"""), {"u": user_id, "n": HISTORY_LIMIT}).mappings().all()
    for p in rows:
        pending = p["status"] == "pending"
        item = _proposal(conn, p, user_id=user_id, protected=protected, with_excerpts=pending)
        out["pending" if pending else "history"].append(item)
    return out


def dumps(data: Any) -> Any:
    """JSON-safe copy (timestamps to ISO strings)."""
    return json.loads(json.dumps(data, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))
