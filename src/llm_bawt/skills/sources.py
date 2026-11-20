"""Skill sources: configured Git repositories updated independently of images (TASK-2534).

A row in ``skill_sources`` names a Git repository (branch or tag) or a
published archive manifest, an optional subdirectory, and records the pinned ``active_revision`` (and the previous one
for rollback). The files live in :class:`.source_store.SourceStore`; this module
keeps the rows and the store in step and is the only writer of either.

Rules:

- Adding a source fetches and validates it first; a source that cannot be read
  is never saved.
- Skill names are unique across configured sources. An update that would
  introduce a name another source already provides is refused (409). The
  image's bundled skills are only a fallback and never collide.
- Check finds the remote revision and stages it, so the owner sees exactly
  which skills are added, changed or removed before activating.
- Update and rollback rebuild the effective tree atomically. A failed fetch or
  validation leaves the active revision, other sources and assignments as they
  were. Bots pick up the change at their next session boundary, through the
  existing skill/system-prompt pin.
- The built-in source (``builtin``) is a published archive configured by the
  deployment (``SKILL_SOURCES_DEFAULT_URL``); it needs no credentials and can
  be checked, updated and rolled back like any other, but not removed.
- Assignments reference bare skill names, so they survive updates and moves
  between revisions.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from ..utils.schema import SchemaBootstrapGuard, ensure_relations
from .source_store import (
    REVISION,
    SOURCE_ID,
    GitSpec,
    Layer,
    SourceError,
    SourceStore,
    Spec,
    archive_spec,
    diff_skills,
    git_spec,
    layer_map,
)

CREDENTIALS = ("none", "github")

_RELATIONS = {
    "skill_sources": """
        CREATE TABLE IF NOT EXISTS skill_sources (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('git', 'archive')),
            url TEXT NOT NULL,
            ref TEXT,
            subdir TEXT NOT NULL DEFAULT '',
            credential TEXT NOT NULL DEFAULT 'none' CHECK (credential IN ('none', 'github')),
            priority INTEGER NOT NULL DEFAULT 100,
            builtin BOOLEAN NOT NULL DEFAULT false,
            active_revision TEXT,
            active_skills JSONB NOT NULL DEFAULT '{}'::jsonb,
            previous_revision TEXT,
            previous_skills JSONB,
            latest_revision TEXT,
            last_checked_at TIMESTAMPTZ,
            last_error TEXT,
            updated_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
}
_LOCK = "llm_bawt.skill_sources"
_schema_guard = SchemaBootstrapGuard()
_COLUMNS = ("id", "label", "kind", "url", "ref", "subdir", "credential", "priority", "builtin",
            "active_revision", "active_skills", "previous_revision", "previous_skills",
            "latest_revision", "last_checked_at", "last_error", "updated_by", "created_at", "updated_at")


def _bootstrap(conn) -> None:
    ensure_relations(conn, _RELATIONS)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SourceRow:
    id: str
    label: str
    kind: str
    url: str
    ref: str | None
    subdir: str
    credential: str
    priority: int
    builtin: bool
    active_revision: str | None
    active_skills: dict[str, str]
    previous_revision: str | None
    previous_skills: dict[str, str] | None
    latest_revision: str | None
    last_checked_at: datetime | None
    last_error: str | None
    updated_by: str | None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def public(self, owners: dict[str, str], effective: dict[str, str] | None = None) -> dict[str, Any]:
        provided = sorted(self.active_skills)
        live = (effective or {}).get(self.id)
        return {
            # What bots read now; differs from active_revision only while a saved
            # change waits to be applied (startup or the next source write applies it).
            "effective_revision": live,
            "pending": live != self.active_revision,
            "id": self.id, "label": self.label, "kind": self.kind, "url": self.url,
            "ref": self.ref, "subdir": self.subdir, "credential": self.credential,
            "priority": self.priority, "builtin": self.builtin,
            "active_revision": self.active_revision,
            "previous_revision": self.previous_revision,
            "latest_revision": self.latest_revision,
            "update_available": bool(self.latest_revision and self.latest_revision != self.active_revision),
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "last_error": self.last_error,
            "skills": provided,
            # A skill this source ships that another source (or the bundle) shadows.
            "shadowed": sorted(name for name in provided if owners.get(name, self.id) != self.id),
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _row(mapping) -> SourceRow:
    data = dict(mapping)
    for key in ("active_skills", "previous_skills"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    data["active_skills"] = data.get("active_skills") or {}
    return SourceRow(**{key: data.get(key) for key in _COLUMNS})


def slug_for(label: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:48] or "source"
    if not SOURCE_ID.fullmatch(base):
        base = "source"
    candidate, n = base, 2
    while candidate in taken or candidate == "bundled":
        candidate, n = f"{base}-{n}", n + 1
    return candidate


TokenFor = Callable[[GitSpec], str | None]


def make_spec(kind: str, url: str, ref: str | None, subdir: str | None,
              keys: tuple[str, ...] = ()) -> Spec:
    if kind == "archive":
        return archive_spec(url, subdir, keys)
    if kind == "git":
        return git_spec(url, ref, subdir)
    raise SourceError(400, "invalid_kind", "Source kind must be git or archive.")


class SkillSourceService:
    def __init__(self, engine, store: SourceStore, token_for: TokenFor,
                 default_keys: tuple[str, ...] = ()):
        self.engine = engine
        self.store = store
        self._token_for = token_for
        # Publisher keys the built-in archive's manifest must be signed with.
        self.default_keys = tuple(default_keys)

    def _spec(self, row: SourceRow) -> Spec:
        keys = self.default_keys if row.builtin else ()
        if row.builtin and not keys:
            raise SourceError(503, "untrusted_manifest", "No trusted publisher key is configured for the built-in source.")
        return make_spec(row.kind, row.url, row.ref, row.subdir, keys)

    # -- reads ---------------------------------------------------------------

    def ensure_schema(self) -> None:
        _schema_guard.run(self.engine, "skill-sources", _bootstrap)

    def _rows(self, conn) -> list[SourceRow]:
        result = conn.execute(text(
            f"SELECT {', '.join(_COLUMNS)} FROM skill_sources ORDER BY priority, created_at, id"))
        return [_row(r) for r in result.mappings()]

    def _get(self, conn, source_id: str, *, lock: bool = False) -> SourceRow:
        suffix = " FOR UPDATE" if lock else ""
        found = conn.execute(text(
            f"SELECT {', '.join(_COLUMNS)} FROM skill_sources WHERE id = :id{suffix}"),
            {"id": source_id}).mappings().first()
        if found is None:
            raise SourceError(404, "not_found", "No such skill source.")
        return _row(found)

    def list(self) -> dict[str, Any]:
        self._require_store()
        self.ensure_schema()
        with self.engine.connect() as conn:
            rows = self._rows(conn)
        owners = self.store.owners()
        effective = self.store.layers()
        return {
            "sources": [row.public(owners, effective) for row in rows],
            # Removed sources whose skills bots still read until the change applies.
            "pending_removals": [
                {"id": sid, "skills": sorted(n for n, o in owners.items() if o == sid)}
                for sid in sorted(effective or {}) if sid not in {r.id for r in rows}],
            "bundled": sorted(name for name, owner in owners.items() if owner == "bundled"),
        }

    # -- writes ----------------------------------------------------------------

    def _require_store(self) -> None:
        if not self.store.available():
            raise SourceError(503, "store_unavailable",
                              "The skill source store is not configured or not writable.")

    def _token(self, row_or_spec: SourceRow | Spec, credential: str) -> str | None:
        if credential == "none":
            return None
        spec = self._spec(row_or_spec) if isinstance(row_or_spec, SourceRow) else row_or_spec
        if not isinstance(spec, GitSpec) or not spec.github:
            raise SourceError(400, "credential_unsupported",
                              "GitHub credentials can only be used with github.com repositories.")
        token = self._token_for(spec)
        if not token:
            raise SourceError(503, "credential_unavailable",
                              "GitHub is not connected; connect it in provider settings.")
        return token

    @staticmethod
    def _collisions(rows: list[SourceRow], source_id: str, skills: dict[str, str]) -> list[str]:
        others = {name for row in rows if row.id != source_id for name in row.active_skills}
        return sorted(others & set(skills))

    def _layers(self, rows: list[SourceRow]) -> list[Layer]:
        return [
            Layer(row.id, self.store.revision_dir(row.id, row.active_revision))
            for row in sorted(rows, key=lambda r: (r.priority, r.id))
            if row.active_revision
        ]

    def _publish(self, rows: list[SourceRow]) -> dict[str, str]:
        """Rebuild ``current`` from ``rows``. Call under the store lock."""
        return self.store.build(self._layers(rows))

    def _transition(self, mutate: Callable[[Any], Any]) -> Any:
        """Apply ``mutate``, commit, then make it live. Call under the store lock.

        The new tree is prepared inside the transaction from the uncommitted
        rows but not made visible, so readers never see uncommitted skills and
        a failed build rolls the change back. Only after the commit does the
        prepared tree replace ``current``. If that swap fails, the change is
        saved but reported as pending (``effective_revision`` still shows what
        bots read) and the next reconcile applies it.
        """
        gen = None
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": _LOCK})
                result = mutate(conn)
                gen = self.store.prepare(self._layers(self._rows(conn)))
        except BaseException:
            if gen is not None:
                self.store.discard(gen)
            raise
        try:
            self.store.activate(gen)
        except OSError as exc:
            self.store.discard(gen)
            raise SourceError(503, "apply_pending",
                              "The change is saved but not applied yet; it applies on restart or the next source change.") from exc
        return result

    def _reconcile_locked(self) -> None:
        with self.engine.connect() as conn:
            rows = self._rows(conn)
        layers = self._layers(rows)
        if not self.store.current.exists() or self.store.layers() != layer_map(layers):
            self.store.build(layers)

    def ensure_current(self) -> None:
        """Make the effective tree match the committed sources (fresh volume, interrupted write)."""
        if not self.store.available():
            return
        self.ensure_schema()
        with self.store.locked():
            self._reconcile_locked()

    def rebuild(self) -> dict[str, str]:
        self._require_store()
        self.ensure_schema()
        with self.store.locked(), self.engine.connect() as conn:
            return self._publish(self._rows(conn))

    def add(self, *, label: str, url: str, ref: str | None, subdir: str | None,
            credential: str = "none", priority: int = 100, updated_by: str | None = None,
            kind: str = "git", builtin: bool = False) -> dict[str, Any]:
        self._require_store()
        self.ensure_schema()
        if credential not in CREDENTIALS:
            raise SourceError(400, "invalid_credential", "Unknown credential.")
        label = (label or "").strip()
        if not label or len(label) > 80:
            raise SourceError(400, "invalid_label", "Give the source a name (up to 80 characters).")
        if builtin and not self.default_keys:
            raise SourceError(503, "untrusted_manifest", "No trusted publisher key is configured for the built-in source.")
        spec = make_spec(kind, url, ref, subdir, self.default_keys if builtin else ())
        ref = getattr(spec, "ref", None)
        token = self._token(spec, credential)
        with self.store.locked():
            self._reconcile_locked()
            with self.engine.connect() as conn:
                rows = self._rows(conn)
            if any(r.url == spec.url and r.ref == ref and r.subdir == spec.subdir for r in rows):
                raise SourceError(409, "duplicate_source", "That repository, branch and folder is already a source.")
            source_id = slug_for(label, {r.id for r in rows})
            # Leftovers of a removed source with this slug are never reused.
            self.store.remove_source(source_id)
            try:
                fetched = self.store.stage(source_id, spec, token)

                def insert(conn) -> None:
                    clash = self._collisions(self._rows(conn), source_id, fetched.skills)
                    if clash:
                        raise SourceError(409, "name_collision",
                                          f"Another source already provides: {', '.join(clash)}.")
                    conn.execute(text("""
                        INSERT INTO skill_sources (id, label, kind, url, ref, subdir, credential, priority,
                            builtin, active_revision, active_skills, latest_revision, last_checked_at, updated_by)
                        VALUES (:id, :label, :kind, :url, :ref, :subdir, :credential, :priority,
                            :builtin, :rev, CAST(:skills AS jsonb), :rev, :now, :by)"""),
                        {"id": source_id, "label": label, "kind": kind, "builtin": builtin,
                         "url": spec.url, "ref": ref, "subdir": spec.subdir, "credential": credential,
                         "priority": priority, "rev": fetched.revision,
                         "skills": json.dumps(fetched.skills), "now": _now(), "by": updated_by})

                self._transition(insert)
            except BaseException:
                # A lost commit acknowledgment does not prove rollback: keep the
                # files unless the row is confirmed absent.
                try:
                    with self.engine.connect() as conn:
                        committed = any(r.id == source_id for r in self._rows(conn))
                except Exception:
                    committed = True
                if not committed:
                    self.store.remove_source(source_id)
                raise
        return {"source": self._public(source_id), "changes": diff_skills(None, fetched.skills)}

    def ensure_default(self, url: str, *, subdir: str | None = None,
                       label: str = "BawtHub") -> dict[str, Any] | None:
        """Install the deployment's built-in archive source once.

        Owners then update it from the UI like any other source. A changed
        configured URL repoints the row; its pinned revision stays active
        until the next update.
        """
        if not url or not self.store.available():
            return None
        self.ensure_schema()
        if not self.default_keys:
            raise SourceError(503, "untrusted_manifest", "No trusted publisher key is configured for the built-in source.")
        spec = archive_spec(url, subdir, self.default_keys)
        with self.engine.begin() as conn:
            existing = next((r for r in self._rows(conn) if r.builtin), None)
            if existing:
                if (existing.kind, existing.url, existing.subdir) != ("archive", spec.url, spec.subdir):
                    conn.execute(text(
                        "UPDATE skill_sources SET kind = 'archive', url = :url, subdir = :subdir, "
                        "latest_revision = NULL, updated_at = :now WHERE id = :id"),
                        {"url": spec.url, "subdir": spec.subdir, "now": _now(), "id": existing.id})
                return None
        return self.add(label=label, url=spec.url, ref=None, subdir=spec.subdir, kind="archive",
                        builtin=True, priority=1000, updated_by="system")

    def _public(self, source_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            return self._get(conn, source_id).public(self.store.owners(), self.store.layers())

    def _record_error(self, source_id: str, message: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(
                "UPDATE skill_sources SET last_error = :e, last_checked_at = :now WHERE id = :id"),
                {"e": message, "now": _now(), "id": source_id})

    def check(self, source_id: str) -> dict[str, Any]:
        """Find the remote revision, stage it, and report what would change."""
        self._require_store()
        self.ensure_schema()
        with self.store.locked():
            with self.engine.connect() as conn:
                row = self._get(conn, source_id)
            try:
                token = self._token(row, row.credential)
                spec = self._spec(row)
                latest = self.store.fetcher_for(spec).remote_revision(spec, token)
                if latest == row.active_revision:
                    skills = row.active_skills
                else:
                    fetched = self.store.stage(row.id, spec, token)
                    latest, skills = fetched.revision, fetched.skills
            except SourceError as exc:
                self._record_error(source_id, exc.detail)
                raise
            with self.engine.begin() as conn:
                rows = self._rows(conn)
                conn.execute(text("""
                    UPDATE skill_sources SET latest_revision = :rev, last_checked_at = :now, last_error = NULL
                    WHERE id = :id"""), {"rev": latest, "now": _now(), "id": source_id})
        return {
            "source": self._public(source_id),
            "active_revision": row.active_revision,
            "latest_revision": latest,
            "changes": diff_skills(row.active_skills, skills),
            "collisions": self._collisions(rows, source_id, skills),
        }

    def update(self, source_id: str, *, revision: str, updated_by: str | None = None) -> dict[str, Any]:
        """Activate exactly ``revision``, which a check must already have staged."""
        self._require_store()
        self.ensure_schema()
        if not revision or not REVISION.fullmatch(revision):
            raise SourceError(400, "invalid_revision", "Check for updates and confirm a revision.")
        try:
            with self.store.locked():
                self._reconcile_locked()
                with self.engine.connect() as conn:
                    row = self._get(conn, source_id)
                if not self.store.revision_dir(row.id, revision).is_dir():
                    raise SourceError(409, "revision_not_staged",
                                      "That revision is no longer staged; check for updates again.")
                skills = self.store.skills_at(row.id, revision)
                return self._activate(source_id, revision, skills, updated_by)
        except SourceError as exc:
            if exc.code not in {"name_collision", "not_found", "revision_not_staged"}:
                self._record_error(source_id, exc.detail)
            raise

    def _activate(self, source_id: str, revision: str, skills: dict[str, str],
                  updated_by: str | None) -> dict[str, Any]:
        """Pin and publish together, then prune. Call under the store lock."""

        def pin(conn) -> dict[str, Any]:
            row = self._get(conn, source_id, lock=True)
            clash = self._collisions(self._rows(conn), source_id, skills)
            if clash:
                raise SourceError(409, "name_collision",
                                  f"Another source already provides: {', '.join(clash)}.")
            changes = diff_skills(row.active_skills, skills)
            if revision != row.active_revision:
                conn.execute(text("""
                    UPDATE skill_sources SET previous_revision = active_revision,
                        previous_skills = active_skills, active_revision = :rev,
                        active_skills = CAST(:skills AS jsonb), last_error = NULL,
                        updated_by = :by, updated_at = :now
                    WHERE id = :id"""),
                    {"rev": revision, "skills": json.dumps(skills), "by": updated_by,
                     "now": _now(), "id": source_id})
            return changes

        changes = self._transition(pin)
        with self.engine.connect() as conn:
            rows = self._rows(conn)
        current = next(r for r in rows if r.id == source_id)
        keep = {r for r in (current.active_revision, current.previous_revision, current.latest_revision) if r}
        self.store.prune_revisions(source_id, keep)  # only once the transition is durable
        return {"source": self._public(source_id), "changes": changes}

    def rollback(self, source_id: str, *, updated_by: str | None = None) -> dict[str, Any]:
        self._require_store()
        self.ensure_schema()
        with self.store.locked():
            self._reconcile_locked()
            with self.engine.connect() as conn:
                row = self._get(conn, source_id)
            if not row.previous_revision:
                raise SourceError(409, "no_previous", "This source has no previous revision to roll back to.")
            skills = self.store.skills_at(row.id, row.previous_revision)
            return self._activate(source_id, row.previous_revision, skills, updated_by)

    def remove(self, source_id: str) -> dict[str, Any]:
        self._require_store()
        self.ensure_schema()
        with self.store.locked():
            def delete(conn) -> SourceRow:
                row = self._get(conn, source_id, lock=True)
                if row.builtin:
                    raise SourceError(409, "builtin_source", "The built-in source can be updated but not removed.")
                conn.execute(text("DELETE FROM skill_sources WHERE id = :id"), {"id": source_id})
                return row

            row = self._transition(delete)
            self.store.remove_source(source_id)
        return {"removed": source_id, "skills": sorted(row.active_skills)}
