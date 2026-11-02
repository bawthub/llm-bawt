"""Read-only release lifecycle projection for the live ops ledger (TASK-1947)."""
from __future__ import annotations

from sqlmodel import Session, select

from .release_models import RELEASE_TERMINAL_STATES, ReleaseRun


def attach_release_status(engine, jobs: list[dict]) -> list[dict]:
    """One bounded-to-the-ledger read, not per-banner queries or output parsing.

    The durable release executor sets release:<id> on its parent. Ordinary
    jobs need no release-table read. Submission age is not active work time;
    approval waits remain visible and explicitly waiting in the client.
    """
    ids = [job["id"] for job in jobs if str(job.get("host_unit_name") or "").startswith("release:")]
    if engine is None or not ids:
        return jobs
    with Session(engine) as session:
        rows = session.exec(select(ReleaseRun).where(ReleaseRun.parent_job_id.in_(ids))).all()
        by_parent = {row.parent_job_id: {
            "id": row.id, "state": row.state, "terminal": row.state in RELEASE_TERMINAL_STATES,
            "error_code": row.error_code, "error_text": row.error_text,
        } for row in rows}
    return [{**job, **({"release": by_parent[job["id"]]} if job["id"] in by_parent else {})}
            for job in jobs]
