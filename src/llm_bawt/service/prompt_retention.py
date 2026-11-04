"""Bounded prompt-run retention; preserve latest receipt and uncertain outcomes."""
from datetime import timedelta

from sqlalchemy import exists, func
from sqlmodel import Session, select

from .prompt_schedule_models import PromptOccurrence
from .scheduler import JobRun


def prune_prompt_runs(engine, now, limit=100):
    """Delete only old terminal history; never touch chat, sessions or deliveries."""
    latest = select(PromptOccurrence.job_id.label("job"), func.max(PromptOccurrence.created_at).label("created")) \
        .group_by(PromptOccurrence.job_id).subquery()
    with Session(engine) as session, session.begin():
        rows = session.exec(select(PromptOccurrence).where(
            PromptOccurrence.state.in_(["succeeded", "failed", "skipped", "cancelled"]),
            PromptOccurrence.updated_at < now - timedelta(days=90),
            # Undelivered main-session receipts survive retention and rotation.
            (PromptOccurrence.main_notice_json.is_(None)
             | PromptOccurrence.main_notice_turn_id.is_not(None)),
            ~exists().where(latest.c.job == PromptOccurrence.job_id,
                            latest.c.created == PromptOccurrence.created_at),
        ).order_by(PromptOccurrence.updated_at).limit(limit).with_for_update(skip_locked=True)).all()
        for occurrence in rows:
            run = session.get(JobRun, occurrence.run_id)
            session.delete(occurrence)
            session.flush()
            if run is not None:
                session.delete(run)
        return len(rows)
