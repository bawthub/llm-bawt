"""Scheduler job and run history routes."""

import json

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func
from sqlmodel import Session, select

from ...utils.config import has_database_credentials
from ..dependencies import get_profile_manager, get_service
from ..job_parameters import JobConfigurationPatch, JobParameterRegistry
from ..schemas import JobRunInfo, JobRunsResponse, ScheduledJobInfo, ScheduledJobsResponse
from ..scheduler import JobRun, JobStatus, JobType, ScheduledJob, create_scheduler_tables

router = APIRouter()


def _get_scheduler_engine():
    service = get_service()
    if not has_database_credentials(service.config):
        raise HTTPException(status_code=503, detail="Job scheduler requires database connection")

    manager = get_profile_manager(service.config)
    create_scheduler_tables(manager.engine)
    return manager.engine


@router.post("/v1/jobs/{job_type}/trigger", tags=["Jobs"])
def trigger_job(job_type: str):
    """Trigger a scheduled job to run immediately.

    Sets the job's next_run_at to the past so the scheduler picks it up
    on its next check interval.
    """
    from datetime import datetime, timedelta, timezone

    engine = _get_scheduler_engine()

    valid_types = {jt.value for jt in JobType if jt != JobType.SEND_PROMPT}
    normalized_job_type = job_type.strip().lower()
    if normalized_job_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid job_type '{job_type}'")

    job_type_enum = JobType(normalized_job_type)

    with Session(engine) as session:
        job = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == job_type_enum)
        ).first()

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"No scheduled job found for type: {job_type}"
            )

        # Set next_run_at to past to trigger immediate run
        job.next_run_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        session.add(job)
        session.commit()

    return {"success": True, "job_type": normalized_job_type}


@router.patch("/v1/jobs/{job_id}", tags=["Jobs"])
def update_job_configuration(job_id: str, patch: JobConfigurationPatch):
    """Edit one maintenance job atomically, without touching legacy resolvers."""
    from datetime import datetime, timedelta, timezone

    from pydantic import ValidationError

    with Session(_get_scheduler_engine()) as session:
        job = session.exec(select(ScheduledJob).where(
            ScheduledJob.id == job_id, ScheduledJob.job_type != JobType.SEND_PROMPT,
        ).with_for_update()).first()
        if job is None:
            raise HTTPException(status_code=404, detail="Scheduled maintenance job not found")
        try:
            if "parameters" in patch.model_fields_set:
                job.config_json = JobParameterRegistry.merge(job.job_type, job.config_json, patch.parameters)
            # Invalid parameters may be paused, but never enabled through this API.
            if patch.enabled is True and job.job_type in JobParameterRegistry.models:
                JobParameterRegistry.resolve(job.job_type, job.config_json)
        except ValidationError as exc:
            errors = exc.errors(include_context=False, include_input=False, include_url=False)
            for error in errors:
                error["loc"] = ["body", "parameters", *error["loc"]]
            raise HTTPException(status_code=422, detail=errors) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        reschedule = (
            patch.interval_minutes is not None and patch.interval_minutes != job.interval_minutes
        ) or (patch.enabled is True and not job.enabled)
        if patch.interval_minutes is not None:
            job.interval_minutes = patch.interval_minutes
        if patch.enabled is not None:
            job.enabled = patch.enabled
        if reschedule:
            job.next_run_at = datetime.now(timezone.utc) + timedelta(minutes=job.interval_minutes)
        session.add(job)
        session.commit()
    return {"success": True, "job_id": job_id}


@router.get("/v1/jobs", response_model=ScheduledJobsResponse, tags=["Jobs"])
def list_scheduled_jobs(
    job_type: str | None = Query(None, description="Filter by job type"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    enabled: bool | None = Query(None, description="Filter by enabled/disabled"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List scheduler jobs with latest run status."""
    engine = _get_scheduler_engine()

    valid_job_types = {jt.value for jt in JobType if jt != JobType.SEND_PROMPT}
    normalized_job_type = job_type.strip().lower() if job_type else None
    if normalized_job_type and normalized_job_type not in valid_job_types:
        raise HTTPException(status_code=400, detail=f"Invalid job_type '{job_type}'")

    normalized_bot_id = bot_id.strip().lower() if bot_id else None

    # Prompt schedules require the owner-scoped API, not maintenance routes.
    conditions: list = [ScheduledJob.job_type != JobType.SEND_PROMPT]
    if normalized_job_type:
        conditions.append(ScheduledJob.job_type == normalized_job_type)
    if normalized_bot_id:
        conditions.append(ScheduledJob.bot_id == normalized_bot_id)
    if enabled is not None:
        conditions.append(ScheduledJob.enabled.is_(enabled))

    statement = select(ScheduledJob)
    count_statement = select(func.count()).select_from(ScheduledJob)
    if conditions:
        statement = statement.where(*conditions)
        count_statement = count_statement.where(*conditions)
    statement = statement.order_by(ScheduledJob.job_type, ScheduledJob.bot_id)

    with Session(engine) as session:
        total_count = int(session.exec(count_statement).one() or 0)
        jobs = session.exec(statement.offset(offset).limit(limit)).all()

        items: list[ScheduledJobInfo] = []
        for job in jobs:
            latest_run = session.exec(
                select(JobRun)
                .where(JobRun.job_id == job.id)
                .order_by(JobRun.started_at.desc())
                .limit(1)
            ).first()

            items.append(
                ScheduledJobInfo(
                    id=job.id,
                    job_type=job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
                    bot_id=job.bot_id,
                    enabled=job.enabled,
                    **JobParameterRegistry.describe(job.job_type, job.config_json),
                    interval_minutes=job.interval_minutes,
                    last_run_at=job.last_run_at,
                    next_run_at=job.next_run_at,
                    created_at=job.created_at,
                    last_status=(
                        latest_run.status.value if latest_run and hasattr(latest_run.status, "value")
                        else (str(latest_run.status) if latest_run else None)
                    ),
                    last_duration_ms=latest_run.duration_ms if latest_run else None,
                    last_error=latest_run.error_message if latest_run else None,
                )
            )

    return ScheduledJobsResponse(
        jobs=items,
        total_count=total_count,
        filters={
            "job_type": normalized_job_type,
            "bot_id": normalized_bot_id,
            "enabled": enabled,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get("/v1/jobs/runs", response_model=JobRunsResponse, tags=["Jobs"])
def list_job_runs(
    job_id: str | None = Query(None, description="Filter by scheduled job ID"),
    job_type: str | None = Query(None, description="Filter by job type"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    status: str | None = Query(None, description="Filter by run status"),
    include_result: bool = Query(False, description="Include parsed result payload"),
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List scheduler run history."""
    engine = _get_scheduler_engine()

    valid_job_types = {jt.value for jt in JobType if jt != JobType.SEND_PROMPT}
    valid_statuses = {js.value for js in JobStatus}

    normalized_job_type = job_type.strip().lower() if job_type else None
    if normalized_job_type and normalized_job_type not in valid_job_types:
        raise HTTPException(status_code=400, detail=f"Invalid job_type '{job_type}'")

    normalized_status = status.strip().lower() if status else None
    if normalized_status and normalized_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status '{status}'")

    normalized_bot_id = bot_id.strip().lower() if bot_id else None
    normalized_job_id = job_id.strip() if job_id else None

    with Session(engine) as session:
        # Build job-type lookup map (and optional job type filter set).
        job_lookup: dict[str, str] = {}
        for row in session.exec(select(ScheduledJob).where(ScheduledJob.job_type != JobType.SEND_PROMPT)).all():
            jtype = row.job_type.value if hasattr(row.job_type, "value") else str(row.job_type)
            job_lookup[row.id] = jtype

        filtered_job_ids: set[str] | None = None
        if normalized_job_type:
            filtered_job_ids = {jid for jid, jtype in job_lookup.items() if jtype == normalized_job_type}
            if not filtered_job_ids:
                return JobRunsResponse(runs=[], total_count=0, filters={
                    "job_id": normalized_job_id,
                    "job_type": normalized_job_type,
                    "bot_id": normalized_bot_id,
                    "status": normalized_status,
                    "include_result": include_result,
                    "limit": limit,
                    "offset": offset,
                })

        conditions: list = [JobRun.job_id.in_(
            select(ScheduledJob.id).where(ScheduledJob.job_type != JobType.SEND_PROMPT)
        )]
        if normalized_job_id:
            conditions.append(JobRun.job_id == normalized_job_id)
        if normalized_bot_id:
            conditions.append(JobRun.bot_id == normalized_bot_id)
        if normalized_status:
            conditions.append(JobRun.status == normalized_status)
        if filtered_job_ids is not None:
            conditions.append(JobRun.job_id.in_(filtered_job_ids))

        statement = select(JobRun)
        count_statement = select(func.count()).select_from(JobRun)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)
        statement = statement.order_by(JobRun.started_at.desc())

        total_count = int(session.exec(count_statement).one() or 0)
        runs = session.exec(statement.offset(offset).limit(limit)).all()

        items = []
        for run in runs:
            parsed_result = None
            if include_result and run.result_json:
                try:
                    parsed_result = json.loads(run.result_json)
                except Exception:
                    parsed_result = run.result_json

            status_value = run.status.value if hasattr(run.status, "value") else str(run.status)
            items.append(
                JobRunInfo(
                    id=run.id,
                    job_id=run.job_id,
                    job_type=job_lookup.get(run.job_id),
                    bot_id=run.bot_id,
                    status=status_value,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    duration_ms=run.duration_ms,
                    error_message=run.error_message,
                    result=parsed_result,
                )
            )

    return JobRunsResponse(
        runs=items,
        total_count=total_count,
        filters={
            "job_id": normalized_job_id,
            "job_type": normalized_job_type,
            "bot_id": normalized_bot_id,
            "status": normalized_status,
            "include_result": include_result,
            "limit": limit,
            "offset": offset,
        },
    )
