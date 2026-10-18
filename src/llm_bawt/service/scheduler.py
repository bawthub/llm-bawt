"""Background job scheduler with database persistence."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, Session, select
from sqlalchemy import Column, DateTime, Text, text
from ..bot_identities import SYSTEM_BOT_ID, WILDCARD_BOT_ID
from ..runtime_settings import RuntimeSettingsResolver

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobType(str, Enum):
    """Types of scheduled jobs."""
    PROFILE_MAINTENANCE = "profile_maintenance"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEMORY_DECAY = "memory_decay"
    HISTORY_SUMMARIZATION = "history_summarization"
    MEMORY_EXTRACTION = "memory_extraction"
    MEDIA_GC = "media_gc"
    TOOL_RESULT_GC = "tool_result_gc"
    SEND_PROMPT = "send_prompt"


class ScheduledJob(SQLModel, table=True):
    """Definition of a recurring job."""
    __tablename__ = "scheduled_jobs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_type: JobType = Field(index=True)
    bot_id: str = Field(index=True, description="Bot this job runs for, or '*' for all")
    enabled: bool = Field(default=True)
    interval_minutes: int = Field(default=60)
    last_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    next_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime(timezone=True)))
    config_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Job-specific config as JSON")


class JobRun(SQLModel, table=True):
    """History of job executions."""
    __tablename__ = "job_runs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_id: str = Field(index=True, foreign_key="scheduled_jobs.id")
    bot_id: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime(timezone=True)))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    duration_ms: Optional[int] = Field(default=None)
    result_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Success result as JSON")
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))


def create_scheduler_tables(engine) -> None:
    """Create scheduler tables if they don't exist."""
    SQLModel.metadata.create_all(engine, tables=[ScheduledJob.__table__, JobRun.__table__])
    _migrate_postgres_jobtype_enum(engine)
    from .prompt_schedule_models import PromptOccurrence, PromptSchedule
    SQLModel.metadata.create_all(engine, tables=[PromptSchedule.__table__, PromptOccurrence.__table__])


def _migrate_postgres_jobtype_enum(engine) -> None:
    """Ensure Postgres jobtype enum includes all currently supported labels."""
    if engine.dialect.name != "postgresql":
        return

    with engine.begin() as conn:
        type_exists = conn.execute(
            text("SELECT 1 FROM pg_type WHERE typname = 'jobtype'")
        ).scalar()
        if not type_exists:
            return

        labels = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT e.enumlabel
                    FROM pg_enum e
                    JOIN pg_type t ON t.oid = e.enumtypid
                    WHERE t.typname = 'jobtype'
                    """
                )
            )
        }

        for member in JobType:
            if member.name not in labels:
                safe_label = member.name.replace("'", "''")
                conn.execute(text(f"ALTER TYPE jobtype ADD VALUE IF NOT EXISTS '{safe_label}'"))
                logger.info("Added missing enum value to jobtype: %s", member.name)


class JobScheduler:
    """
    Async scheduler that watches for due jobs and enqueues them.
    
    Usage:
        scheduler = JobScheduler(engine, task_processor)
        await scheduler.start()  # Runs until stopped
        await scheduler.stop()
    """
    
    def __init__(self, engine, task_processor, check_interval: int = 30, *, prompt_sweep=None):
        """
        Args:
            engine: SQLAlchemy engine for job tables
            task_processor: TaskProcessor instance to enqueue tasks
            check_interval: Seconds between checking for due jobs
        """
        self.engine = engine
        self.task_processor = task_processor
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # Wired only once the durable delivery consumer is ready (TASK-168).
        # Do not consume billable occurrences into an unconsumed outbox on upgrade.
        self._prompt_sweep = prompt_sweep
        self._prompt_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        if self._prompt_sweep is not None:
            self._prompt_task = asyncio.create_task(self._run_prompt_loop(), name="prompt-scheduler")
        logger.debug(f"Scheduler started (check interval: {self.check_interval}s)")
    
    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._running = False
        tasks = [task for task in (self._task, self._prompt_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("Scheduler stopped")

    async def _run_prompt_loop(self) -> None:
        """Independent cadence: maintenance/model processing cannot delay prompts."""
        while self._running:
            try:
                await self._prompt_sweep()
            except Exception:
                logger.exception("Prompt scheduler sweep failed; will retry")
            await asyncio.sleep(5)
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_due_jobs()
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_and_run_due_jobs(self) -> None:
        """Check for due jobs and enqueue them."""
        loop = asyncio.get_event_loop()
        
        def get_due_jobs():
            with Session(self.engine) as session:
                now = datetime.now(timezone.utc)
                statement = select(ScheduledJob).where(
                    ScheduledJob.enabled.is_(True),
                    ScheduledJob.job_type != JobType.SEND_PROMPT,
                    (ScheduledJob.next_run_at.is_(None)) | (ScheduledJob.next_run_at <= now)
                )
                return session.exec(statement).all()
        
        due_jobs = await loop.run_in_executor(None, get_due_jobs)
        
        for job in due_jobs:
            await self._execute_job(job)
    
    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a single job.

        When ``job.bot_id`` is ``"*"``, the job is expanded into one run per
        bot so that newly-created bots are automatically included without
        needing a restart or per-bot job rows.
        """
        # Expand wildcard jobs into per-bot runs
        if job.bot_id == WILDCARD_BOT_ID and job.job_type in (
            JobType.HISTORY_SUMMARIZATION,
            JobType.MEMORY_EXTRACTION,
            JobType.MEMORY_CONSOLIDATION,
            JobType.MEMORY_DECAY,
        ):
            from llm_bawt.bots import BotManager
            bot_manager = BotManager(self.task_processor.config)
            all_bots = [bot.slug for bot in bot_manager.list_bots()]
            if not all_bots:
                logger.debug("No bots found for wildcard job %s", job.job_type)
                return

            logger.debug(
                "Expanding wildcard job %s to %d bots: %s",
                job.job_type, len(all_bots), all_bots,
            )
            for bot_id in all_bots:
                # Create a lightweight copy with the specific bot_id
                per_bot_job = ScheduledJob(
                    id=job.id,
                    job_type=job.job_type,
                    bot_id=bot_id,
                    enabled=job.enabled,
                    interval_minutes=job.interval_minutes,
                    last_run_at=job.last_run_at,
                    next_run_at=job.next_run_at,
                    config_json=job.config_json,
                )
                await self._execute_single_job(per_bot_job)

            # Update the real job's timestamps
            await self._update_job_timestamps(job)
            return

        await self._execute_single_job(job)

    async def _update_job_timestamps(self, job: ScheduledJob) -> None:
        """Update last_run_at and next_run_at for a job after execution."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)

        def _update():
            with Session(self.engine) as session:
                db_job = session.get(ScheduledJob, job.id)
                if db_job:
                    db_job.last_run_at = now
                    db_job.next_run_at = now + timedelta(minutes=db_job.interval_minutes)
                    session.add(db_job)
                    session.commit()

        await loop.run_in_executor(None, _update)

    async def _execute_single_job(self, job: ScheduledJob) -> None:
        """Execute a single job for a specific bot."""
        logger.debug(f"Running scheduled job: {job.job_type} for bot {job.bot_id}")

        loop = asyncio.get_event_loop()

        # Create job run record
        def create_run():
            with Session(self.engine) as session:
                run = JobRun(
                    job_id=job.id,
                    bot_id=job.bot_id,
                    status=JobStatus.RUNNING,
                )
                session.add(run)
                session.commit()
                session.refresh(run)
                return run.id

        run_id = await loop.run_in_executor(None, create_run)
        start_time = datetime.now(timezone.utc)
        status = JobStatus.RUNNING
        result_json: str | None = None
        error_message: str | None = None

        try:
            should_run, skip_reason = await loop.run_in_executor(
                None,
                lambda: self._has_new_activity_for_job(job),
            )
            if not should_run:
                status = JobStatus.SKIPPED
                result_json = json.dumps(
                    {
                        "skipped": True,
                        "reason": skip_reason or "No new activity since last run",
                    }
                )
                logger.debug(
                    "Skipping scheduled job %s for bot %s: %s",
                    job.job_type,
                    job.bot_id,
                    skip_reason or "no new activity",
                )
            else:
                # Create and execute task based on job type
                task = self._create_task_for_job(job)
                if task:
                    task_result = await self.task_processor.process_task(task)
                    # TaskResult has .result (dict) and .status
                    from llm_bawt.service.tasks import TaskStatus as TStatus
                    if task_result.status == TStatus.COMPLETED:
                        result_data = task_result.result or {}
                        status = JobStatus.SUCCESS if result_data.get("error") is None else JobStatus.FAILED
                        result_json = json.dumps(result_data)
                        error_message = result_data.get("error")
                    else:
                        status = JobStatus.FAILED
                        result_json = None
                        error_message = task_result.error
                else:
                    status = JobStatus.SKIPPED
                    result_json = None
                    error_message = f"Unknown job type: {job.job_type}"
        except Exception as e:
            status = JobStatus.FAILED
            result_json = None
            error_message = str(e)
            logger.exception(f"Job {job.id} failed")

        # Update job run record
        finish_time = datetime.now(timezone.utc)
        duration_ms = int((finish_time - start_time).total_seconds() * 1000)
        
        def update_run():
            with Session(self.engine) as session:
                run = session.get(JobRun, run_id)
                if run:
                    run.status = status
                    run.finished_at = finish_time
                    run.duration_ms = duration_ms
                    run.result_json = result_json
                    run.error_message = error_message
                    session.add(run)
                
                # Update job's next_run_at
                db_job = session.get(ScheduledJob, job.id)
                if db_job:
                    db_job.last_run_at = start_time
                    db_job.next_run_at = start_time + timedelta(minutes=db_job.interval_minutes)
                    session.add(db_job)
                
                session.commit()
        
        await loop.run_in_executor(None, update_run)
        logger.debug(f"Job {job.job_type} completed with status {status}")

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """Sanitize table identifier to lowercase alnum/underscore."""
        return re.sub(r"[^a-z0-9_]", "", (identifier or "").lower())

    @staticmethod
    def _parse_config_json(config_json: str | None) -> dict:
        """Parse job config JSON safely."""
        if not config_json:
            return {}
        try:
            return json.loads(config_json)
        except Exception:
            return {}

    def _max_timestamp_message_activity(self, bot_id: str) -> float | None:
        """Return latest non-summary message timestamp for a bot.

        Queries the partitioned ``messages`` parent (TASK-571); the
        ``bot_id`` equality prunes the plan to that bot's partition.
        """
        table_bot = self._sanitize_identifier(bot_id)
        if not table_bot:
            return None
        sql = text(
            """
            SELECT MAX(timestamp) AS latest_ts
            FROM messages
            WHERE bot_id = :bot_id
              AND role != 'summary'
              AND COALESCE(recalled_history, FALSE) = FALSE
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"bot_id": table_bot}).fetchone()
            if not row:
                return None
            latest_ts = row[0]
            if latest_ts is None:
                return None
            return float(latest_ts)

    def _max_memory_updated_at(self, bot_id: str) -> datetime | None:
        """Return latest memory update time for a bot."""
        table_bot = self._sanitize_identifier(bot_id)
        if not table_bot:
            return None
        sql = text(
            "SELECT MAX(updated_at) AS latest_updated_at FROM memories "
            "WHERE bot_id = :bot_id"
        )
        with self.engine.connect() as conn:
            row = conn.execute(sql, {"bot_id": table_bot}).fetchone()
            if not row:
                return None
            latest = row[0]
            if isinstance(latest, datetime):
                return latest
            return None

    def _count_unprocessed_messages(self, bot_id: str) -> int:
        """Return count of unprocessed messages for a bot."""
        table_bot = self._sanitize_identifier(bot_id)
        if not table_bot:
            return 0
        sql = text(
            """
            SELECT COUNT(*) AS cnt
            FROM messages
            WHERE bot_id = :bot_id
              AND COALESCE(processed, FALSE) = FALSE
              AND role IN ('user', 'assistant')
            """
        )
        try:
            with self.engine.connect() as conn:
                row = conn.execute(sql, {"bot_id": table_bot}).fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def _profile_needs_rebuild(self, entity_id: str, entity_type: str) -> tuple[bool, str | None]:
        """Check if profile attributes were modified after the last summary build."""
        safe_entity_id = (entity_id or "").strip().lower()
        safe_entity_type = (entity_type or "user").strip().lower()
        if not safe_entity_id:
            return False, "No entity_id"

        with self.engine.connect() as conn:
            attr_row = conn.execute(
                text(
                    """
                    SELECT MAX(updated_at) AS latest_updated_at
                    FROM entity_profile_attributes
                    WHERE entity_id = :entity_id
                      AND entity_type = :entity_type
                    """
                ),
                {"entity_id": safe_entity_id, "entity_type": safe_entity_type},
            ).fetchone()
            summary_row = conn.execute(
                text(
                    """
                    SELECT summary_updated_at
                    FROM entity_profiles
                    WHERE entity_id = :entity_id
                      AND entity_type = :entity_type
                    """
                ),
                {"entity_id": safe_entity_id, "entity_type": safe_entity_type},
            ).fetchone()

            latest_attr = attr_row[0] if attr_row and isinstance(attr_row[0], datetime) else None
            summary_at = summary_row[0] if summary_row and isinstance(summary_row[0], datetime) else None

            if latest_attr is None:
                return False, "No profile attributes found"
            if summary_at is None:
                return True, "Profile summary has never been built"
            if latest_attr > summary_at:
                return True, None
            return False, "No profile attribute changes since last summary build"

    def _has_pending_history_summaries(
        self,
        bot_id: str,
        job_config: dict | None = None,
    ) -> tuple[bool, int]:
        """Return whether there are currently eligible unsummarized sessions.

        This guards against backlog starvation where no *new* activity happened
        since the previous check, but old sessions are still eligible and
        unsummarized.
        """
        try:
            from ..bots import BotManager
            from ..memory.summarization import HistorySummarizer

            config = self.task_processor.config
            job_config = job_config or {}

            # Match _process_history_summarization exactly.  The old probe used
            # global settings and the default model while execution used the
            # bot's runtime settings and resolved background model.  That made
            # the probe say "pending" and execution immediately return zero.
            from ..runtime_settings import resolve_job_model

            # TASK-610: Tier-1 job params from the ONE global dict (model +
            # trigger below). model=None => inherit maintenance_model, as before.
            sum_job = config.resolve_summarization_job()
            requested_model = (
                job_config.get("model")
                or sum_job.get("model")
                or resolve_job_model(config, "maintenance_model")
            )
            preferred_model = (
                requested_model
                or resolve_job_model(config, "extraction_model")
                or resolve_job_model(config, "maintenance_model")
            )
            model_alias = None
            resolve_model = getattr(self.task_processor, "_resolve_request_model", None)
            if callable(resolve_model):
                model_alias, _ = resolve_model(
                    preferred_model,
                    bot_id=bot_id,
                    local_mode=False,
                )

            bot_manager = BotManager(config)
            bot_obj = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()
            resolver = RuntimeSettingsResolver(
                config=config,
                bot=bot_obj,
                bot_id=bot_id,
            )

            # Budget MUST mirror _process_history_summarization exactly: the
            # summarization TRIGGER budget (default 12000), NOT the model context
            # window.  Deriving it from the model window made this probe ~10x more
            # lenient than execution, so any bot whose unsummarized backlog sat
            # between the trigger and the model window reported "0 pending" and was
            # skipped every cycle (backlog starvation).
            # TASK-610: trigger from the global Tier-1 job dict (was a per-bot
            # runtime setting — trigger is a global job param, not bot-aware).
            trigger_tokens = int(sum_job.get("trigger_tokens") or 0)
            if trigger_tokens > 0:
                max_context_tokens = trigger_tokens
            else:
                max_context_tokens = 0
                if model_alias:
                    # TASK-609: single Tier-2 budget authority
                    _, _, max_context_tokens = config.resolve_context_budget(model_alias)

            summarizer = HistorySummarizer(
                config,
                bot_id=bot_id,
                settings_getter=resolver.resolve,
                max_context_tokens=max_context_tokens,
            )
            eligible_sessions = summarizer.preview_summarizable_sessions()
            count = len(eligible_sessions)
            return count > 0, count
        except Exception as e:
            logger.debug(
                "Pending-summary probe failed for bot %s: %s",
                bot_id,
                e,
            )
            return False, 0

    @staticmethod
    def _has_datetime_activity_since(
        latest_value: datetime | None, last_run_at: datetime | None
    ) -> bool:
        """Check if a datetime value indicates new activity since last run."""
        if latest_value is None:
            return False
        if last_run_at is None:
            return True
        return latest_value > last_run_at

    def _has_new_activity_for_job(self, job: ScheduledJob) -> tuple[bool, str | None]:
        """Return whether this job has relevant new activity since last run."""
        config = self._parse_config_json(job.config_json)
        last_run_at = job.last_run_at

        try:
            if job.job_type == JobType.HISTORY_SUMMARIZATION:
                # Primary gate: if there are eligible unsummarized sessions,
                # run even without fresh message activity since last_run_at.
                has_pending, pending_count = self._has_pending_history_summaries(
                    job.bot_id,
                    config,
                )
                if has_pending:
                    return True, f"{pending_count} pending unsummarized session(s)"

                # Secondary gate: new raw activity may create soon-to-be-eligible
                # sessions even when none are currently eligible.
                latest_ts = self._max_timestamp_message_activity(job.bot_id)
                if latest_ts is None:
                    return False, "No conversation messages found"
                if last_run_at is None:
                    return False, "No eligible sessions yet"
                if latest_ts > last_run_at.timestamp():
                    return True, "New conversation activity detected"
                return False, "No new conversation activity since last run"

            if job.job_type == JobType.PROFILE_MAINTENANCE:
                entity_id = config.get("entity_id", "user")
                entity_type = config.get("entity_type", "user")
                return self._profile_needs_rebuild(entity_id, entity_type)

            if job.job_type == JobType.MEMORY_CONSOLIDATION:
                latest_memory_update = self._max_memory_updated_at(job.bot_id)
                if self._has_datetime_activity_since(latest_memory_update, last_run_at):
                    return True, None
                return False, "No memory changes since last run"

            if job.job_type == JobType.MEMORY_DECAY:
                latest_memory_update = self._max_memory_updated_at(job.bot_id)
                if latest_memory_update is None:
                    return False, "No memories available for decay"
                return True, None

            if job.job_type == JobType.MEMORY_EXTRACTION:
                # Check if there are unprocessed messages
                count = self._count_unprocessed_messages(job.bot_id)
                if count > 0:
                    return True, f"{count} unprocessed message(s)"
                return False, "No unprocessed messages"

            if job.job_type == JobType.MEDIA_GC:
                # Always-on: GC sweep is cheap when there are no orphans
                # and the worst case is a 24h pile-up at most. Activity
                # gating would add a separate scan with no real win.
                return True, None

            if job.job_type == JobType.TOOL_RESULT_GC:
                # Always-on for the same reason as MEDIA_GC — a cheap sweep.
                return True, None
        except Exception as e:
            logger.debug(
                "Activity-gate check failed for job %s (%s); falling back to run: %s",
                job.id,
                job.job_type,
                e,
            )
            return True, None

        return True, None
    
    def _create_task_for_job(self, job: ScheduledJob):
        """Create a Task from a ScheduledJob."""
        from llm_bawt.service.tasks import (
            create_history_summarization_task,
            create_maintenance_task,
            create_media_assets_gc_task,
            create_memory_extraction_task,
            create_profile_maintenance_task,
            create_tool_result_gc_task,
        )
        
        if job.job_type == JobType.PROFILE_MAINTENANCE:
            # For profile maintenance, entity_id comes from config or is the bot's user
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "user")  # Default user
            return create_profile_maintenance_task(
                entity_id=entity_id,
                entity_type=config.get("entity_type", "user"),
                bot_id=job.bot_id,
            )
        elif job.job_type == JobType.MEMORY_CONSOLIDATION:
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=True,
                run_recurrence_detection=False,
                run_decay_pruning=False,
            )
        elif job.job_type == JobType.MEMORY_DECAY:
            config = self._parse_config_json(job.config_json)
            entity_id = config.get("entity_id", "system")
            return create_maintenance_task(
                bot_id=job.bot_id,
                user_id=entity_id,
                run_consolidation=False,
                run_recurrence_detection=False,
                run_decay_pruning=True,
            )
        elif job.job_type == JobType.HISTORY_SUMMARIZATION:
            config = self._parse_config_json(job.config_json)
            return create_history_summarization_task(
                bot_id=job.bot_id,
                user_id=config.get("user_id", "system"),
                use_heuristic_fallback=config.get("use_heuristic_fallback", True),
                max_tokens_per_chunk=config.get("max_tokens_per_chunk", 4000),
                model=config.get("model"),
            )
        elif job.job_type == JobType.MEMORY_EXTRACTION:
            config = self._parse_config_json(job.config_json)
            return create_memory_extraction_task(
                bot_id=job.bot_id,
                user_id=config.get("user_id", "system"),
                batch_size=config.get("batch_size", 50),
                model=config.get("model"),
            )
        elif job.job_type == JobType.MEDIA_GC:
            config = self._parse_config_json(job.config_json)
            return create_media_assets_gc_task(
                grace_days=int(config.get("grace_days", 7)),
                dry_run=bool(config.get("dry_run", False)),
                bot_id=job.bot_id or "system",
                user_id=config.get("user_id", "system"),
            )
        elif job.job_type == JobType.TOOL_RESULT_GC:
            config = self._parse_config_json(job.config_json)
            return create_tool_result_gc_task(
                retention_days=int(config.get("retention_days", 14)),
                dry_run=bool(config.get("dry_run", False)),
                bot_id=job.bot_id or "system",
                user_id=config.get("user_id", "system"),
            )

        return None


def init_default_jobs(engine, config) -> None:
    """Initialize default scheduled jobs if they don't exist."""
    global_settings = RuntimeSettingsResolver(config=config)
    with Session(engine) as session:
        # Check if profile maintenance job exists
        existing = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.PROFILE_MAINTENANCE)
        ).first()
        
        if not existing:
            default_user = config.DEFAULT_USER or "user"
            job = ScheduledJob(
                job_type=JobType.PROFILE_MAINTENANCE,
                bot_id=WILDCARD_BOT_ID,  # All bots
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    global_settings.resolve(
                        "profile_maintenance_interval_minutes",
                        config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
                    )
                ),
                config_json=json.dumps({"entity_id": default_user, "entity_type": "user"}),
            )
            session.add(job)
            session.commit()
            logger.debug("Created default profile maintenance job")

        # Ensure a single wildcard history summarization job exists.
        # At execution time, _execute_job expands bot_id="*" to all bots
        # so newly-created bots are automatically included.
        # Migrate: remove legacy per-bot jobs if a wildcard doesn't exist yet.
        existing_history_jobs = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.HISTORY_SUMMARIZATION)
        ).all()
        existing_history_bot_ids = {j.bot_id for j in existing_history_jobs}
        if "*" not in existing_history_bot_ids:
            # Delete old per-bot jobs — the wildcard replaces them
            for old_job in existing_history_jobs:
                session.delete(old_job)
            if existing_history_jobs:
                session.commit()
                logger.info(
                    "Migrated %d per-bot history summarization jobs to wildcard",
                    len(existing_history_jobs),
                )

        existing_history = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.HISTORY_SUMMARIZATION)
        ).first()

        if not existing_history:
            history_job = ScheduledJob(
                job_type=JobType.HISTORY_SUMMARIZATION,
                bot_id=WILDCARD_BOT_ID,
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    global_settings.resolve(
                        "history_summarization_interval_minutes",
                        getattr(config, "HISTORY_SUMMARIZATION_INTERVAL_MINUTES", 30),
                    )
                ),
                config_json=json.dumps(
                    {
                        "user_id": "system",
                        "use_heuristic_fallback": True,
                        "max_tokens_per_chunk": 4000,
                    }
                ),
            )
            session.add(history_job)
            session.commit()
            logger.debug("Created wildcard history summarization job")

        # Ensure a single wildcard memory extraction job exists.
        # Migrate: remove legacy per-bot jobs if a wildcard doesn't exist yet.
        existing_extraction_jobs = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.MEMORY_EXTRACTION)
        ).all()
        existing_extraction_bot_ids = {j.bot_id for j in existing_extraction_jobs}
        if "*" not in existing_extraction_bot_ids:
            for old_job in existing_extraction_jobs:
                session.delete(old_job)
            if existing_extraction_jobs:
                session.commit()
                logger.info(
                    "Migrated %d per-bot memory extraction jobs to wildcard",
                    len(existing_extraction_jobs),
                )

        existing_extraction = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.MEMORY_EXTRACTION)
        ).first()

        if not existing_extraction:
            default_user = config.DEFAULT_USER or "user"
            extraction_job = ScheduledJob(
                job_type=JobType.MEMORY_EXTRACTION,
                bot_id=WILDCARD_BOT_ID,
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    getattr(config, "MEMORY_EXTRACTION_INTERVAL_MINUTES", 30),
                ),
                config_json=json.dumps(
                    {
                        "user_id": default_user,
                        "batch_size": 50,
                    }
                ),
            )
            session.add(extraction_job)
            session.commit()
            logger.debug("Created wildcard memory extraction job")

        # TASK-231: nightly media_assets GC. Single global job (bot_id="system"
        # — it scans the whole table, not per-bot). Runs once a day at 04:00
        # UTC: we anchor next_run_at to the next 04:00 boundary so the cadence
        # holds across restarts. interval_minutes=1440 (24h) keeps it daily.
        existing_media_gc = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.MEDIA_GC)
        ).first()

        if not existing_media_gc:
            now = datetime.now(timezone.utc)
            # Next 04:00 UTC strictly after `now`.
            next_run = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run + timedelta(days=1)

            media_gc_job = ScheduledJob(
                job_type=JobType.MEDIA_GC,
                bot_id=SYSTEM_BOT_ID,
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    getattr(config, "MEDIA_GC_INTERVAL_MINUTES", 1440),
                ),
                next_run_at=next_run,
                config_json=json.dumps(
                    {
                        "user_id": "system",
                        "grace_days": int(
                            getattr(config, "MEDIA_GC_GRACE_DAYS", 7),
                        ),
                        "dry_run": False,
                    }
                ),
            )
            session.add(media_gc_job)
            session.commit()
            logger.debug(
                "Created media_assets GC job (nightly @ 04:00 UTC, next: %s)",
                next_run.isoformat(),
            )

        # TASK-594: nightly tool-result GC. Prunes transient overflow blobs +
        # legacy payload rows past retention. Single global job (scans the whole
        # table). Anchored to 04:30 UTC so it doesn't overlap media_gc @ 04:00.
        existing_tool_result_gc = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.TOOL_RESULT_GC)
        ).first()

        if not existing_tool_result_gc:
            now = datetime.now(timezone.utc)
            next_run = now.replace(hour=4, minute=30, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run + timedelta(days=1)

            tool_result_gc_job = ScheduledJob(
                job_type=JobType.TOOL_RESULT_GC,
                bot_id=SYSTEM_BOT_ID,
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=int(
                    getattr(config, "TOOL_RESULT_GC_INTERVAL_MINUTES", 1440),
                ),
                next_run_at=next_run,
                config_json=json.dumps(
                    {
                        "user_id": "system",
                        "retention_days": int(
                            getattr(config, "TOOL_RESULT_GC_RETENTION_DAYS", 14),
                        ),
                        "dry_run": False,
                    }
                ),
            )
            session.add(tool_result_gc_job)
            session.commit()
            logger.debug(
                "Created tool_result GC job (nightly @ 04:30 UTC, next: %s)",
                next_run.isoformat(),
            )
