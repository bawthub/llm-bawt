"""Trusted operator API adapter for a scoped GPU handoff.

The one-use handoff confirmation approves only the persisted, named operations,
like the existing HTTP operator ops surface. It is not an agent MCP bypass:
agents must still use approval-gated ops tools, not simulate operator consent.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from agent_bridge.approval import PolicyAction, evaluate

from llm_bawt.integrations.home_audio import HomeAudioSettings

from .gpu_handoff import START_OPERATIONS, STOP_OPERATIONS, HandoffAdapter
from .gpu_handoff_store import TTS_RESTART_OPERATION, GpuHandoffFence, HandoffConflict


log = logging.getLogger(__name__)

#: The product's own voice services, by Compose identity.
VOICE_PROJECT = "bawthub"
VOICE_SERVICES = ("stt", "tts")
#: The reconciler's /v1/health/services schema this adapter reads.
SERVICES_SCHEMA_VERSION = 1
_INSTANCE_STATES = frozenset({"created", "running", "paused", "restarting", "removing", "exited", "dead"})
_INSTANCE_HEALTH = frozenset({"healthy", "unhealthy", "starting", None})


def _field(mapping, key, types, service_id):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if not isinstance(value, types) or isinstance(value, bool) and bool not in types:
        raise HandoffConflict(f"Container observation for {service_id} is malformed ({key})")
    return value


def parse_service_observation(service_id, status_code, response):
    """One fresh instance of ``service_id`` in the handoff's consumer shape.

    Refuses, with the reason, on: 404 (not catalogued), 503 or ``fresh:false``
    (stale or unavailable), a schema change, ``instances:null``, zero or
    several instances, or any missing or ill-typed field.
    """
    try:
        return _parse_service_observation(service_id, status_code, response)
    except HandoffConflict as exc:
        if exc.code == "handoff_conflict":
            exc.code = "observation_unavailable"
        raise


def _parse_service_observation(service_id, status_code, response):
    if status_code == 404:
        raise HandoffConflict(f"{service_id} is not a catalogued ops service")
    try:
        body = response.json()
    except ValueError as exc:
        raise HandoffConflict(f"Container observation for {service_id} is malformed") from exc
    if not isinstance(body, dict) or body.get("schemaVersion") != SERVICES_SCHEMA_VERSION:
        raise HandoffConflict(f"Container observation for {service_id} has an unsupported schema")
    if status_code != 200 or body.get("fresh") is not True:
        reason = body.get("reason") if isinstance(body.get("reason"), str) else "unavailable"
        raise HandoffConflict(f"Container observation for {service_id} is not fresh: {reason[:120]}")
    service = _field(body, "service", (dict,), service_id)
    if service.get("id") != service_id:
        raise HandoffConflict(f"Container observation answered for another service than {service_id}")
    instances = service.get("instances")
    if instances is None:
        raise HandoffConflict(f"Container observation for {service_id} has no current inventory")
    if not isinstance(instances, list):
        raise HandoffConflict(f"Container observation for {service_id} is malformed (instances)")
    if len(instances) != 1:
        raise HandoffConflict(
            f"{service_id} has {len(instances)} instances; the GPU handoff supports exactly one",
            code="ambiguous_instances")
    instance = instances[0]
    instance_id = _field(instance, "instanceId", (str,), service_id)
    state = _field(instance, "state", (str,), service_id)
    health = instance.get("health") if isinstance(instance, dict) else None
    if state not in _INSTANCE_STATES or health not in _INSTANCE_HEALTH or "health" not in instance:
        raise HandoffConflict(f"Container observation for {service_id} is malformed (state)")
    started = instance.get("startedAt")
    if started is not None:
        try:
            parsed = datetime.fromisoformat(started) if isinstance(started, str) else None
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            raise HandoffConflict(f"Container observation for {service_id} is malformed (startedAt)")
    restarts = instance.get("restartCount")
    if restarts is not None and (type(restarts) is not int or restarts < 0):
        raise HandoffConflict(f"Container observation for {service_id} is malformed (restartCount)")
    return {"id": instance_id, "running": state == "running", "status": state, "health": health,
            "started_at": started, "restart_count": restarts}


def _snapshot_diff(old, new, path="") -> list[str]:
    """Differing key paths (values truncated) between two operation snapshots."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [d for key in sorted(set(old) | set(new), key=str)
                for d in _snapshot_diff(old.get(key), new.get(key), f"{path}.{key}")]
    return [] if old == new else [f"{path}: {str(old)[:80]!r} -> {str(new)[:80]!r}"]


class LocalHandoffAdapter(HandoffAdapter):
    def __init__(self, *, config, http, video, ops, policies):
        self.config, self.http, self.video = config, http, video
        self.ops, self.policies = ops, policies
        self._voice_url: str | None = None

    async def voice_url(self):
        if self._voice_url is None:
            settings = await asyncio.to_thread(HomeAudioSettings.load, self.config)
            self._voice_url = settings.tts_url.rstrip("/")
        return self._voice_url

    async def _services(self):
        """The Moshi containers as the ops reconciler observes them (TASK-1171).

        The app holds no Docker client. Anything short of one fresh, typed
        instance per service refuses; nothing here picks among instances.
        """
        return {service: await self.observe_service(service) for service in VOICE_SERVICES}

    async def observe_service(self, service):
        service_id = f"{VOICE_PROJECT}.{service}"
        base = self.config.OPS_HEALTH_URL.rstrip("/")
        try:
            response = await self.http.get(f"{base}/v1/health/services/{service_id}", timeout=3)
        except httpx.HTTPError as exc:
            raise HandoffConflict(f"Container observation unavailable for {service_id}",
                                  code="observation_unavailable") from exc
        return parse_service_observation(service_id, response.status_code, response)

    async def observe(self):
        response = await self.http.get((await self.voice_url()) + "/v1/internal/voice/sessions", timeout=3)
        response.raise_for_status()
        gpu, model, services = await asyncio.gather(self.video.gpu_telemetry(), self.video.model_status(),
                                                   self._services())
        return {"voice": response.json(), "gpu": gpu, "model": model, "services": services}

    async def prepare_operations(self, slugs=STOP_OPERATIONS):
        return [await asyncio.to_thread(self.ops.prepare_invocation, slug, {}) for slug in slugs]

    async def check_operations(self, snapshots, slugs=STOP_OPERATIONS):
        if slugs not in (STOP_OPERATIONS, START_OPERATIONS) or len(snapshots) != 2:
            raise HandoffConflict("Expected exactly two scoped Moshi operations")
        action = "stop" if slugs == STOP_OPERATIONS else "start"
        bundle = await asyncio.to_thread(self.policies.compile_bundle)
        for snapshot, slug, service in zip(snapshots, slugs, ("stt", "tts"), strict=True):
            expected = {"action": action, "compose_project": "bawthub", "compose_service": service}
            if (snapshot["operation"]["slug"] != slug or snapshot["spec"] != expected
                    or snapshot["resolved_args"] != {} or snapshot["input_args"] != {}
                    or snapshot["execution"]["executor_kind"] != "docker"
                    or snapshot["execution"]["target_host"] != ""):
                raise HandoffConflict(f"Catalog operation is not the exact local Moshi {action}")
            # Recheck live enablement/identity/spec; never broaden an offered plan.
            current = await asyncio.to_thread(self.ops.prepare_invocation, slug, {})
            if current["snapshot_hash"] != snapshot["snapshot_hash"]:
                log.warning("GPU handoff operation %s snapshot drift: %s", slug,
                            _snapshot_diff(snapshot, current))
                raise HandoffConflict("Operation changed; request a fresh switch offer")
            decision = evaluate(bundle.policies, "http-operator", "ops_run", {"operation": slug, "args": {}})
            if decision.action is PolicyAction.DENY:
                raise HandoffConflict("Moshi operation denied by existing ops policy")
            # REQUIRE_APPROVAL is fulfilled ONLY by GpuHandoff.confirm's scoped,
            # one-use user confirmation. Preparing/checking never dispatches.

    async def park(self):
        response = await self.http.post((await self.voice_url()) + "/v1/internal/voice/park", timeout=3)
        response.raise_for_status()
        return response.json()["lease_id"]

    async def unpark(self, lease_id):
        response = await self.http.post((await self.voice_url()) + "/v1/internal/voice/unpark",
                                        json={"lease_id": lease_id}, timeout=3)
        response.raise_for_status()

    async def reset_video(self):
        await self.video.reset_worker()

    async def dispatch(self, snapshot, *, user, generation):
        slug = snapshot["operation"]["slug"]
        # A Moshi start is created only under the handoff's own fence; the
        # ops guard refuses it otherwise (TASK-1171).
        fence = (GpuHandoffFence.voice_start(generation=generation, operation_slug=slug)
                 if slug in START_OPERATIONS else None)
        return await asyncio.to_thread(
            self.ops.dispatch_job, operation_slug=slug, args={},
            approved_snapshot=snapshot,
            idempotency_key=f"gpu-handoff:{generation}:{slug}",
            caller_actor=f"gpu-handoff-consent:{generation}", caller_user_id=user,
            caller_backend="http-operator", create_fence=fence,
        )

    async def prepare_tts_restart(self):
        snapshot = await asyncio.to_thread(self.ops.prepare_invocation, TTS_RESTART_OPERATION, {})
        expected = {"action": "restart", "compose_project": VOICE_PROJECT, "compose_service": "tts"}
        if (snapshot["operation"]["slug"] != TTS_RESTART_OPERATION or snapshot["spec"] != expected
                or snapshot["resolved_args"] != {} or snapshot["input_args"] != {}
                or snapshot["execution"]["executor_kind"] != "docker"
                or snapshot["execution"]["target_host"] != ""):
            raise HandoffConflict("Catalog operation is not the exact local TTS restart",
                                  code="snapshot_invalid")
        bundle = await asyncio.to_thread(self.policies.compile_bundle)
        decision = evaluate(bundle.policies, "http-operator", "ops_run",
                            {"operation": TTS_RESTART_OPERATION, "args": {}})
        if decision.action is PolicyAction.DENY:
            raise HandoffConflict("TTS restart denied by existing ops policy", code="policy_denied")
        return snapshot

    async def dispatch_tts_restart(self, snapshot, *, key):
        return await asyncio.to_thread(
            self.ops.dispatch_job, operation_slug=TTS_RESTART_OPERATION, args={},
            approved_snapshot=snapshot, idempotency_key=key,
            caller_actor="bawthub-tts-admin", caller_backend="http-operator",
            create_fence=GpuHandoffFence.tts_restart(key),
        )

    async def job_by_key(self, key):
        job = await asyncio.to_thread(self.ops.store.get_job_by_key, key)
        return await self.job_status(job.id, reconcile=False) if job is not None else None

    async def job_status(self, job_id, *, reconcile=True):
        return await asyncio.to_thread(self.ops.get_job_status, job_id, reconcile_if_active=reconcile)

    async def find_operation(self, *, generation: int, slug: str):
        key = f"gpu-handoff:{generation}:{slug}"
        job = await asyncio.to_thread(self.ops.store.get_job_by_key, key)
        return await self.job_status(job.id) if job is not None else None
