"""Trusted operator API adapter for a scoped GPU handoff.

The one-use handoff confirmation approves only the persisted, named operations,
like the existing HTTP operator ops surface. It is not an agent MCP bypass:
agents must still use approval-gated ops tools, not simulate operator consent.
"""
from __future__ import annotations

import asyncio
import logging

from agent_bridge.approval import PolicyAction, evaluate

from llm_bawt.integrations.home_audio import HomeAudioSettings

from .gpu_handoff import START_OPERATIONS, STOP_OPERATIONS, HandoffAdapter
from .gpu_handoff_store import HandoffConflict


log = logging.getLogger(__name__)


def _snapshot_diff(old, new, path="") -> list[str]:
    """Differing key paths (values truncated) between two operation snapshots."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [d for key in sorted(set(old) | set(new), key=str)
                for d in _snapshot_diff(old.get(key), new.get(key), f"{path}.{key}")]
    return [] if old == new else [f"{path}: {str(old)[:80]!r} -> {str(new)[:80]!r}"]


class LocalHandoffAdapter(HandoffAdapter):
    def __init__(self, *, config, http, video, ops, policies, docker_factory=None):
        self.config, self.http, self.video = config, http, video
        self.ops, self.policies = ops, policies
        self.docker_factory = docker_factory
        self._voice_url: str | None = None

    async def voice_url(self):
        if self._voice_url is None:
            settings = await asyncio.to_thread(HomeAudioSettings.load, self.config)
            self._voice_url = settings.tts_url.rstrip("/")
        return self._voice_url

    def _services(self):
        import docker

        factory = self.docker_factory or (lambda: docker.from_env(timeout=5))
        # docker-py's DockerClient is not a context manager; close explicitly.
        client = factory()
        try:
            result = {}
            for service in ("stt", "tts"):
                containers = client.containers.list(all=True, filters={"label": [
                    "com.docker.compose.project=bawthub", f"com.docker.compose.service={service}",
                ]})
                if len(containers) != 1:
                    raise HandoffConflict("Moshi service selector must resolve exactly one container")
                container = containers[0]
                container.reload()
                state = container.attrs["State"]
                result[service] = {"id": container.id, "running": state.get("Running"),
                                   "status": state.get("Status"), "health": state.get("Health", {}).get("Status")}
            return result
        finally:
            client.close()

    async def observe(self):
        response = await self.http.get((await self.voice_url()) + "/v1/internal/voice/sessions", timeout=3)
        response.raise_for_status()
        gpu, model, services = await asyncio.gather(self.video.gpu_telemetry(), self.video.model_status(),
                                                   asyncio.to_thread(self._services))
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
        return await asyncio.to_thread(
            self.ops.dispatch_job, operation_slug=snapshot["operation"]["slug"], args={},
            approved_snapshot=snapshot,
            idempotency_key=f"gpu-handoff:{generation}:{snapshot['operation']['slug']}",
            caller_actor=f"gpu-handoff-consent:{generation}", caller_user_id=user,
            caller_backend="http-operator",
        )

    async def job_status(self, job_id):
        return await asyncio.to_thread(self.ops.get_job_status, job_id)
