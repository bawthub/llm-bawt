"""GPU voice guard: no ops job starts a Moshi container outside a GPU handoff (TASK-1171).

Starting or recreating BawtHub's Moshi STT/TTS while the GPU belongs to video
would put both models on one card. Only the GPU handoff controller may
dispatch such a job, with a :class:`GpuHandoffFence` it built, and the job
dispatches only while the handoff row still admits its idempotency key.

The predicate is deny-by-default on action: every action except the two
below is gated, so an action added later is gated until it is classified.
"""
from __future__ import annotations

from .executor import ExecutorError
from .image_deploy import IMAGE_ACTIONS
from .release_spec import RELEASE_ACTION
from .worker import target_selector

#: The product's own GPU voice services, by Compose identity.
GPU_VOICE_PROJECT = "bawthub"
GPU_VOICE_SERVICES = frozenset({"stt", "tts"})
#: ``stop`` frees VRAM and ``pull`` touches no container. A release spec runs
#: on the release executor, which dispatches a workflow and no container; its
#: image deploy is a separate ``deploy_image`` job, gated on its own.
UNGATED_ACTIONS = frozenset({"stop", "pull", RELEASE_ACTION})
#: Actions the worker resolves with :func:`target_selector`; every other
#: action names its target by literal Compose identity plus container_name.
SELECTOR_ACTIONS = frozenset({"start", "restart"})


class GpuGuardUnverifiable(Exception):
    """The target's Compose identity could not be read; the guard fails closed."""


def _gated(project, service) -> bool:
    return project == GPU_VOICE_PROJECT and service in GPU_VOICE_SERVICES


def _inspect(executor, spec, args) -> tuple[str | None, str | None] | None:
    try:
        return executor.target_compose_identity(spec, args)
    except ExecutorError as exc:
        raise GpuGuardUnverifiable(f"cannot read the target's Compose identity: {exc}") from exc


def gpu_voice_target(spec: dict, resolved_args: dict, executor, *, live: bool = False) -> str | None:
    """The gated voice service this spec would start or recreate, else None.

    ``live`` is the dispatch-time re-check: it also reads the labels of a
    literally named container, so a container renamed or relabelled after the
    job was created still gates. Raises :class:`GpuGuardUnverifiable` when the
    identity cannot be established.
    """
    action = spec.get("action")
    if action in UNGATED_ACTIONS:
        return None
    if action in SELECTOR_ACTIONS:
        try:
            name, filters = target_selector(spec, resolved_args)
        except (KeyError, TypeError) as exc:
            raise GpuGuardUnverifiable(f"cannot resolve the target: {exc}") from exc
        if name is None:
            # The worker selects exactly the containers carrying these labels.
            labels = dict(item.split("=", 1) for item in filters["label"])
            identities = [(labels.get("com.docker.compose.project"), labels.get("com.docker.compose.service"))]
        else:
            identities = [_inspect(executor, spec, resolved_args)]
    else:
        literal = (spec.get("compose_project"), spec.get("compose_service"))
        if None in literal and not spec.get("container_name"):
            raise GpuGuardUnverifiable(f"{action!r} names no Compose identity or container")
        identities = [literal]
        if live and spec.get("container_name"):
            identities.append(_inspect(executor, spec, resolved_args))
    for identity in identities:
        if identity is not None and _gated(*identity):
            return identity[1]
    return None

