"""GPU voice guard: no ops job starts a Moshi container outside a GPU handoff (TASK-1171).

Starting or recreating BawtHub's Moshi STT/TTS while the GPU belongs to video
would put both models on one card. Only the GPU handoff controller may
dispatch such a job, with a :class:`GpuHandoffFence` it built, and the job
dispatches only while the handoff row still admits its idempotency key.

The predicate is deny-by-default on action: every action except those in
UNGATED_ACTIONS below is gated, so an action added later is gated until it is
classified.
"""
from __future__ import annotations

from .container_remove import ACTION as REMOVE_ACTION
from .executor import ExecutorError
from .image_deploy import IMAGE_ACTIONS
from .plan_reconcile import ACTION as PLAN_ACTION
from .release_spec import RELEASE_ACTION
from .worker import target_selector

#: The product's own GPU voice services, by Compose identity.
GPU_VOICE_PROJECT = "bawthub"
GPU_VOICE_SERVICES = frozenset({"stt", "tts"})
#: ``stop`` frees VRAM and ``pull`` touches no container. A release spec runs
#: on the release executor, which dispatches a workflow and no container; its
#: image deploy is a separate ``deploy_image`` job, gated on its own.
#: ``remove`` (TASK-1439 T2c) deletes one stopped container and starts nothing.
UNGATED_ACTIONS = frozenset({"stop", "pull", RELEASE_ACTION, REMOVE_ACTION})
#: Actions the worker resolves with :func:`target_selector`; every other
#: action names its target by literal Compose identity plus container_name.
SELECTOR_ACTIONS = frozenset({"start", "restart"})


class GpuGuardUnverifiable(Exception):
    """The target's Compose identity could not be read; the guard fails closed."""


def _gated(project, service) -> bool:
    return project == GPU_VOICE_PROJECT and service in GPU_VOICE_SERVICES


def _complete(identity) -> bool:
    return (isinstance(identity, tuple) and len(identity) == 2
            and all(isinstance(part, str) and part for part in identity))


def _inspect(executor, spec, args) -> tuple[str, str]:
    """The named container's live Compose identity. A container without both
    labels could still be a voice service, so a missing or partial identity
    refuses like an unreadable one."""
    try:
        identity = executor.target_compose_identity(spec, args)
    except ExecutorError as exc:
        raise GpuGuardUnverifiable(f"cannot read the target's Compose identity: {exc}") from exc
    if not _complete(identity):
        raise GpuGuardUnverifiable(f"the target container carries no complete Compose identity: {identity!r}")
    return identity


def gpu_voice_target(spec: dict, resolved_args: dict, executor, *, live: bool = False,
                     plan_targets: list[dict] | None = None) -> str | None:
    """The gated voice service this spec would start or recreate, else None.

    ``live`` is the dispatch-time re-check: it also reads the labels of a
    literally named container, so a container renamed or relabelled after the
    job was created still gates. Raises :class:`GpuGuardUnverifiable` when the
    identity cannot be established.

    A service plan (TASK-1476) names no container itself: it recreates each of
    its bound targets, so it is gated exactly when one of them is.
    ``plan_targets`` are those bound target specs; a plan without them refuses.
    """
    action = spec.get("action")
    if action in UNGATED_ACTIONS:
        return None
    if action == PLAN_ACTION:
        if not plan_targets:
            raise GpuGuardUnverifiable("a service plan's targets are not bound")
        for target in plan_targets:
            if target.get("action") == PLAN_ACTION:
                raise GpuGuardUnverifiable("a service plan cannot target another plan")
            service = gpu_voice_target(target, {}, executor, live=live)
            if service is not None:
                return service
        return None
    if action in SELECTOR_ACTIONS:
        try:
            name, filters = target_selector(spec, resolved_args)
        except (KeyError, TypeError) as exc:
            raise GpuGuardUnverifiable(f"cannot resolve the target: {exc}") from exc
        if name is None:
            # The worker selects exactly the containers carrying these labels.
            labels = dict(item.split("=", 1) for item in filters["label"])
            identity = (labels.get("com.docker.compose.project"), labels.get("com.docker.compose.service"))
            if not _complete(identity):
                raise GpuGuardUnverifiable(f"the target selector names no complete Compose identity: {identity!r}")
            identities = [identity]
        else:
            identities = [_inspect(executor, spec, resolved_args)]
    else:
        literal = (spec.get("compose_project"), spec.get("compose_service"))
        if _complete(literal):
            identities = [literal]
            if live and spec.get("container_name"):
                identities.append(_inspect(executor, spec, resolved_args))
        elif spec.get("container_name"):
            # A partial literal identity proves nothing: read the container's.
            identities = [_inspect(executor, spec, resolved_args)]
        else:
            raise GpuGuardUnverifiable(f"{action!r} names no Compose identity or container")
    for identity in identities:
        if _gated(*identity):
            return identity[1]
    return None

