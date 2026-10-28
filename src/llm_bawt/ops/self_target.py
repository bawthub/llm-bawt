"""Self-management requires isolated mode (TASK-1169 S4c).

A process-mode job runs as a child of the ops service, so a job that stops,
restarts or recreates the ops service's own container would kill the worker
running it, before any receipt. Such a job must run isolated: the worker is
then a separate container, outlives the restart and writes its receipt, which
the restarted service imports normally.

Classification compares by identity, never by a name string:

- the ops service's own identity is its inspected FULL container ID plus its
  Compose project/service labels;
- each container a job targets is inspected (at prepare and again at every
  dispatch preflight), so a name alias or an abbreviated ID resolves to the
  same full ID; a Compose selector is compared by its labels;
- a target is the ops service when the full IDs match or both carry the same
  complete Compose identity (a recreated ops service has a new ID, same labels).

Refusals, raised at prepare and at preflight before any slot or spawn:

- ``self_identity_unknown``: the ops service cannot identify its own
  container, so no Docker job can be proven not to target it (fail closed);
- ``self_target_unverifiable``: a target that could be the ops service cannot
  be inspected;
- ``self_management_requires_isolated``: a process-mode job targets it;
- ``self_stop_unrecoverable``: a job would leave it stopped or removed. Nothing
  would be left running to import the receipt, in either mode.

Isolated restart, recreate, config reconcile and image deploy/rollback of the
ops service are allowed, and the approval card names the self target.
"""
from __future__ import annotations

from typing import Callable, Mapping, NamedTuple

from .container_remove import ACTION as REMOVE_ACTION
from .execution_mode import EXECUTION_PROCESS, snapshot_execution_mode
from .executor_types import ExecutorError
from .plan_reconcile import ACTION as PLAN_ACTION
from .worker import target_selector

COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
#: ``pull`` fetches an image and touches no container.
NO_CONTAINER_ACTIONS = frozenset({"pull"})
#: Actions whose end state leaves their target stopped or removed.
STOPPING_ACTIONS = frozenset({"stop", REMOVE_ACTION})

SELF_IDENTITY_UNKNOWN = "self_identity_unknown"
SELF_TARGET_UNVERIFIABLE = "self_target_unverifiable"
SELF_REQUIRES_ISOLATED = "self_management_requires_isolated"
SELF_STOP_UNRECOVERABLE = "self_stop_unrecoverable"


class SelfTargetError(ExecutorError):
    """A self-target refusal. As an ExecutorError, a dispatch preflight keeps
    the job queued with this reason until its preflight deadline fails it."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class ContainerIdentity(NamedTuple):
    container_id: str | None
    project: str | None
    service: str | None

    @classmethod
    def from_labels(cls, container_id: str | None, labels: Mapping[str, str] | None) -> "ContainerIdentity":
        labels = labels or {}
        return cls(container_id or None, labels.get(COMPOSE_PROJECT_LABEL) or None,
                   labels.get(COMPOSE_SERVICE_LABEL) or None)

    @property
    def compose(self) -> tuple[str, str] | None:
        return (self.project, self.service) if self.project and self.service else None

    def describe(self) -> str:
        if self.compose:
            return f"Compose service {self.project}/{self.service}"
        return f"container {self.container_id}"


def is_self(own: ContainerIdentity, target: ContainerIdentity) -> bool:
    if own.container_id and target.container_id == own.container_id:
        return True
    return own.compose is not None and target.compose == own.compose


def _targets(spec: Mapping, args: Mapping, inspect: Callable[[str], ContainerIdentity],
             plan_targets: list[dict] | None) -> list[ContainerIdentity]:
    action = spec.get("action")
    if action == PLAN_ACTION:
        if not plan_targets:
            raise SelfTargetError(SELF_TARGET_UNVERIFIABLE, "a service plan's targets are not bound")
        found = []
        for target in plan_targets:
            if target.get("action") == PLAN_ACTION:
                raise SelfTargetError(SELF_TARGET_UNVERIFIABLE, "a service plan cannot target another plan")
            found.extend(_targets(target, {}, inspect, None))
        return found
    found = []
    literal = ContainerIdentity(None, spec.get("compose_project"), spec.get("compose_service"))
    if literal.compose:
        found.append(literal)
    try:
        name, filters = target_selector(spec, args)
    except (KeyError, TypeError):
        name, filters = None, None
    if name is not None:
        found.append(inspect(name))
    elif filters is not None:
        # The worker selects exactly the containers carrying these labels.
        labels = dict(item.split("=", 1) for item in filters["label"])
        found.append(ContainerIdentity.from_labels(None, labels))
    if not found:
        raise SelfTargetError(SELF_TARGET_UNVERIFIABLE, f"{action!r} names no container or Compose identity")
    return found


def _plan_targets(snapshot: Mapping) -> list[dict] | None:
    plan = snapshot.get("plan")
    if not isinstance(plan, dict):
        return None
    return [target["spec"] for target in plan.get("targets", [])]


def self_target(snapshot: Mapping, *, own: Callable[[], ContainerIdentity],
                inspect: Callable[[str], ContainerIdentity]) -> str | None:
    """What this job targets of the ops service itself, else None.

    ``own`` returns the ops service's identity and raises
    :class:`SelfTargetError` (``self_identity_unknown``) when it cannot;
    ``inspect`` returns a named container's live identity and raises
    (``self_target_unverifiable``) when it cannot be read. A bound service
    plan (TASK-1476) is checked through each of its targets.
    """
    identity = own()
    spec, args = snapshot["spec"], snapshot.get("resolved_args") or {}
    if spec.get("action") in NO_CONTAINER_ACTIONS:
        return None
    targets = _targets(spec, args, inspect, _plan_targets(snapshot))
    matched = next((target for target in targets if is_self(identity, target)), None)
    if matched is None:
        return None
    described = matched.describe()
    try:
        mode = snapshot_execution_mode(snapshot, (snapshot.get("execution") or {}).get("executor_kind"))
    except ValueError as exc:
        raise SelfTargetError(SELF_TARGET_UNVERIFIABLE, str(exc)) from exc
    if mode == EXECUTION_PROCESS:
        raise SelfTargetError(SELF_REQUIRES_ISOLATED, (
            f"this job targets the ops service itself ({described}); a process-mode worker would die with it. "
            "Set the operation's execution_mode to isolated"))
    if spec.get("action") in STOPPING_ACTIONS:
        raise SelfTargetError(SELF_STOP_UNRECOVERABLE, (
            f"{spec.get('action')!r} would leave the ops service itself ({described}) stopped or removed, "
            "with nothing left to import the receipt"))
    return described
