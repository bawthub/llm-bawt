"""Approval-time binding and card text for a service plan (TASK-1476).

Runs in the app, read-only. Each target is an existing reconcile catalog row:
its own executor bind (render, compare, source guard) runs exactly as for a
standalone approval. The plan then reads, from the running app, the
``records`` probe line and the credential env NAMES the app has now and will
have after its swap, and classifies each env-import destination
(:func:`plan_reconcile.classify`). Values never leave this process; the
snapshot carries names, booleans and closed codes only.
"""
from __future__ import annotations

from ..env_import import SPECS
from .app_config_reconcile import (ACTION as APP_CONFIG_ACTION, newly_trusted, snapshot_approval_text)
from .executor import ConfigCheckError, ExecutorError, validate_spec
from .plan_reconcile import (ACCEPTED_LOSS, DESTINATIONS, KINDS, MIGRATED, CredentialLoss,
                             classify, container_env, env_names, plan_probes, probe_argv, read_probe,
                             target_snapshot)
from .probe_contract import RECORDS_PROBE
from .tagged_config_reconcile import approval_text as tagged_approval_text
from .validation import ArgValidationError, validate_args


class PlanError(ValueError):
    """A plan cannot be bound; ``code`` is fixed and the message names only."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def credential_names() -> dict:
    """``{destination: sorted env names}``: env_import's closed list."""
    return {spec.destination: sorted(set(spec.env_names)) for spec in SPECS}


def bind_plan(spec: dict, accept_loss, *, executor, get_operation, executor_kind: str,
              check_source, last_classification) -> dict:
    """The ``plan`` block of a snapshot. ``get_operation(slug)`` reads a catalog
    row, ``check_source(commit)`` refuses unpublished app source and
    ``last_classification(slug)`` is the target's last executed trust."""
    targets, containers, desired_env = [], {}, {}
    for slug in spec["targets"]:
        op = get_operation(slug)
        if op is None or not op.enabled or op.soft_deleted_at is not None:
            raise PlanError("plan_target_unavailable", f"plan target {slug} is missing or disabled")
        if op.executor_kind != executor_kind:
            raise PlanError("plan_target_invalid", f"plan target {slug} runs on another executor")
        try:
            target_spec = validate_spec(op.command_script)
            if target_spec.get("action") not in KINDS:
                raise ValueError("not an app or tagged-image config reconcile")
            if validate_args({}, op.args_schema_json, op.args_defaults_json):
                raise ValueError("a plan target accepts no arguments")
        except (ValueError, ArgValidationError) as exc:
            raise PlanError("plan_target_invalid", f"plan target {slug}: {exc}") from exc
        try:
            binding, desired, container = executor.config_bind(target_spec)
        except ConfigCheckError as exc:
            raise PlanError(exc.code, f"{slug}: {exc}") from exc
        except (ValueError, ExecutorError) as exc:
            raise PlanError("target_unavailable", f"{slug}: {exc}") from exc
        newly = None
        if binding.get("source_commit") is not None:  # app source, or a guarded tagged row (TASK-1831)
            check_source(binding["source_commit"])
        if target_spec["action"] == APP_CONFIG_ACTION:
            newly = newly_trusted(binding["classification"], last_classification(slug))
        targets.append({"slug": slug, "operation": op.to_api(), "spec": target_spec, "config": binding,
                        "newly_trusted": newly})
        containers[slug], desired_env[slug] = container, desired["env"]
    names = credential_names()
    records_slug = spec["records_target"]
    records_spec = targets[spec["targets"].index(records_slug)]["spec"]
    try:
        line = read_probe(RECORDS_PROBE, executor.probe(records_spec["container_name"], probe_argv(RECORDS_PROBE)))
    except ExecutorError as exc:
        raise PlanError("records_unavailable", f"records probe: {exc}") from exc
    except ValueError as exc:
        raise PlanError("probe_unverifiable", "records probe printed no valid line") from exc
    if not line["ok"]:
        raise PlanError("records_unavailable", f"records probe failed: {line['code']}")
    env_now = env_names(container_env(containers[records_slug]), names)
    env_next = env_names(desired_env[records_slug], names)
    accept = sorted(set(accept_loss or []))
    try:
        credentials = classify(line["records"], env_now, env_next, accept)
    except CredentialLoss as exc:
        raise PlanError(CredentialLoss.code, str(exc)) from exc
    except ValueError as exc:
        raise PlanError("args_invalid", str(exc)) from exc
    return {"targets": targets, "records": line["records"], "credential_names": names, "env_now": env_now,
            "env_next": env_next, "credentials": credentials, "accept_loss": accept,
            "probes": plan_probes(spec, credentials)}


def approval_text(snapshot: dict, prefix: str | None) -> str:
    """The plan card: order, credential classes, then every target's own card."""
    plan = snapshot["plan"]
    lines = [prefix or "Authorize the ordered service configuration plan?",
             f"Order: {' -> '.join(t['slug'] for t in plan['targets'])} (each recreated only after the "
             "previous one is verified; any failure reverts every recreated target in reverse)"]
    for label, cls in (("Migrated to stored records (probed after)", MIGRATED),
                       ("Env kept", "env_kept"), ("Unconfigured", "unconfigured")):
        dests = [d for d in DESTINATIONS if plan["credentials"][d] == cls]
        lines.append(f"{label}: {', '.join(dests) or 'none'}")
    for dest in DESTINATIONS:
        if plan["credentials"][dest] == ACCEPTED_LOSS:
            dropped = sorted(set(plan["env_now"][dest]) - set(plan["env_next"][dest]))
            lines.append(f"ACCEPTED LOSS: {dest} stops working (drops {', '.join(dropped)}; no stored record)")
    lines.append("Probes after: " + (", ".join(f"{p['id']}@{p['target']}" for p in plan["probes"]) or "none"))
    for index, target in enumerate(plan["targets"], 1):
        if target["spec"]["action"] == APP_CONFIG_ACTION:
            body = snapshot_approval_text(target_snapshot(target), f"[{index}] {target['slug']}")
        else:
            body = tagged_approval_text(target["spec"], target["config"], f"[{index}] {target['slug']}")
        lines.append(body)
    return "\n".join(lines)


__all__ = ["PlanError", "approval_text", "bind_plan", "credential_names"]
