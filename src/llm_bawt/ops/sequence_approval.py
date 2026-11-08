"""Catalog-owned operation plan binding, using ordinary single-op admission."""
from __future__ import annotations

import json

from . import sequence
from .executor import validate_spec
from .house_guard import house_targets
from .restart_health import has_health


def bind(service, snapshot: dict, display_args) -> dict:
    if snapshot["operation"]["risk_level"] != "critical":
        raise ValueError("initial service sequences require critical risk")
    steps = []
    for request in sequence.validate_steps(snapshot["spec"], snapshot["resolved_args"]):
        op = service.store.get_operation_by_slug(request["operation"])
        if op is None or not op.enabled or op.soft_deleted_at is not None:
            raise ValueError("sequence target operation unavailable")
        spec = validate_spec(op.command_script)
        if spec["action"] not in sequence.SUPPORTED:
            raise ValueError("unsupported sequence operation kind")
        if spec["action"] in ("restart", "start") and not has_health(spec):
            raise ValueError("sequence restart/start requires a bounded health check")
        # This is intentionally the exact existing admission path, not a partial
        # reimplementation of runtime source, schema, capability or GPU checks.
        child = service.prepare_invocation(request["operation"], request["args"])
        if child.get("self_target") or child.get("gpu_target") or child.get("gpu_stop_target"):
            raise ValueError("sequence cannot target its executor or a GPU handoff")
        if house_targets(child["spec"], child["resolved_args"]):
            raise ValueError("house operations are not supported in sequences")
        executor = service._resolve_executor(child["execution"]["executor_kind"])
        if (child["execution"]["executor_kind"] != snapshot["execution"]["executor_kind"]
                or child["execution"]["worker_image"] != snapshot["execution"]["worker_image"]):
            raise ValueError("sequence steps must share the parent executor and worker")
        executor.preflight(child)
        target = executor.sequence_target(child)
        steps.append({"snapshot": child, "target": target,
                      "display_args": display_args(child["resolved_args"], child["operation"]["args_schema_json"])})
    return {"steps": steps}


def approval_text(snapshot: dict, prefix: str | None) -> str:
    lines = [prefix or "Authorize this ordered service operation sequence?",
             "One approval and terminal callback; quiesce held for the entire sequence.",
             "Stop on first failure. Earlier successful restarts are not undone.",
             "Interrupted mutations are reported for inspection, NEVER automatically retried."]
    for index, item in enumerate(snapshot["sequence"]["steps"], 1):
        child, target = item["snapshot"], item["target"]
        lines.append(f"[{index}] {child['operation']['slug']} args={json.dumps(item['display_args'], sort_keys=True)}"
                     f" -> {target['name']} ({target['id']}); health budget"
                     f" {child['spec']['health_timeout_seconds']}s")
        if child["spec"]["action"] == "reconcile_app_config":
            from .app_config_reconcile import snapshot_approval_text

            lines.append(snapshot_approval_text(child, None))
        elif child["spec"]["action"] == "reconcile_tagged_config":
            from .tagged_config_reconcile import approval_text as tagged_text

            lines.append(tagged_text(child["spec"], child["config"], None))
    return "\n".join(lines)
