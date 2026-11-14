"""Bounded operation-plan contract and durable parent ledger. STANDARD LIBRARY ONLY.

Admission follows the existing service-plan binding pattern. Execution stays in
the existing worker, using the same per-action handlers as a standalone job.
There are no child jobs, callbacks, recovery dispatches or automatic retries.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone

ACTION = "operation_sequence"
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.operation-sequence"
SCHEMA = "llm-bawt.ops.sequence/v1"
MAX_STEPS = 6
TAIL_BYTES = 1024
MAX_LEDGER_BYTES = 65536
SUPPORTED = frozenset({"restart", "start", "reconcile_app_config", "reconcile_tagged_config"})
SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
TERMINAL = frozenset({"succeeded", "failed", "skipped", "lost"})


def validate_spec(spec: dict) -> dict:
    if (not isinstance(spec, dict) or set(spec) != {"action", "operations"}
            or spec["action"] != ACTION or not isinstance(spec["operations"], list)
            or not 1 <= len(spec["operations"]) <= 64
            or not all(isinstance(s, str) and SLUG.fullmatch(s) for s in spec["operations"])
            or len(set(spec["operations"])) != len(spec["operations"])):
        raise ValueError("sequence spec requires action and 1..64 distinct allowed operation slugs")
    return spec


def validate_steps(spec: dict, args: dict) -> list[dict]:
    steps = args.get("steps")
    if set(args) != {"steps"} or not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError(f"sequence requires 1..{MAX_STEPS} ordered steps")
    for step in steps:
        if (not isinstance(step, dict) or set(step) != {"operation", "args"}
                or not isinstance(step["operation"], str) or step["operation"] not in spec["operations"]
                or not isinstance(step["args"], dict)):
            raise ValueError("each step must name an allowed operation and its args object")
    return steps


def snapshots(snapshot: dict) -> list[dict]:
    return [step["snapshot"] for step in snapshot.get("sequence", {}).get("steps", [])]


def step_budget(snapshot: dict) -> int:
    execution = snapshot["execution"]
    return (execution["start_delay_seconds"] + execution["timeout_seconds"]
            + snapshot["spec"].get("health_timeout_seconds", 0))


def validate_binding(snapshot: dict) -> None:
    """Validate the sequence wrapper; each nested snapshot is verified by OpsService."""
    spec = validate_spec(snapshot["spec"])
    requested = validate_steps(spec, snapshot["resolved_args"])
    bound = snapshot["sequence"]["steps"]
    if len(bound) != len(requested):
        raise ValueError("sequence binding length mismatch")
    if snapshot["operation"]["risk_level"] != "critical":
        raise ValueError("initial service sequences require critical risk")
    identities = set()
    names = set()
    for request, step in zip(requested, bound):
        child, target = step["snapshot"], step["target"]
        if (set(step) != {"snapshot", "target", "display_args"} or child["operation"]["slug"] != request["operation"]
                or child["input_args"] != request["args"] or child["spec"]["action"] not in SUPPORTED
                or not isinstance(step["display_args"], dict)
                or child["execution"]["executor_kind"] != snapshot["execution"]["executor_kind"]
                or child["execution"]["worker_image"] != snapshot["execution"]["worker_image"]
                or any(child.get(key) for key in ("self_target", "gpu_target", "gpu_stop_target"))
                or type(child["spec"].get("health_timeout_seconds")) is not int
                or not 10 <= child["spec"]["health_timeout_seconds"] <= 600
                or set(target) != {"id", "name"}
                or not re.fullmatch(r"[0-9a-f]{64}", target["id"])
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", target["name"])
                or target["id"] in identities or target["name"] in names):
            raise ValueError("sequence target binding invalid or duplicate")
        identities.add(target["id"])
        names.add(target["name"])
    if snapshot["execution"]["timeout_seconds"] < sum(step_budget(s) for s in snapshots(snapshot)):
        raise ValueError("parent timeout must cover every step's delay, execution and health budget")
    if not snapshot.get("quiesce") or snapshot["quiesce"]["scope"] != {"all_agents": True}:
        raise ValueError("service sequences require whole-window all-agent quiesce")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail(text) -> str:
    return str(text or "").encode()[-TAIL_BYTES:].decode(errors="replace")


def read_ledger(value, snapshot=None) -> dict | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("state") not in {"pending", "running", "succeeded", "failed", "lost_requires_inspection"}
            or not isinstance(value.get("steps"), list) or not 1 <= len(value["steps"]) <= MAX_STEPS
            or len(json.dumps(value, allow_nan=False).encode()) > MAX_LEDGER_BYTES):
        raise ValueError("invalid or oversized sequence ledger")
    for index, step in enumerate(value["steps"], 1):
        if (not isinstance(step, dict) or step.get("step") != index
                or not isinstance(step.get("op"), str) or not SLUG.fullmatch(step["op"])
                or not isinstance(step.get("args"), dict)
                or not isinstance(step.get("target"), str)
                or step.get("state") not in TERMINAL | {"pending", "running"}
                or step.get("exit") not in (None, 0, 1)
                or not all(step.get(key) is None or isinstance(step[key], dict) for key in ("health", "rollback"))
                or not isinstance(step.get("output_tail"), str)
                or len(step["output_tail"].encode()) > TAIL_BYTES + 3
                or not all(step.get(key) is None or isinstance(step[key], str)
                           for key in ("started_at", "finished_at", "reason"))):
            raise ValueError("invalid sequence step")
    if value["state"] == "succeeded" and any(s["state"] != "succeeded" for s in value["steps"]):
        raise ValueError("sequence success requires all steps")
    if value["state"] in ("failed", "lost_requires_inspection") and any(
            s["state"] not in TERMINAL for s in value["steps"]):
        raise ValueError("terminal sequence has unfinished steps")
    if snapshot is not None:
        expected = initial_ledger(snapshot)["steps"]
        if len(expected) != len(value["steps"]) or any(
                any(actual[key] != wanted[key] for key in ("step", "op", "args", "target"))
                for actual, wanted in zip(value["steps"], expected)):
            raise ValueError("sequence ledger does not match the approved steps")
    return value


def initial_ledger(snapshot: dict) -> dict:
    return {"schema": SCHEMA, "state": "pending", "steps": [
        {"step": index, "op": item["snapshot"]["operation"]["slug"],
         # Display arguments are already redacted by the app, never resolved secrets.
         "args": item["display_args"], "target": item["target"]["name"],
         "state": "pending", "exit": None, "health": None, "rollback": None,
         "started_at": None, "finished_at": None, "output_tail": "", "reason": None}
        for index, item in enumerate(snapshot["sequence"]["steps"], 1)]}


def interrupted(ledger: dict, reason: str, *, unknown=True) -> dict:
    """Report stop position without replay. Recorded results are never erased."""
    ledger = copy.deepcopy(ledger)
    for step in ledger["steps"]:
        if step["state"] in TERMINAL:
            continue
        step["state"] = "lost" if step["state"] == "running" else "skipped"
        step["reason"] = tail(reason)
        step["finished_at"] = now()
    ledger["state"] = "lost_requires_inspection" if unknown else "failed"
    return ledger


def run(snapshot: dict, *, api, preflight, execute, checkpoint, sleep, clock=now) -> dict:
    """One worker, strict order, durable intent, no restart of unfinished work."""
    ledger = initial_ledger(snapshot)
    checkpoint(ledger)
    ledger["state"] = "running"
    active = None
    try:
        # Prove all bindings before touching the first target.
        for item in snapshot["sequence"]["steps"]:
            preflight(item)
        for item, step in zip(snapshot["sequence"]["steps"], ledger["steps"]):
            active = step
            api.side_effect_started = False
            sleep(item["snapshot"]["execution"]["start_delay_seconds"])
            # Recheck immediately before this step, after earlier targets changed.
            preflight(item)
            step.update(state="running", started_at=clock())
            checkpoint(ledger)  # intent BEFORE the shared handler can submit a mutation
            output = execute(item)
            step.update(state="succeeded", exit=0, health={"state": "healthy"},
                        finished_at=clock(), output_tail=tail(output))
            checkpoint(ledger)
            active = None
    except Exception as exc:  # worker interruption/SIGALRM also retains the ledger
        uncertain = bool(active and active["state"] == "running"
                         and api.side_effect_started and not getattr(exc, "outcome_known", False))
        if active and active["state"] in TERMINAL:
            # Its mutation ran and was verified; only persisting that failed.
            # Keep the known result and stop the remaining steps.
            active["reason"] = tail(f"progress checkpoint failed after this step ({type(exc).__name__})")
        elif active:
            submitted = active["state"] == "running"
            active.update(state="lost" if uncertain else "failed" if submitted else "skipped",
                          exit=None if uncertain or not submitted else 1,
                          finished_at=clock(), reason=tail(type(exc).__name__))
            detail = getattr(exc, "detail", {})
            active["rollback"] = {"restored": bool(detail.get("restored"))} if detail else None
            active["health"] = {"state": "unverified" if uncertain else "failed"} if submitted else None
        for step in ledger["steps"]:
            if step["state"] == "pending":
                step.update(state="skipped", finished_at=clock(), reason="earlier step or preflight failed")
        ledger["state"] = "lost_requires_inspection" if uncertain else "failed"
        checkpoint(ledger)
        return ledger
    ledger["state"] = "succeeded"
    checkpoint(ledger)
    return ledger
