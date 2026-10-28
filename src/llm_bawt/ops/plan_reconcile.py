"""Ordered same-image reconcile of several Compose services as ONE job. STANDARD LIBRARY ONLY (TASK-1476).

The cutover plan recreates the app, crawl4ai and the agent bridges together:
each target alone would leave the others reading a config the first no longer
serves. A plan is an ordered list of existing reconcile catalog rows (the rows
are the allowlist: their env names and data mounts stay DB-owned) plus a
closed probe list. It reuses each target's own reconciler (bind, compare,
source guard, dependency check, swap, verify, restore); nothing is forked.

Order, all under every target's lock (taken in plan order, non-blocking):

1. **Preflight, read-only.** Every target is inspected, re-rendered, compared
   with its approved binding and given its last-read checks. The ``records``
   probe runs in the running app and, with the app's env names now and as
   declared, must reproduce the approved credential classes exactly. Named
   volumes are checked. Any failure: nothing changed.
2. **Volumes.** Missing project volumes are created with Compose's labels
   (the job's first write), so a later ``compose up`` adopts them.
3. **Each target in order:** the last-read recheck right before its swap
   (re-bind, compare, running and healthy, source and dependencies), then
   verify-only (no drift) or swap. A swapped target's previous container
   stays stopped as ``<name>-prev-<job8>`` until the end.
4. **End checks:** the new app reports the same records, carries exactly the
   approved credential env names, and every plan probe passes.
5. **Commit:** every ``-prev`` container is deleted.

Any failure at or after step 2 reverts every swapped target in reverse order
(continuing past a failed revert, which makes the outcome uncertain) and,
after a full revert, removes the volumes this job created.

Credential classes (r2 B2): a destination whose stored record is connected is
``migrated`` and its probe must pass afterwards. One whose env names the
running app has but the declared app drops, with no record, would silently
stop working: approval refuses it (``credential_without_record``) unless the
invocation names it in ``accept_loss``. Values never leave the containers:
the snapshot and receipt carry names, booleans and closed codes only.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import ExitStack
from urllib.parse import quote

if __package__:
    from .app_config_reconcile import (ACTION as APP_CONFIG_ACTION, AppConfigReconciler, valid_app_binding,
                                       validate_app_spec)
    from .image_deploy import DeployFailed, StepError, safe_error
    from .probe_contract import DESTINATION_PROBES, PROBE_IDS, RECORDS_PROBE, parse_line, validate
    from .tagged_config_reconcile import (ACTION as TAGGED_CONFIG_ACTION, TaggedConfigReconciler,
                                          valid_tagged_binding, validate_tagged_spec)
else:  # standalone immutable worker
    from app_config_reconcile import (ACTION as APP_CONFIG_ACTION, AppConfigReconciler, valid_app_binding,
                                      validate_app_spec)
    from image_deploy import DeployFailed, StepError, safe_error
    from probe_contract import DESTINATION_PROBES, PROBE_IDS, RECORDS_PROBE, parse_line, validate
    from tagged_config_reconcile import (ACTION as TAGGED_CONFIG_ACTION, TaggedConfigReconciler,
                                         valid_tagged_binding, validate_tagged_spec)

ACTION = "reconcile_service_plan"
SCHEMA = "llm-bawt.ops.service-plan-reconcile/v1"
# Worker image label proving it carries this module and the probe contract.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.service-plan-reconcile"
MAX_TARGETS = 6
MAX_PROBES = 16
SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

#: Target kinds a plan may order: ``(reconciler, snapshot validator, spec validator)``.
KINDS = {
    APP_CONFIG_ACTION: (AppConfigReconciler, valid_app_binding, validate_app_spec),
    TAGGED_CONFIG_ACTION: (TaggedConfigReconciler, valid_tagged_binding, validate_tagged_spec),
}

# The probe modules (TASK-1476 §3). The claude-code bridge serves ``broker``;
# the app serves every other id.
APP_PROBE_MODULE = "llm_bawt.service.cutover_probe"
BRIDGE_PROBE_MODULE = "claude_code_bridge.cutover_probe"
BRIDGE_PROBES = frozenset({"broker"})
#: Ids a plan spec may list. ``records`` and the destination probes are the
#: plan's own: it runs them from the approved credential classes.
SPEC_PROBES = PROBE_IDS - {RECORDS_PROBE} - set(DESTINATION_PROBES.values())

MIGRATED = "migrated"
UNCONFIGURED = "unconfigured"
ENV_KEPT = "env_kept"
ACCEPTED_LOSS = "accepted_loss"
CREDENTIAL_CLASSES = frozenset({MIGRATED, UNCONFIGURED, ENV_KEPT, ACCEPTED_LOSS})
DESTINATIONS = tuple(sorted(DESTINATION_PROBES))

_PLAN_KEYS = frozenset({"targets", "records", "credential_names", "env_now", "env_next", "credentials",
                        "accept_loss", "probes"})
_TARGET_KEYS = frozenset({"slug", "operation", "spec", "config", "newly_trusted"})
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_VOLUME_LABEL = "com.docker.compose.volume"
CREATED_BY_LABEL = "llm-bawt.ops.created-by"


class CredentialLoss(ValueError):
    """A credential the running app has would stop working: refused unless accepted."""

    code = "credential_without_record"

    def __init__(self, destination: str, names):
        super().__init__(f"credential_without_record:{destination}: the declared app drops "
                         f"{', '.join(sorted(names))} and no connected record replaces them; run the "
                         "env import for it first, or name it in accept_loss")


def probe_argv(probe_id: str) -> list[str]:
    """The fixed argv of a probe, chosen by id alone."""
    module = BRIDGE_PROBE_MODULE if probe_id in BRIDGE_PROBES else APP_PROBE_MODULE
    return ["python", "-m", module, probe_id]


def read_probe(probe_id: str, stdout: str) -> dict:
    """The schema-exact line for ``probe_id``; ValueError otherwise."""
    line = parse_line(stdout)
    if line["probe"] != probe_id:
        raise ValueError("probe line answers another probe")
    return line


def validate_plan_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"action", "targets", "probes", "records_target"}:
        raise ValueError("service plan spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not a service plan spec")
    targets = spec["targets"]
    if (not isinstance(targets, list) or not 1 <= len(targets) <= MAX_TARGETS
            or not all(isinstance(t, str) and SLUG.fullmatch(t) for t in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError(f"service plan targets must be 1..{MAX_TARGETS} distinct operation slugs")
    if spec["records_target"] not in targets:
        raise ValueError("service plan records_target must be one of its targets")
    probes = spec["probes"]
    if not isinstance(probes, list) or len(probes) > MAX_PROBES:
        raise ValueError(f"service plan probes must be a list of at most {MAX_PROBES}")
    seen = set()
    for probe in probes:
        if (not isinstance(probe, dict) or set(probe) != {"id", "target"} or probe["id"] not in SPEC_PROBES
                or probe["target"] not in targets):
            raise ValueError("service plan probe must be {id, target} with a closed id and a plan target")
        key = (probe["id"], probe["target"])
        if key in seen:
            raise ValueError("service plan probes must be distinct")
        seen.add(key)
    return spec


def env_names(env: dict, credential_names: dict) -> dict:
    """``{destination: sorted names set with a nonempty value}``. Names only."""
    return {dest: sorted(n for n in names if env.get(n)) for dest, names in credential_names.items()}


def container_env(container: dict) -> dict:
    """``Config.Env`` of an inspect as a dict; the values stay in this process."""
    pairs = (item.partition("=") for item in (container.get("Config") or {}).get("Env") or [])
    return {name: value for name, _, value in pairs}


def classify(records: dict, env_now: dict, env_next: dict, accept_loss) -> dict:
    """``{destination: class}`` (§3a); raises :class:`CredentialLoss` or ValueError."""
    classes = {}
    for dest in DESTINATIONS:
        dropped = set(env_now[dest]) - set(env_next[dest])
        if records[dest]["connected"]:
            classes[dest] = MIGRATED
        elif dropped:
            if dest not in accept_loss:
                raise CredentialLoss(dest, dropped)
            classes[dest] = ACCEPTED_LOSS
        elif env_next[dest]:
            classes[dest] = ENV_KEPT
        else:
            classes[dest] = UNCONFIGURED
    stray = sorted(set(accept_loss) - {d for d, c in classes.items() if c == ACCEPTED_LOSS})
    if stray:
        raise ValueError(f"accept_loss names destinations that lose nothing: {', '.join(stray)}")
    return classes


def plan_probes(spec: dict, credentials: dict) -> list[dict]:
    """The spec's probes, then each migrated destination's own, in the records target."""
    return [*spec["probes"], *({"id": DESTINATION_PROBES[d], "target": spec["records_target"]}
                               for d in DESTINATIONS if credentials[d] == MIGRATED)]


def target_snapshot(target: dict) -> dict:
    """The single-target snapshot a target's own reconciler validates and executes."""
    return {"operation": target["operation"], "spec": target["spec"], "resolved_args": {},
            "config": target["config"], "newly_trusted": target["newly_trusted"]}


def _service(dependency: str) -> str:
    return dependency.rpartition(":")[0]


def _names_map(value, names: dict) -> bool:
    return (isinstance(value, dict) and set(value) == set(DESTINATIONS)
            and all(isinstance(value[d], list) and value[d] == sorted(set(value[d]))
                    and set(value[d]) <= set(names[d]) for d in DESTINATIONS))


def valid_plan_snapshot(snapshot: dict) -> bool:
    """Structure, every target's own binding, order and the credential classes."""
    plan = snapshot.get("plan")
    if snapshot["spec"].get("action") != ACTION:
        return plan is None
    try:
        spec = validate_plan_spec(snapshot["spec"])
        if not isinstance(plan, dict) or set(plan) != _PLAN_KEYS:
            return False
        accept = plan["accept_loss"]
        if (not isinstance(accept, list) or accept != sorted(set(accept)) or not set(accept) <= set(DESTINATIONS)
                or not set(snapshot["resolved_args"]) <= {"accept_loss"}
                or sorted(set(snapshot["resolved_args"].get("accept_loss", []))) != accept):
            return False
        targets = plan["targets"]
        if not isinstance(targets, list) or [t.get("slug") for t in targets] != spec["targets"]:
            return False
        services = []
        for target in targets:
            if (set(target) != _TARGET_KEYS or target["spec"].get("action") not in KINDS
                    or not isinstance(target["operation"], dict) or target["operation"].get("slug") != target["slug"]):
                return False
            if target["spec"]["action"] != APP_CONFIG_ACTION and target["newly_trusted"] is not None:
                return False
            if not KINDS[target["spec"]["action"]][1](target_snapshot(target)):
                return False
            services.append(target["spec"]["compose_service"])
        if (len({t["spec"]["container_name"] for t in targets}) != len(targets)
                or len({t["spec"]["compose_project"] for t in targets}) != 1 or len(set(services)) != len(services)):
            return False
        # A target may depend only on earlier targets or on services outside the plan.
        for index, target in enumerate(targets):
            if any(_service(dep) in services[index:] for dep in target["config"]["depends_on"]):
                return False
        records_target = targets[spec["targets"].index(spec["records_target"])]
        if records_target["spec"]["action"] != APP_CONFIG_ACTION:
            return False
        validate({"probe": RECORDS_PROBE, "ok": True, "status": None, "code": "ok", "records": plan["records"]})
        names = plan["credential_names"]
        if (not isinstance(names, dict) or set(names) != set(DESTINATIONS)
                or not all(isinstance(names[d], list) and names[d] and names[d] == sorted(set(names[d]))
                           and all(isinstance(n, str) and ENV_NAME.fullmatch(n) for n in names[d])
                           for d in DESTINATIONS)):
            return False
        if not _names_map(plan["env_now"], names) or not _names_map(plan["env_next"], names):
            return False
        if plan["credentials"] != classify(plan["records"], plan["env_now"], plan["env_next"], accept):
            return False
        return plan["probes"] == plan_probes(spec, plan["credentials"])
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


class _TargetFailed(Exception):
    """A target's own reconciler refused or failed (its DeployFailed inside)."""

    def __init__(self, index: int, failure: DeployFailed):
        super().__init__(str(failure))
        self.index = index
        self.failure = failure


class _CheckFailed(StepError):
    """A plan-level check failed after production was touched: revert all."""


class PlanReconciler:
    """Run an approved service plan. ``build(cls, api, snapshot, job_id, **kw)``
    constructs each target's reconciler (injectable in tests)."""

    ACTION_NAME = "reconcile-service-plan"
    _STEP_KEYS = ("phase", "config", "health", "unchanged", "previous", "deployed", "restored", "failed_phase",
                  "warnings", "error")

    def __init__(self, api, snapshot: dict, job_id: str, *, lock_root, sleep=time.sleep, clock=time.monotonic,
                 build=None):
        self.api = api
        self.snapshot = snapshot
        self.spec = snapshot["spec"]
        self.plan = snapshot["plan"]
        self.job_id = job_id
        build = build or (lambda cls, *a, **kw: cls(*a, **kw))
        self.steps = [build(KINDS[t["spec"]["action"]][0], api, target_snapshot(t), job_id,
                            lock_root=lock_root, sleep=sleep, clock=clock) for t in self.plan["targets"]]
        self.slugs = [t["slug"] for t in self.plan["targets"]]
        self.records_index = self.slugs.index(self.spec["records_target"])
        self.status = ["untouched"] * len(self.steps)
        self.current = [None] * len(self.steps)  # container id serving each target now
        self.swaps = []  # (index, Swap) in plan order
        self.created_volumes = []
        self.detail = {"schema": SCHEMA, "job_id": job_id, "action": self.ACTION_NAME, "phase": "init",
                       "targets": [], "credentials": self.plan["credentials"], "probes": [],
                       "volumes_created": [], "warnings": [], "error": None}

    # ── entry ────────────────────────────────────────────────────────────
    def run(self) -> str:
        try:
            with ExitStack() as locks:
                for index, step in enumerate(self.steps):
                    self._phase(f"lock:{self.slugs[index]}")
                    self._guard(index, locks.enter_context, step._target_lock())
                self._perform()
        except (DeployFailed, TimeoutError):
            raise
        except _TargetFailed as exc:  # before any write: the target refused, nothing changed
            self._fail(f"{self.slugs[exc.index]}: {exc}", known=True)
        except Exception as exc:  # noqa: BLE001 - never let external text reach a receipt
            self._fail(f"{self.detail['phase']} failed ({safe_error(exc)})",
                       known=not getattr(self.api, "side_effect_started", False))
        self._phase("done")
        return json.dumps(self._receipt(), sort_keys=True)

    def _perform(self):
        olds = self._preflight()
        missing = self._check_volumes()
        try:
            self._create_volumes(missing)
            for index, step in enumerate(self.steps):
                self._advance(index, step, olds[index])
            self._end_checks()
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure after the first write reverts
            self._unwind(exc)
        self._phase("cleanup")
        for _index, swap in self.swaps:
            self.steps[_index]._commit_swap(swap)

    # ── phases ───────────────────────────────────────────────────────────
    def _preflight(self):
        """Read-only: bind and check every target before touching any."""
        olds = []
        for index, step in enumerate(self.steps):
            self._phase(f"preflight:{self.slugs[index]}")
            old, _image = self._guard(index, step._prepare)
            self._guard(index, step._before_swap)
            olds.append(old)
            self.current[index] = old["Id"]
        self._phase("preflight:credentials")
        self._guard(self.records_index, self._check_credentials, self.steps[self.records_index],
                    olds[self.records_index])
        return olds

    def _advance(self, index, step, old):
        """Target ``index``'s last-read recheck, then verify-only or swap."""
        slug = self.slugs[index]
        self._phase(f"recheck:{slug}")
        # The stopped -prev containers of swapped targets are not dependencies.
        step.ignored_ids = frozenset(swap.old_id for _i, swap in self.swaps)
        old, image = self._guard(index, step._prepare)
        if index == self.records_index:
            self._guard(index, self._check_credentials, step, old)
        self._guard(index, step._before_swap)
        self._phase(f"apply:{slug}")
        if self._guard(index, step._verify_only, old):
            self.status[index] = "unchanged"
            return
        binding = step.snapshot["config"]
        swap = self._guard(index, step._swap, old, image, image, step._swap_ref(binding), step._release(binding))
        self.swaps.append((index, swap))
        self.status[index] = "swapped"
        self.current[index] = swap.new_id

    def _end_checks(self):
        step = self.steps[self.records_index]
        app_id = self.current[self.records_index]
        self._phase("verify:records")
        if self._records(step, app_id) != self.plan["records"]:
            raise _CheckFailed("records_changed: the recreated app reports different stored records")
        self._phase("verify:credential-env")
        new = self.api.request("GET", f"/containers/{quote(app_id, safe='')}/json")
        if env_names(container_env(new), self.plan["credential_names"]) != self.plan["env_next"]:
            raise _CheckFailed("credential_env_mismatch: the recreated app's credential env names differ")
        self._phase("verify:probes")
        failed = []
        for probe in self.plan["probes"]:
            index = self.slugs.index(probe["target"])
            receipt = {"id": probe["id"], "target": probe["target"]}
            try:
                line = self._probe(self.steps[index], self.current[index], probe["id"])
                receipt.update(ok=line["ok"], status=line["status"], code=line["code"])
            except ValueError:
                receipt.update(ok=False, status=None, code="probe_unverifiable")
            self.detail["probes"].append(receipt)
            if not receipt["ok"]:
                failed.append(f"{probe['id']}@{probe['target']}:{receipt['code']}")
        if failed:
            raise _CheckFailed(f"probe_failed: {', '.join(failed)}")

    # ── credentials and probes ───────────────────────────────────────────
    def _check_credentials(self, step, old):
        """The approved classes from fresh facts: records now, the app's env
        names now and as declared. Any difference refuses."""
        names = self.plan["credential_names"]
        if self._records(step, old["Id"]) != self.plan["records"]:
            step._fail("records_changed_since_approval; nothing changed", known=True)
        now, desired = env_names(container_env(old), names), env_names(step._desired["env"], names)
        if now != self.plan["env_now"] or desired != self.plan["env_next"]:
            step._fail("credential_env_changed_since_approval; nothing changed", known=True)
        try:
            classes = classify(self.plan["records"], now, desired, self.plan["accept_loss"])
        except ValueError as exc:
            step._fail(f"{exc}; nothing changed", known=True)
        if classes != self.plan["credentials"]:
            step._fail("credential_classes_changed_since_approval; nothing changed", known=True)

    def _records(self, step, container_id):
        try:
            line = self._probe(step, container_id, RECORDS_PROBE)
        except ValueError:
            raise StepError("probe_unverifiable: records") from None
        if not line["ok"]:
            raise StepError(f"records probe failed: {line['code']}")
        return line["records"]

    def _probe(self, step, container_id, probe_id):
        """The probe line; ValueError when it is not exactly the schema."""
        _code, stdout = step._exec(container_id, probe_argv(probe_id))
        return read_probe(probe_id, stdout)

    # ── volumes ──────────────────────────────────────────────────────────
    def _check_volumes(self):
        """Read-only: the project volumes the targets mount that are missing.
        A volume another project or key owns refuses, nothing changed."""
        self._phase("preflight:volumes")
        project = self.steps[0].spec["compose_project"]
        prefix = f"{project}_"
        names = sorted({mount["source"] for step in self.steps for mount in step._desired["mounts"].values()
                        if mount["type"] == "volume"})
        missing = []
        for name in names:
            volume = self._volume(name)
            if volume is None:
                if not name.startswith(prefix) or not name[len(prefix):]:
                    self._fail(f"volume_missing: {name} is not this project's to create; nothing changed",
                               known=True)
                missing.append(name)
                continue
            labels = volume.get("Labels") or {}
            owner, key = labels.get(COMPOSE_PROJECT_LABEL), labels.get(COMPOSE_VOLUME_LABEL)
            if (owner is not None and owner != project) or (
                    key is not None and name.startswith(prefix) and key != name[len(prefix):]):
                self._fail(f"volume_foreign: {name} belongs to another Compose project or volume; "
                           "nothing changed", known=True)
        return missing

    def _volume(self, name):
        try:
            return self.api.request("GET", f"/volumes/{quote(name, safe='')}")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def _create_volumes(self, names):
        project = self.steps[0].spec["compose_project"]
        for name in names:
            self._phase("volumes")
            self.api.side_effect_started = True
            self.api.request("POST", "/volumes/create", body={
                "Name": name, "Labels": {COMPOSE_PROJECT_LABEL: project,
                                         COMPOSE_VOLUME_LABEL: name[len(project) + 1:],
                                         CREATED_BY_LABEL: self.job_id}})
            self.created_volumes.append(name)
            self.detail["volumes_created"].append(name)

    # ── failure ──────────────────────────────────────────────────────────
    def _unwind(self, exc):
        """Revert every swapped target in reverse; then fail with the plan record."""
        known = True
        if isinstance(exc, _TargetFailed):
            index, failure = exc.index, exc.failure
            step = self.steps[index]
            if failure.outcome_known:
                self.status[index] = "restored" if step.detail.get("restored") else "untouched"
            else:
                self.status[index] = "uncertain"
                known = False
            reason = f"{self.slugs[index]}: {failure}"
        else:
            reason = f"{self.detail['phase']} failed ({safe_error(exc)})"
        self._phase("revert")
        for index, swap in reversed(self.swaps):
            step = self.steps[index]
            try:
                step._revert(swap)
                self.status[index] = "reverted"
                self.current[index] = swap.old_id
            except Exception as revert_exc:  # noqa: BLE001 - keep going: every target is tried
                self.status[index] = "uncertain"
                known = False
                self.detail["warnings"].append(f"{self.slugs[index]} revert failed: {safe_error(revert_exc)}")
        self.swaps = []
        if known:
            for name in self.created_volumes:
                try:
                    self.api.request("DELETE", f"/volumes/{quote(name, safe='')}")
                    self.detail.setdefault("volumes_removed", []).append(name)
                except Exception as volume_exc:  # noqa: BLE001 - an unused empty volume is harmless
                    self.detail["warnings"].append(f"volume_left: {name} ({safe_error(volume_exc)})")
        outcome = "every target reverted" if known else "REVERT INCOMPLETE; production state unknown"
        self._fail(f"{reason}; {outcome}", known=known)

    def _guard(self, index, fn, *args):
        try:
            return fn(*args)
        except DeployFailed as exc:
            raise _TargetFailed(index, exc) from None

    def _fail(self, message: str, *, known: bool):
        raise DeployFailed(message, self._receipt(), outcome_known=known)

    def _phase(self, phase: str):
        self.detail["phase"] = phase

    def _receipt(self) -> dict:
        self.detail["targets"] = [
            {"slug": slug, "target": step.name, "status": status, "container_id": current,
             **{k: step.detail[k] for k in self._STEP_KEYS if step.detail.get(k) not in (None, [], False)}}
            for slug, step, status, current in zip(self.slugs, self.steps, self.status, self.current)]
        return self.detail


__all__ = ["ACCEPTED_LOSS", "ACTION", "APP_PROBE_MODULE", "BRIDGE_PROBE_MODULE", "CREDENTIAL_CLASSES",
           "CredentialLoss", "DESTINATIONS", "ENV_KEPT", "KINDS", "MIGRATED", "PlanReconciler", "SCHEMA",
           "SPEC_PROBES", "UNCONFIGURED", "WORKER_CAPABILITY_LABEL", "classify", "container_env", "env_names",
           "plan_probes", "probe_argv", "read_probe", "target_snapshot", "valid_plan_snapshot",
           "validate_plan_spec"]
