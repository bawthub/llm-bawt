"""Same-image frontend routing repair. STANDARD LIBRARY ONLY (TASK-1179).

No Compose/shell runner and no arbitrary Config/HostConfig patch. A reviewed
catalog spec may append the health prefix to an existing host-scoped API rule.
The approval stores only public routing metadata and a hash of private config.
"""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import quote

if __package__:
    from .health_probe import validate_health_probe
    from .image_deploy import ImageDeployer, clone_create_body, demux_stdout, release_from_labels
    from .service_identity import SPEC_PATTERNS, ComposeServiceRef
else:  # standalone immutable worker
    from health_probe import validate_health_probe
    from image_deploy import ImageDeployer, clone_create_body, demux_stdout, release_from_labels
    from service_identity import SPEC_PATTERNS, ComposeServiceRef

ACTION = "reconcile_frontend_routing"
SCHEMA = "llm-bawt.ops.routing-reconcile/v1"
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.routing-reconcile"
_HOST = r"Host\(`[a-zA-Z0-9.-]+`\)"
_PATH = r"PathPrefix\(`/api/[a-zA-Z0-9_-]+`\)"
_RULE = re.compile(rf"\((?P<hosts>{_HOST}(?: \|\| {_HOST})*)\) && \((?P<paths>{_PATH}(?: \|\| {_PATH})*)\)")


def validate_routing_spec(spec: dict) -> dict:
    patterns = {
        "container_name": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",
        **SPEC_PATTERNS,
        "image_repository": r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*",
        "router_name": r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}",
        "route_service": r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}",
        "hostname": r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}",
    }
    required = {"action", "add_prefixes", "stop_grace_seconds", "health_timeout_seconds", *patterns}
    # TASK-1801: the row's own post-swap probe (optional until the app requires it).
    if not required <= set(spec) <= required | {"health_probe"}:
        raise ValueError("routing spec has missing or unsupported fields")
    if "health_probe" in spec:
        validate_health_probe(spec["health_probe"])
    if spec["action"] != ACTION or spec["add_prefixes"] != ["/api/system"]:
        raise ValueError("routing repair supports only the system-health prefix")
    for key, pattern in patterns.items():
        if not isinstance(spec[key], str) or not re.fullmatch(pattern, spec[key]):
            raise ValueError(f"routing spec {key} is invalid")
    for key, low, high in (("stop_grace_seconds", 0, 60), ("health_timeout_seconds", 10, 180)):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError(f"routing spec {key} is out of bounds")
    return spec


def patched_rule(spec: dict, before: str) -> str:
    match = _RULE.fullmatch(before) if isinstance(before, str) else None
    if not match or f"Host(`{spec['hostname']}`)" not in match["hosts"].split(" || "):
        raise ValueError("existing routing rule is not the expected host-scoped API allowlist")
    paths = match["paths"].split(" || ")
    for prefix in spec["add_prefixes"]:
        term = f"PathPrefix(`{prefix}`)"
        if term not in paths:
            paths.append(term)
    return f"({match['hosts']}) && ({' || '.join(paths)})"


def routing_binding(spec: dict, container: dict, image: dict) -> dict:
    """Validate actual target and bind its effective runtime configuration."""
    validate_routing_spec(spec)
    config = container["Config"]
    labels = config.get("Labels") or {}
    if (container["Name"] != "/" + spec["container_name"]
            or not ComposeServiceRef.from_spec(spec).matches(labels)):
        raise ValueError("routing target Compose identity mismatch")
    ref = config.get("Image", "")
    if (not re.fullmatch(re.escape(spec["image_repository"]) + r"@sha256:[0-9a-f]{64}", ref)
            or image["Id"] != container["Image"] or ref not in (image.get("RepoDigests") or [])):
        raise ValueError("routing target must retain its locally available pinned image")
    release = release_from_labels(image["Config"].get("Labels"))
    if not release or not all(release.values()):
        raise ValueError("routing target has no verifiable release identity")
    router = f"traefik.http.routers.{spec['router_name']}"
    service = f"traefik.http.services.{spec['route_service']}"
    if (labels.get(router + ".service") != spec["route_service"]
            or labels.get(router + ".entrypoints") != "web"
            or labels.get(router + ".priority") != "200"
            or labels.get(service + ".loadbalancer.server.port") != "3000"
            or labels.get("traefik.enable") != "true"):
        raise ValueError("routing target is not the expected frontend API router")
    before = labels.get(router + ".rule", "")
    # Include full Config + HostConfig, plus stable network creation settings.
    # Never put environment values/host mount paths in the approval or receipt.
    body = clone_create_body(container, image, image, ref, "binding")
    private = {"config": config, "host": container["HostConfig"], "networks": body["NetworkingConfig"]}
    digest = hashlib.sha256(json.dumps(private, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
            "config_hash": digest, "before_rule": before, "after_rule": patched_rule(spec, before),
            "release": release}


def valid_routing_binding(snapshot: dict) -> bool:
    binding = snapshot.get("routing")
    if snapshot["spec"].get("action") != ACTION:
        return binding is None
    if snapshot["resolved_args"] != {} or not isinstance(binding, dict):
        return False
    if set(binding) != {"container_id", "image_id", "image_ref", "config_hash", "before_rule", "after_rule", "release"}:
        return False
    try:
        spec = validate_routing_spec(snapshot["spec"])
        return (bool(re.fullmatch(r"[0-9a-f]{64}", binding["container_id"]))
                and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_id"]))
                and bool(re.fullmatch(r"[0-9a-f]{64}", binding["config_hash"]))
                and bool(re.fullmatch(re.escape(spec["image_repository"]) + r"@sha256:[0-9a-f]{64}", binding["image_ref"]))
                and isinstance(binding["release"], dict)
                and set(binding["release"]) == {"version", "source_sha", "workflow_run_id"}
                and all(isinstance(v, str) and v for v in binding["release"].values())
                and binding["after_rule"] == patched_rule(spec, binding["before_rule"]))
    except (TypeError, ValueError, KeyError):
        return False


class RoutingReconciler(ImageDeployer):
    """Use the existing target lock, swap, health and failure-restore machinery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detail.update(schema=SCHEMA, action="reconcile-routing")

    def _perform(self):
        if not valid_routing_binding(self.snapshot):
            self._fail("invalid routing approval binding", known=True)
        binding = self.snapshot["routing"]
        old, image = self._target(expected_image_id=binding["image_id"])
        try:
            actual = routing_binding(self.spec, old, image)
        except ValueError as exc:
            self._fail(str(exc), known=True)
        if actual != binding:
            self._fail("target container/configuration changed since approval; nothing changed", known=True)
        state = old.get("State") or {}
        if not state.get("Running") or (state.get("Health") or {}).get("Status") != "healthy":
            self._fail("routing repair requires a running healthy frontend", known=True)
        self.detail["routing"] = {k: binding[k] for k in ("before_rule", "after_rule", "config_hash")}
        self._approved_image_id = binding["image_id"]
        if binding["before_rule"] == binding["after_rule"]:
            self.detail["health"] = self._verify_release(old["Id"], binding["release"])
            self.detail["unchanged"] = True
            return
        self._replace(old, image, image, binding["image_ref"], binding["release"])

    def _create_body(self, old, old_image, new_image, ref):
        body = super()._create_body(old, old_image, new_image, ref)
        body["Labels"][f"traefik.http.routers.{self.spec['router_name']}.rule"] = self.snapshot["routing"]["after_rule"]
        return body

    def _verify_release(self, container_id, expect):
        health = super()._verify_release(container_id, expect)
        # The worker remains network=none. Execute a fixed curl argv in the
        # frontend to exercise Traefik's hostname routing, not just localhost.
        deadline = self.clock() + self.spec["health_timeout_seconds"]
        while True:
            try:
                report = self._routed_report(container_id)
                health["routing"] = {"host": self.spec["hostname"], "path": "/api/system/health",
                                     "status": report["status"], "summary": report["summary"]}
                return health
            except (RuntimeError, ValueError, KeyError, TypeError):
                if self.clock() >= deadline:
                    raise RuntimeError("production hostname did not serve a valid detailed health report") from None
                self.sleep(2)

    def _routed_report(self, container_id):
        exec_id = self.api.request("POST", f"/containers/{quote(container_id, safe='')}/exec", body={
            "AttachStdout": True, "AttachStderr": True,
            "Cmd": ["curl", "-fsS", "--max-time", "5", "-H", f"Host: {self.spec['hostname']}",
                    "http://traefik/api/system/health"]})["Id"]
        raw = self.api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
        if self.api.request("GET", f"/exec/{exec_id}/json").get("ExitCode") != 0:
            raise RuntimeError("routed health request failed")
        report = json.loads(demux_stdout(raw))
        if not isinstance(report, dict) or report.get("status") not in {"ok", "degraded", "down"}:
            raise ValueError("invalid routed health report")
        checks, summary = report.get("checks"), report.get("summary")
        if (not isinstance(checks, list) or not checks or not isinstance(summary, dict)
                or any(not isinstance(c, dict) or c.get("status") not in {"ok", "degraded", "down"}
                       or not isinstance(c.get("slug"), str) for c in checks)
                or any(type(summary.get(s)) is not int or summary[s] != sum(c["status"] == s for c in checks)
                       for s in ("ok", "degraded", "down"))):
            raise ValueError("routed health report lacks individual results")
        return report
