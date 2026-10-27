"""Compose-declared vs running container configuration. STANDARD LIBRARY ONLY (TASK-1518).

Compose is the declared truth of a Compose-managed target; a deploy that clones
the running container forward silently keeps whatever configuration the target
was last created with. This module:

* translates the rendered service (``docker compose config --format json``) for
  an ALLOWLISTED subset of keys. Any other key refuses, so an operation can never
  apply a privilege or setting it does not model;
* projects both sides onto the compared fields (environment, labels, mounts,
  networks, extra hosts, restart policy), normalising image-supplied env/labels
  and Compose/ops bookkeeping labels away on BOTH sides;
* reports drift as NAMES only (added/removed/changed), never values;
* overlays the desired fields onto a cloned create body.

Fields outside the allowlist (privileges, devices, user, …) are carried from the
running container exactly as a plain image deploy would.

The allowlist is a frozen per-kind :class:`ServiceProfile` (TASK-1474). The
pinned frontend models no process or port settings. A Compose-BUILT service
(the llm-bawt app) additionally models command/entrypoint, published ports,
stdin_open/tty, container_name and depends_on; its ``build`` key is accepted
but inert, because a reconcile never builds.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

if __package__:
    from .compose_render import ComposeError, compose_source
else:  # standalone immutable worker
    from compose_render import ComposeError, compose_source

BOOKKEEPING_PREFIXES = ("com.docker.compose.", "llm-bawt.ops.")
CONFIG_HASH_LABEL = "com.docker.compose.config-hash"
SERVICE_KEYS = frozenset({"command", "entrypoint", "environment", "extra_hosts", "image", "labels",
                          "networks", "pull_policy", "restart", "volumes"})
# The worker never pulls; the pinned image must already be on the daemon.
PULL_POLICIES = frozenset({"missing", "never"})
_VOLUME_KEYS = frozenset({"type", "source", "target", "read_only", "bind", "volume"})
_BIND_KEYS = frozenset({"create_host_path"})
_RESTART = re.compile(r"(no|always|unless-stopped|on-failure)(?::([0-9]{1,3}))?")
_PORT_KEYS = frozenset({"target", "published", "protocol", "host_ip", "mode"})
_PORT_PROTOCOLS = frozenset({"tcp", "udp"})
_HOST_IP = re.compile(r"[0-9A-Fa-f.:]{0,45}")
_SERVICE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}")
_DEPENDS_KEYS = frozenset({"condition", "required", "restart"})
DEPENDENCY_CONDITIONS = frozenset({"service_started", "service_healthy"})
# Compared only by a profile that models the process (``models_process``).
_PROCESS_FIELDS = (("cmd", "command"), ("entrypoint", "entrypoint"), ("stdin_open", "stdin_open"),
                   ("tty", "tty"))
UNEXPECTED_VALUE = "<unexpected value>"
UNSET_VALUE = "<unset>"


@dataclass(frozen=True)
class ServiceProfile:
    """What one target kind may declare. ``built``: Compose builds the image, so
    ``build`` is accepted and inert and ``image`` must be absent; otherwise the
    service interpolates the pinned ``image``. ``models_process``: command,
    entrypoint, ports, stdin_open/tty, container_name and depends_on."""

    name: str
    keys: frozenset
    built: bool
    models_process: bool


PINNED_PROFILE = ServiceProfile("pinned", SERVICE_KEYS, built=False, models_process=False)
BUILT_PROFILE = ServiceProfile("built", frozenset({
    "build", "command", "container_name", "depends_on", "entrypoint", "environment", "extra_hosts",
    "labels", "networks", "ports", "restart", "stdin_open", "tty", "volumes"}), built=True, models_process=True)


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _bookkeeping(key: str) -> bool:
    return key.startswith(BOOKKEEPING_PREFIXES)


def _host_entry(raw: str) -> str:
    """Compose renders ``host=ip`` (newer) or ``host:ip``; Docker stores ``host:ip``."""
    host, sep, address = raw.partition("=") if "=" in raw else raw.partition(":")
    if not sep or not host or not address:
        raise ComposeError("unsupported_compose_key", ["extra_hosts"])
    return f"{host}:{address}"


def _strings(value, key: str) -> dict:
    if isinstance(value, list):
        pairs = {}
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                raise ComposeError("unsupported_compose_key", [key])
            name, _, text = item.partition("=")
            pairs[name] = text
        value = pairs
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and k for k in value):
        raise ComposeError("unsupported_compose_key", [key])
    unset = [k for k, v in value.items() if v is None]
    if unset:
        raise ComposeError("unset_compose_variable", unset)
    if not all(isinstance(v, str) for v in value.values()):
        raise ComposeError("unsupported_compose_key", [key])
    return dict(value)


def _argv(value, key: str):
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ComposeError("unsupported_compose_key", [key])
    return list(value)


def _flag(value, key: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ComposeError("unsupported_compose_key", [key])
    return value


def _ports(value) -> dict:
    """Long-form ``ports`` → Docker ``PortBindings`` (single published ports only)."""
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ComposeError("unsupported_compose_key", ["ports"])
    bindings: dict[str, list] = {}
    for port in value:
        if not isinstance(port, dict) or set(port) - _PORT_KEYS:
            raise ComposeError("unsupported_compose_key", ["ports"])
        if port.get("mode") != "ingress":
            raise ComposeError("unsupported_compose_key", ["ports.mode"])
        target, published = port.get("target"), port.get("published")
        if type(target) is not int or not 1 <= target <= 65535:
            raise ComposeError("unsupported_compose_key", ["ports.target"])
        published = str(published) if type(published) is int else published
        if not isinstance(published, str) or not re.fullmatch(r"[1-9][0-9]{0,4}", published) or int(published) > 65535:
            raise ComposeError("unsupported_compose_key", ["ports.published"])
        protocol = port.get("protocol", "tcp")
        if protocol not in _PORT_PROTOCOLS:
            raise ComposeError("unsupported_compose_key", ["ports.protocol"])
        host_ip = port.get("host_ip", "")
        if not isinstance(host_ip, str) or not _HOST_IP.fullmatch(host_ip):
            raise ComposeError("unsupported_compose_key", ["ports.host_ip"])
        entry = {"HostIp": host_ip, "HostPort": published}
        key = f"{target}/{protocol}"
        if entry in bindings.get(key, []):
            raise ComposeError("unsupported_compose_key", ["ports"])
        bindings.setdefault(key, []).append(entry)
    return {k: sorted(v, key=lambda b: (b["HostIp"], b["HostPort"])) for k, v in sorted(bindings.items())}


def _depends_on(value) -> dict:
    """``{service: condition}``. A precondition checked before the swap, never
    drift: a dependency is never started, restarted or recreated."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ComposeError("unsupported_compose_key", ["depends_on"])
    out = {}
    for name, dep in value.items():
        if (not isinstance(name, str) or not _SERVICE_NAME.fullmatch(name) or not isinstance(dep, dict)
                or set(dep) - _DEPENDS_KEYS or dep.get("condition") not in DEPENDENCY_CONDITIONS
                or dep.get("required", True) is not True or dep.get("restart", False) is not False):
            raise ComposeError("unsupported_compose_key", ["depends_on"])
        out[name] = dep["condition"]
    return dict(sorted(out.items()))


def _process(service: dict, spec: dict) -> dict:
    if service.get("entrypoint") is not None:
        raise ComposeError("unsupported_compose_key", ["entrypoint"])
    if service.get("container_name") != spec["container_name"]:
        raise ComposeError("compose_container_name_mismatch")
    return {"cmd": _argv(service.get("command"), "command"), "entrypoint": None,
            "ports": _ports(service.get("ports")),
            "stdin_open": _flag(service.get("stdin_open"), "stdin_open"),
            "tty": _flag(service.get("tty"), "tty"),
            "depends_on": _depends_on(service.get("depends_on"))}


def translate(rendered: dict, spec: dict, image_ref: str, profile: ServiceProfile = PINNED_PROFILE) -> dict:
    """The desired service, restricted to the profile's keys. Values included."""
    project = spec["compose_project"]
    service = (rendered.get("services") or {}).get(spec["compose_service"])
    if not isinstance(service, dict):
        raise ComposeError("compose_service_missing")
    unknown = sorted(set(service) - profile.keys)
    if unknown:
        raise ComposeError("unsupported_compose_key", unknown)
    if service.get("pull_policy", "missing") not in PULL_POLICIES:
        raise ComposeError("unsupported_pull_policy")
    if not profile.models_process:
        for key in ("command", "entrypoint"):
            if service.get(key) is not None:
                raise ComposeError("unsupported_compose_key", [key])
    if profile.built:
        # The running image ID is bound instead; ``build`` is never acted on.
        if not isinstance(service.get("build"), dict):
            raise ComposeError("compose_image_mismatch")
    elif service.get("image") != image_ref:
        raise ComposeError("compose_image_mismatch")
    top_volumes = rendered.get("volumes") or {}
    mounts = {}
    for volume in service.get("volumes") or []:
        if not isinstance(volume, dict) or set(volume) - _VOLUME_KEYS:
            raise ComposeError("unsupported_compose_key", ["volumes"])
        if set(volume.get("bind") or {}) - _BIND_KEYS or (volume.get("volume") or {}):
            raise ComposeError("unsupported_compose_key", ["volumes"])
        kind, source, target = volume.get("type"), volume.get("source"), volume.get("target")
        if not isinstance(target, str) or not target.startswith("/") or target in mounts:
            raise ComposeError("unsupported_compose_key", ["volumes.target"])
        if kind == "bind" and isinstance(source, str) and source.startswith("/"):
            pass
        elif kind == "volume" and isinstance(source, str) and source:
            declared = top_volumes.get(source)
            if not isinstance(declared, dict):
                raise ComposeError("unsupported_compose_key", ["volumes.source"])
            source = declared.get("name") or f"{project}_{source}"
        else:
            raise ComposeError("unsupported_compose_key", ["volumes.type"])
        mounts[target] = {"type": kind, "source": source, "read_only": bool(volume.get("read_only"))}
    top_networks = rendered.get("networks") or {}
    networks = []
    raw_networks = service.get("networks") or {"default": None}
    if not isinstance(raw_networks, dict):
        raise ComposeError("unsupported_compose_key", ["networks"])
    for key, settings in raw_networks.items():
        if settings not in (None, {}):
            raise ComposeError("unsupported_compose_key", [f"networks.{key}"])
        declared = top_networks.get(key) or {}
        if not isinstance(declared, dict):
            raise ComposeError("unsupported_compose_key", ["networks"])
        networks.append(declared.get("name") or f"{project}_{key}")
    hosts = service.get("extra_hosts") or []
    if isinstance(hosts, dict):
        hosts = [f"{host}={address}" for host, address in hosts.items()]
    if not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts):
        raise ComposeError("unsupported_compose_key", ["extra_hosts"])
    restart = _RESTART.fullmatch(service.get("restart") or "no")
    if not restart:
        raise ComposeError("unsupported_compose_key", ["restart"])
    desired = {"env": _strings(service.get("environment"), "environment"),
               "labels": _strings(service.get("labels"), "labels"),
               "mounts": mounts, "networks": sorted(set(networks)),
               "extra_hosts": sorted(_host_entry(h) for h in hosts),
               "restart": [restart.group(1), int(restart.group(2) or 0)]}
    if profile.models_process:
        desired.update(_process(service, spec))
    return desired


def _parse_bind(raw: str) -> tuple[str, dict]:
    parts = raw.split(":")
    if len(parts) < 2:
        raise ComposeError("running_config_unsupported", ["HostConfig.Binds"])
    source, target = parts[0], parts[1]
    options = parts[2].split(",") if len(parts) > 2 else []
    kind = "bind" if source.startswith("/") else "volume"
    return target, {"type": kind, "source": source, "read_only": "ro" in options}


def running(container: dict) -> dict:
    """The running container on the same fields as :func:`translate`."""
    config, host = container.get("Config") or {}, container.get("HostConfig") or {}
    env = {}
    for entry in config.get("Env") or []:
        name, _, value = entry.partition("=")
        env[name] = value
    mounts = {}
    for raw in host.get("Binds") or []:
        target, mount = _parse_bind(raw)
        mounts[target] = mount
    for mount in host.get("Mounts") or []:
        mounts[mount.get("Target")] = {"type": (mount.get("Type") or "").lower(),
                                       "source": mount.get("Source") or "",
                                       "read_only": bool(mount.get("ReadOnly"))}
    policy = host.get("RestartPolicy") or {}
    ports = {key: sorted(({"HostIp": b.get("HostIp") or "", "HostPort": b.get("HostPort") or ""}
                          for b in bindings), key=lambda b: (b["HostIp"], b["HostPort"]))
             for key, bindings in sorted((host.get("PortBindings") or {}).items()) if bindings}
    return {"env": env, "labels": dict(config.get("Labels") or {}), "mounts": mounts,
            "networks": sorted((container.get("NetworkSettings") or {}).get("Networks") or {}),
            "extra_hosts": sorted(host.get("ExtraHosts") or []),
            "restart": [policy.get("Name") or "no", int(policy.get("MaximumRetryCount") or 0)],
            "cmd": config.get("Cmd"), "entrypoint": config.get("Entrypoint"), "ports": ports,
            "stdin_open": bool(config.get("OpenStdin")), "tty": bool(config.get("Tty"))}


def normalized(state: dict, image: dict) -> dict:
    """Drop what the image supplies and Compose/ops bookkeeping (both sides).
    An absent command/entrypoint means the image's own."""
    image_config = image.get("Config") or {}
    image_env = dict(e.partition("=")[::2] for e in image_config.get("Env") or [])
    image_labels = image_config.get("Labels") or {}
    result = {**state,
              "env": {k: v for k, v in state["env"].items() if image_env.get(k) != v},
              "labels": {k: v for k, v in state["labels"].items()
                         if not _bookkeeping(k) and image_labels.get(k) != v}}
    for field, key in (("cmd", "Cmd"), ("entrypoint", "Entrypoint")):
        if field in result and result[field] is None:
            result[field] = image_config.get(key)
    return result


def drift(desired: dict, actual: dict) -> list[str]:
    """NAMES of differing fields (``env:+KEY``, ``label:~KEY``, ``mount:-/path``…)."""
    names = []
    fields = [("env", "env"), ("labels", "label"), ("mounts", "mount")]
    if "ports" in desired:
        fields.append(("ports", "port"))
    for field, prefix in fields:
        want, have = desired[field], actual[field]
        names += [f"{prefix}:+{k}" for k in want if k not in have]
        names += [f"{prefix}:-{k}" for k in have if k not in want]
        names += [f"{prefix}:~{k}" for k in want if k in have and want[k] != have[k]]
    want, have = set(desired["networks"]), set(actual["networks"])
    names += [f"network:+{n}" for n in want - have] + [f"network:-{n}" for n in have - want]
    for field in ("extra_hosts", "restart"):
        if desired[field] != actual[field]:
            names.append(f"{field}:~")
    for field, name in _PROCESS_FIELDS:
        if field in desired and desired[field] != actual[field]:
            names.append(f"{name}:~")
    return sorted(names)


def mount_list(mounts: dict) -> list[str]:
    return [f"{m['source']}->{target}:{'ro' if m['read_only'] else 'rw'}"
            for target, m in sorted(mounts.items())]


def port_list(ports: dict) -> list[str]:
    """Public port publishing, e.g. ``*:8642->8642/tcp``."""
    return [f"{b['HostIp'] or '*'}:{b['HostPort']}->{key}" for key, bindings in sorted(ports.items())
            for b in bindings]


def display_values(spec: dict, env: dict) -> dict:
    """A1: a value is shown only on an exact match of the spec's closed enum."""
    shown = {}
    for name, allowed in sorted((spec.get("display_values") or {}).items()):
        if name not in env:
            shown[name] = UNSET_VALUE
        else:
            shown[name] = env[name] if env[name] in allowed else UNEXPECTED_VALUE
    return shown


def evaluate(spec: dict, container: dict, image: dict, renderer, image_ref: str,
             profile: ServiceProfile = PINNED_PROFILE) -> dict:
    """Render the target's own Compose service at ``image_ref`` and compare.

    Returns private material (``desired``, values included) and public material
    (hashes, drift names, mounts, A1 display values). ``image`` is the image the
    container RUNS; image-supplied env/labels are normalised against it.
    """
    source = compose_source(spec, container)
    rendered = renderer.render(source, image_ref)
    desired = translate(rendered["config"], spec, image_ref, profile)
    names = drift(normalized(desired, image), normalized(running(container), image))
    return {"source": source, "rendered_hash": rendered["hash"], "desired": desired, "drift": names,
            "desired_hash": canonical_hash({"source": source, "image_ref": image_ref,
                                            "rendered_hash": rendered["hash"], "desired": desired}),
            "mounts": mount_list(desired["mounts"]), "display": display_values(spec, desired["env"])}


def apply_desired(body: dict, old: dict, desired: dict, *, container_name: str, service: str,
                  rendered_hash: str, image: dict | None = None) -> dict:
    """Overlay the compose-declared fields onto a cloned create body. ``image``
    (the image the new container runs) is required when the profile models the
    process: published ports are added to its exposed ports."""
    body = json.loads(json.dumps(body))
    if "ports" in desired:
        image_config = image["Config"] or {}
        for key, field in (("Cmd", "cmd"), ("Entrypoint", "entrypoint")):
            if desired[field] is None:
                body.pop(key, None)  # the image's own applies
            else:
                body[key] = list(desired[field])
        body["OpenStdin"], body["Tty"] = desired["stdin_open"], desired["tty"]
        body["ExposedPorts"] = {**(image_config.get("ExposedPorts") or {}), **{k: {} for k in desired["ports"]}}
    body["Env"] = [f"{k}={v}" for k, v in sorted(desired["env"].items())]
    labels = {k: v for k, v in body["Labels"].items() if _bookkeeping(k)}
    labels.update(desired["labels"])
    labels[CONFIG_HASH_LABEL] = rendered_hash
    body["Labels"] = labels
    host = body["HostConfig"] = dict(body["HostConfig"])
    host["Binds"] = [f"{m['source']}:{target}:{'ro' if m['read_only'] else 'rw'}"
                     for target, m in sorted(desired["mounts"].items())]
    host["Mounts"] = None
    host["ExtraHosts"] = list(desired["extra_hosts"]) or None
    host["RestartPolicy"] = {"Name": desired["restart"][0], "MaximumRetryCount": desired["restart"][1]}
    if "ports" in desired:
        host["PortBindings"] = json.loads(json.dumps(desired["ports"]))
    if host.get("NetworkMode") not in desired["networks"]:
        host["NetworkMode"] = desired["networks"][0]
    current = body.get("NetworkingConfig", {}).get("EndpointsConfig") or {}
    body["NetworkingConfig"] = {"EndpointsConfig": {
        net: current.get(net) or {"Aliases": [container_name, service]} for net in desired["networks"]}}
    return body
