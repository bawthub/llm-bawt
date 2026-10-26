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

Fields outside the allowlist (ports, privileges, devices, user, …) are carried
from the running container exactly as a plain image deploy would.
"""
from __future__ import annotations

import hashlib
import json
import re

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
UNEXPECTED_VALUE = "<unexpected value>"
UNSET_VALUE = "<unset>"


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


def translate(rendered: dict, spec: dict, image_ref: str) -> dict:
    """The desired service, restricted to the modelled keys. Values included."""
    project = spec["compose_project"]
    service = (rendered.get("services") or {}).get(spec["compose_service"])
    if not isinstance(service, dict):
        raise ComposeError("compose_service_missing")
    unknown = sorted(set(service) - SERVICE_KEYS)
    if unknown:
        raise ComposeError("unsupported_compose_key", unknown)
    if service.get("pull_policy", "missing") not in PULL_POLICIES:
        raise ComposeError("unsupported_pull_policy")
    for key in ("command", "entrypoint"):
        if service.get(key) is not None:
            raise ComposeError("unsupported_compose_key", [key])
    if service.get("image") != image_ref:
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
    return {"env": _strings(service.get("environment"), "environment"),
            "labels": _strings(service.get("labels"), "labels"),
            "mounts": mounts, "networks": sorted(set(networks)),
            "extra_hosts": sorted(_host_entry(h) for h in hosts),
            "restart": [restart.group(1), int(restart.group(2) or 0)]}


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
    return {"env": env, "labels": dict(config.get("Labels") or {}), "mounts": mounts,
            "networks": sorted((container.get("NetworkSettings") or {}).get("Networks") or {}),
            "extra_hosts": sorted(host.get("ExtraHosts") or []),
            "restart": [policy.get("Name") or "no", int(policy.get("MaximumRetryCount") or 0)]}


def normalized(state: dict, image: dict) -> dict:
    """Drop what the image supplies and Compose/ops bookkeeping (both sides)."""
    image_config = image.get("Config") or {}
    image_env = dict(e.partition("=")[::2] for e in image_config.get("Env") or [])
    image_labels = image_config.get("Labels") or {}
    return {**state,
            "env": {k: v for k, v in state["env"].items() if image_env.get(k) != v},
            "labels": {k: v for k, v in state["labels"].items()
                       if not _bookkeeping(k) and image_labels.get(k) != v}}


def drift(desired: dict, actual: dict) -> list[str]:
    """NAMES of differing fields (``env:+KEY``, ``label:~KEY``, ``mount:-/path``…)."""
    names = []
    for field, prefix in (("env", "env"), ("labels", "label"), ("mounts", "mount")):
        want, have = desired[field], actual[field]
        names += [f"{prefix}:+{k}" for k in want if k not in have]
        names += [f"{prefix}:-{k}" for k in have if k not in want]
        names += [f"{prefix}:~{k}" for k in want if k in have and want[k] != have[k]]
    want, have = set(desired["networks"]), set(actual["networks"])
    names += [f"network:+{n}" for n in want - have] + [f"network:-{n}" for n in have - want]
    for field in ("extra_hosts", "restart"):
        if desired[field] != actual[field]:
            names.append(f"{field}:~")
    return sorted(names)


def mount_list(mounts: dict) -> list[str]:
    return [f"{m['source']}->{target}:{'ro' if m['read_only'] else 'rw'}"
            for target, m in sorted(mounts.items())]


def display_values(spec: dict, env: dict) -> dict:
    """A1: a value is shown only on an exact match of the spec's closed enum."""
    shown = {}
    for name, allowed in sorted((spec.get("display_values") or {}).items()):
        if name not in env:
            shown[name] = UNSET_VALUE
        else:
            shown[name] = env[name] if env[name] in allowed else UNEXPECTED_VALUE
    return shown


def evaluate(spec: dict, container: dict, image: dict, renderer, image_ref: str) -> dict:
    """Render the target's own Compose service at ``image_ref`` and compare.

    Returns private material (``desired``, values included) and public material
    (hashes, drift names, mounts, A1 display values). ``image`` is the image the
    container RUNS; image-supplied env/labels are normalised against it.
    """
    source = compose_source(spec, container)
    rendered = renderer.render(source, image_ref)
    desired = translate(rendered["config"], spec, image_ref)
    names = drift(normalized(desired, image), normalized(running(container), image))
    return {"source": source, "rendered_hash": rendered["hash"], "desired": desired, "drift": names,
            "desired_hash": canonical_hash({"source": source, "image_ref": image_ref,
                                            "rendered_hash": rendered["hash"], "desired": desired}),
            "mounts": mount_list(desired["mounts"]), "display": display_values(spec, desired["env"])}


def apply_desired(body: dict, old: dict, desired: dict, *, container_name: str, service: str,
                  rendered_hash: str) -> dict:
    """Overlay the compose-declared fields onto a cloned create body."""
    body = json.loads(json.dumps(body))
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
    if host.get("NetworkMode") not in desired["networks"]:
        host["NetworkMode"] = desired["networks"][0]
    current = body.get("NetworkingConfig", {}).get("EndpointsConfig") or {}
    body["NetworkingConfig"] = {"EndpointsConfig": {
        net: current.get(net) or {"Aliases": [container_name, service]} for net in desired["networks"]}}
    return body
