"""Render ONE Compose service in a sealed one-shot container. STANDARD LIBRARY ONLY (TASK-1518).

Compose is the declared configuration of a Compose-managed target, but neither
the app image nor the stdlib-only ops worker ships the Compose CLI or the
project files. Both therefore ask the Docker daemon to run a digest-pinned
Docker CLI image with:

* no network, no Docker socket, a read-only root filesystem, every capability
  dropped and ``no-new-privileges``;
* the logging driver ``none``, so rendered values never reach daemon logs, and
  output read once through ``attach``;
* exactly one read-only bind: the target's own Compose working directory, at
  the same path, so relative paths and the project ``.env`` resolve as they do
  for the operator. Both directory and files come from the target's Compose
  labels and must match the catalog spec's file names;
* a fixed argv (``compose -p … -f … config``) run under ``env -i``: the
  renderer image's own environment (notably ``HOME=/root``) never reaches
  interpolation (TASK-1476). Compose sees only a fixed ``PATH``, the pinned
  image reference for the image overlay (none for a Compose-built target) and
  the spec's declared ``interpolation_env`` (e.g. the operator's ``HOME``,
  deployment data owned by the catalog row). Anything else a Compose file
  references is unset and refuses.

The container is removed (with its anonymous volumes) in every outcome.
Renderer stderr is never echoed: it can carry interpolated values. Only fixed
error codes, exit status and the names of unset variables are reported.
An unset variable fails closed. The operator's shell may have supplied it when
the target was created, and the renderer cannot reproduce that shell.
"""
from __future__ import annotations

import json
import posixpath
import re

# The container itself is sealed_container's, shared with the TASK-1474 source
# check. OPS_JOB_LABEL is re-exported for the target selectors.
if __package__:
    from .sealed_container import OPS_JOB_LABEL, SealedContainer, SealedRunError  # noqa: F401
    from .sealed_container import demux_streams as demux_bytes
else:  # standalone immutable worker
    from sealed_container import OPS_JOB_LABEL, SealedContainer, SealedRunError  # noqa: F401
    from sealed_container import demux_streams as demux_bytes

RENDER_LABEL = "llm-bawt.ops.compose-render"
RENDER_ROLE = "compose-render"
MAX_RENDER_BYTES = 4 * 1024 * 1024
_WORKING_DIR = "com.docker.compose.project.working_dir"
_CONFIG_FILES = "com.docker.compose.project.config_files"
SAFE_PATH = re.compile(r"/[A-Za-z0-9._/-]{1,255}")
COMPOSE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.ya?ml")
MAX_COMPOSE_FILES = 4
RENDER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_ENV_VALUE = re.compile(r"[A-Za-z0-9._/:@-]{1,255}")
# A CLOSED allowlist (Codex r1 Blocker): the renderer executes docker with the
# project bound, so a name like LD_PRELOAD, DOCKER_* or COMPOSE_* would let a
# catalog row run project code or redirect the CLI before the source check.
# Only process-environment identity Compose cannot read from the project
# ``.env`` is declarable. Path-valued names must be clean absolute paths.
_PATH_NAMES = frozenset({"HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"})
INTERPOLATION_NAMES = _PATH_NAMES | {"USER", "LOGNAME"}
# The image variable carries only the bound image ref, but its NAME is catalog
# data too: never one that steers the loader or the docker/compose CLI.
_CONTROL_PREFIXES = ("LD_", "DOCKER_", "COMPOSE_", "BUILDKIT_", "GODEBUG", "GOTRACEBACK", "SSL_",
                     "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_UNSET = re.compile(r'time="[^"]*" level=warning msg="The \\"([A-Za-z_][A-Za-z0-9_]{0,127})\\" '
                    r'variable is not set\. Defaulting to a blank string\."')


class ComposeError(ValueError):
    """Fixed code plus optional NAMES (variables/keys); never a value."""

    def __init__(self, code: str, names=(), *, detail: str = ""):
        self.code = code
        self.names = sorted(set(names))
        text = code
        if self.names:
            text += ": " + ", ".join(self.names)
        if detail:
            text += f" ({detail})"
        super().__init__(text)


def validate_interpolation_env(config: dict) -> dict:
    """``compose_config.interpolation_env`` (optional): the only variables besides
    the image variable Compose may interpolate. Allowlisted names, plain values."""
    value = config.get("interpolation_env", {})
    if not isinstance(value, dict):
        raise ValueError("compose_config.interpolation_env must be an object")
    for name, item in value.items():
        if name not in INTERPOLATION_NAMES or name == config.get("image_variable"):
            raise ValueError("compose_config.interpolation_env names must be one of "
                             + ", ".join(sorted(INTERPOLATION_NAMES)))
        if not isinstance(item, str) or not _ENV_VALUE.fullmatch(item):
            raise ValueError(f"compose_config.interpolation_env.{name} is invalid")
        if name in _PATH_NAMES and not working_dir_valid(item):
            raise ValueError(f"compose_config.interpolation_env.{name} must be a clean absolute path")
    return value


def renderer_env(config: dict, image_ref: str) -> list[str]:
    """The renderer's whole environment, ``env -i`` arguments: a fixed PATH, the
    image variable (if any) and the declared ``interpolation_env``."""
    declared = [f"{name}={value}" for name, value in sorted(validate_interpolation_env(config).items())]
    variable = config.get("image_variable")
    if variable and (variable == "PATH" or variable in INTERPOLATION_NAMES
                     or variable.startswith(_CONTROL_PREFIXES)):
        raise ValueError("compose_config.image_variable must not be an execution-control name")
    return [f"PATH={RENDER_PATH}", *([f"{variable}={image_ref}"] if variable else []), *declared]


def compose_source(spec: dict, container: dict) -> dict:
    """Project directory and files the target was created from, as the spec allows."""
    labels = container.get("Config", {}).get("Labels") or {}
    working_dir = labels.get(_WORKING_DIR) or ""
    files = (labels.get(_CONFIG_FILES) or "").split(",")
    names = spec["compose_config"].get("compose_files")
    if not working_dir_valid(working_dir):
        raise ComposeError("compose_source_invalid", detail="working directory label")
    if names is None:
        # TASK-1474: which overlays an install layers is that install's own
        # choice, so the spec names none. The files still must be distinct
        # Compose basenames directly in the working directory, and the bound
        # list is shown on the approval card.
        if not label_files_valid(working_dir, files):
            raise ComposeError("compose_source_invalid", detail="config files label")
    elif files != [posixpath.join(working_dir, name) for name in names]:
        raise ComposeError("compose_source_invalid", detail="config files differ from the catalog spec")
    return {"project_dir": working_dir, "compose_files": files}


def working_dir_valid(working_dir: str) -> bool:
    """A Compose working-directory label safe to bind read-only at the same path."""
    return (isinstance(working_dir, str) and bool(SAFE_PATH.fullmatch(working_dir))
            and posixpath.normpath(working_dir) == working_dir and working_dir != "/"
            and ".." not in working_dir.split("/"))


def label_files_valid(working_dir: str, files: list) -> bool:
    return (1 <= len(files) <= MAX_COMPOSE_FILES and len(set(files)) == len(files)
            and all(isinstance(f, str) and posixpath.dirname(f) == working_dir
                    and COMPOSE_FILE.fullmatch(posixpath.basename(f)) for f in files))


def renderer_argv(spec: dict, source: dict, *, hash_only: bool) -> list[str]:
    argv = ["compose", "-p", spec["compose_project"]]
    for path in source["compose_files"]:
        argv += ["-f", path]
    service = spec["compose_service"]
    return argv + (["config", "--hash", service] if hash_only else ["config", "--format", "json", service])


def demux_streams(raw: bytes) -> tuple[str, str]:
    """(stdout, stderr) text of Docker's multiplexed non-TTY attach stream."""
    out, err = demux_bytes(raw)
    return out.decode(errors="replace"), err.decode(errors="replace")


class ComposeRenderer:
    """Render the spec's service. ``api`` is the worker's stdlib DockerAPI shape:
    ``request(method, path, body=..., raw=...)`` plus ``attach_stream(id)``."""

    def __init__(self, api, spec: dict, *, tag: str):
        if not re.fullmatch(r"[a-z0-9]{1,32}", tag):
            raise ValueError("renderer tag must be short lowercase alphanumerics")
        self.api = api
        self.spec = spec
        self.tag = tag

    def render(self, source: dict, image_ref: str) -> dict:
        """``{"config": <rendered project JSON>, "hash": <compose config hash>}``."""
        raw = self._run(source, image_ref, renderer_argv(self.spec, source, hash_only=False))
        try:
            config = json.loads(raw)
        except ValueError:
            raise ComposeError("renderer_output_invalid") from None
        if not isinstance(config, dict):
            raise ComposeError("renderer_output_invalid")
        line = self._run(source, image_ref, renderer_argv(self.spec, source, hash_only=True)).strip()
        match = re.fullmatch(re.escape(self.spec["compose_service"]) + r" ([0-9a-f]{64})", line)
        if not match:
            raise ComposeError("renderer_output_invalid", detail="config hash")
        return {"config": config, "hash": match.group(1)}

    def project(self, source: dict, project: str) -> dict:
        """The WHOLE rendered project, every profile included (TASK-1439 T2c):
        a container-removal check must see every service that could claim a
        name, not only one service or the active profiles. No injected env."""
        argv = ["compose", "-p", project]
        for path in source["compose_files"]:
            argv += ["-f", path]
        raw = self._run(source, "", argv + ["--profile", "*", "config", "--format", "json"])
        try:
            config = json.loads(raw)
        except ValueError:
            raise ComposeError("renderer_output_invalid") from None
        if not isinstance(config, dict) or not isinstance(config.get("services"), dict):
            raise ComposeError("renderer_output_invalid")
        return config

    def _run(self, source: dict, image_ref: str, argv: list[str]) -> str:
        sealed = SealedContainer(self.api, self.spec["compose_config"]["renderer_image"], role=RENDER_ROLE,
                                 tag=self.tag, labels={RENDER_LABEL: "1"})
        # A Compose-built target (TASK-1474) has no image overlay to inject.
        env = renderer_env(self.spec["compose_config"], image_ref)
        try:
            # ``env -i``: Docker ``Env`` would only add to the image's own (HOME=/root).
            status, raw = sealed.run(entrypoint=["env", "-i", *env, "docker"], cmd=argv, env=[],
                                     directory=source["project_dir"], max_bytes=MAX_RENDER_BYTES)
        except SealedRunError as exc:
            raise ComposeError(f"renderer_{exc.kind}") from None
        stdout, stderr = demux_streams(raw)
        unset, unexpected = set(), False
        for line in stderr.splitlines():
            if not line.strip():
                continue
            match = _UNSET.fullmatch(line.strip())
            if match:
                unset.add(match.group(1))
            else:
                unexpected = True
        if status != 0:
            raise ComposeError("renderer_failed", detail=f"exit {status}")
        if unset:
            raise ComposeError("unset_compose_variable", unset)
        if unexpected:
            raise ComposeError("renderer_warning")
        return stdout
