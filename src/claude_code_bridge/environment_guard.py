"""Pre-execution protection for the host-owned shared Python environment.

This is an accident guard for ordinary shell commands, not a shell sandbox.
Direct pytest/ruff do not rebuild environments; heavy-check admission remains
the runner's job. Task-local environments are deliberately allowed.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import re
import shutil

from agent_bridge.bridge_token import BROKER


DENY_REASON = (
    "BLOCKED: this command can rewrite the shared checkout's host-owned .venv "
    "from inside the bridge. Use a task-local environment in an isolated worktree, "
    "or run the repository's Make target on the configured test host. Focused "
    "pytest/ruff using an existing environment are allowed; broad checks still "
    "belong on the heavy queue or make gate."
)
_VARIABLE = re.compile(r"\$(?:\{(\w+)\}|(\w+))")
_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_UV_WRITERS = {"run", "sync", "add", "remove", "venv"}
_PIP_WRITERS = {"install", "uninstall", "sync"}
_PIP_VALUES = {"--python", "-p", "--log", "--proxy", "--timeout", "--retries",
               "--cache-dir", "--client-cert", "--cert"}
_UV_VALUES = {"--directory", "-C", "--project", "--config-file", "--color",
              "--cache-dir", "--python", "-p", "--package", "--extra", "--group",
              "--with", "--with-editable", "--with-requirements", "--index",
              "--index-url", "--default-index", "--python-preference"}


def _path(value: str, cwd: Path, env: Mapping[str, str]) -> Path:
    value = _VARIABLE.sub(lambda m: env.get(m[1] or m[2], m[0]), value)
    if value == "~" or value.startswith("~/"):
        value = env.get("HOME", str(Path.home())) + value[1:]
    path = Path(value)
    return Path(os.path.abspath(path if path.is_absolute() else cwd / path))


def _option(words: list[str], *names: str) -> str | None:
    found = None
    for i, word in enumerate(words):
        if word == "--":
            break
        for name in names:
            if word == name and i + 1 < len(words):
                found = words[i + 1]
            elif word.startswith(name + "="):
                found = word[len(name) + 1:]
            elif len(name) == 2 and word.startswith(name) and len(word) > 2:
                found = word[2:]
    return found


def _project(cwd: Path) -> Path:
    return next((p for p in (cwd, *cwd.parents) if (p / "pyproject.toml").is_file()), cwd)


def _venv(cwd: Path, env: Mapping[str, str]) -> Path:
    if env.get("VIRTUAL_ENV"):
        return _path(env["VIRTUAL_ENV"], cwd, env)
    return next((p / ".venv" for p in (cwd, *cwd.parents) if (p / ".venv").exists()),
                cwd / ".venv")


def _touches(target: Path, shared: Path) -> bool:
    # An interpreter symlink inside a venv still selects *that* environment.
    return any(p == shared or p.is_relative_to(shared)
               for p in (target, target.resolve(), target.parent.resolve() / target.name))


def _leading_options(words: list[str]) -> tuple[list[str], list[str]]:
    """Separate uv options from the program/destination they precede."""
    i = 0
    while i < len(words) and words[i].startswith("-"):
        if words[i] == "--":
            return words[:i], words[i + 1:]
        i += 2 if words[i] in _UV_VALUES else 1
    return words[:i], words[i:]


def _uv_writes(words: list[str], cwd: Path, env: Mapping[str, str], shared: Path) -> bool:
    # Find the uv subcommand without mistaking a global option's value for it.
    i = 1
    while i < len(words) and words[i].startswith("-"):
        word = words[i]
        i += 2 if word in _UV_VALUES else 1
    if i == len(words):
        return False
    sub, args = words[i], words[i + 1:]
    # uv run's trailing program is not part of uv's option namespace.
    options, operands = _leading_options(args) if sub in {"run", "venv"} else (args, [])
    directory = _option(words[1:i] + options, "--directory", "-C")
    if directory:
        cwd = _path(directory, cwd, env)
    project = _option(words[1:i] + options, "--project")
    root = _project(_path(project, cwd, env) if project else cwd)
    if sub == "pip":
        return _pip_writes(args, cwd, env, shared, uv=True)
    if sub not in _UV_WRITERS:
        return False
    if sub in {"add", "remove"} and "--no-sync" in options:
        return False
    if sub == "venv":
        # The common implicit destination plus any explicit destination.
        return _touches(_path(operands[0], cwd, env) if operands else cwd / ".venv", shared)
    target = (_venv(cwd, env) if "--active" in options and env.get("VIRTUAL_ENV")
              else _path(env.get("UV_PROJECT_ENVIRONMENT") or ".venv", root, env))
    return _touches(target, shared)


def _pip_writes(words: list[str], cwd: Path, env: Mapping[str, str],
                shared: Path, *, uv: bool = False, executable: str = "") -> bool:
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in _PIP_VALUES else 1
    if i >= len(words) or words[i] not in _PIP_WRITERS:
        return False
    destination = (_option(words, "--target", "-t", "--prefix", "--root")
                   or env.get("PIP_TARGET") or env.get("PIP_PREFIX") or env.get("PIP_ROOT"))
    if destination:
        return _touches(_path(destination, cwd, env), shared)
    python = _option(words, "--python", "-p") or (env.get("UV_PYTHON") if uv else None)
    if python and ("/" in python or python.startswith(".")):
        return _touches(_path(python, cwd, env), shared)
    if "--system" in words or "--user" in words:
        return False
    if uv:
        return _touches(_venv(cwd, env), shared)
    # pip/python use their executable, not a .venv merely present in cwd.
    search_path = os.pathsep.join(str(_path(part or ".", cwd, env))
                                 for part in env.get("PATH", "").split(os.pathsep))
    binary = (str(_path(executable, cwd, env)) if "/" in executable
              else shutil.which(executable, path=search_path))
    return bool(binary and _touches(Path(binary).absolute(), shared))


def _unsafe(command: str, cwd: Path, env: Mapping[str, str], shared: Path,
            depth: int = 0) -> bool:
    from .skill_hooks import _COMMAND_PREFIXES, _RESERVED_PREFIXES, _SHELLS
    from .skill_hooks import _quoted_substitutions, _segments

    if depth > 4:
        return False
    if any(_unsafe(body, cwd, env, shared, depth + 1)
           for body in _quoted_substitutions(command)):
        return True
    try:
        segments = _segments(command, preserve_scopes=True)
    except ValueError:
        return False  # Bash rejects malformed quoting before it can run uv.
    state = dict(env)
    scopes = []
    directories = []
    for segment in segments:
        if segment == ["("]:
            scopes.append((cwd, dict(state), list(directories)))
            continue
        if segment == [")"]:
            if scopes:
                cwd, state, directories = scopes.pop()
            continue
        words, local = list(segment), dict(state)
        while words and (words[0] in _RESERVED_PREFIXES or _ASSIGNMENT.match(words[0])):
            first = words.pop(0)
            if _ASSIGNMENT.match(first):
                key, value = first.split("=", 1)
                local[key] = _VARIABLE.sub(lambda m: local.get(m[1] or m[2], m[0]), value)
        if not words:
            state = local
            continue
        if words[0] == "export":
            for word in words[1:]:
                if _ASSIGNMENT.match(word):
                    key, value = word.split("=", 1)
                    state[key] = _VARIABLE.sub(lambda m: state.get(m[1] or m[2], m[0]), value)
            continue
        if words[0] == "unset":
            for word in words[1:]:
                state.pop(word, None)
            continue
        while words and words[0].rsplit("/", 1)[-1] in _COMMAND_PREFIXES | {"timeout"}:
            wrapper = words.pop(0).rsplit("/", 1)[-1]
            while words and (_ASSIGNMENT.match(words[0]) or words[0].startswith("-")):
                word = words.pop(0)
                if _ASSIGNMENT.match(word):
                    key, value = word.split("=", 1)
                    local[key] = _VARIABLE.sub(lambda m: local.get(m[1] or m[2], m[0]), value)
                elif wrapper == "env" and word in {"-i", "--ignore-environment"}:
                    local.clear()
                elif words and word in {"-u", "--unset", "-n", "--adjustment", "-g", "--group"}:
                    value = words.pop(0)
                    if wrapper == "env" and word in {"-u", "--unset"}:
                        local.pop(value, None)
                elif wrapper == "timeout" and words and word in {
                        "-s", "--signal", "-k", "--kill-after"}:
                    words.pop(0)
            if wrapper == "timeout" and words:
                words.pop(0)  # duration is positional, before the wrapped program
        if not words:
            continue
        program = words[0].rsplit("/", 1)[-1]
        if program == "cd":
            cwd = _path(words[-1] if len(words) > 1 else "~", cwd, local)
        elif program == "pushd":
            if len(words) > 1:
                directories.append(cwd)
                cwd = _path(words[-1], cwd, local)
            elif directories:
                cwd, directories[-1] = directories[-1], cwd
        elif program == "popd" and directories:
            cwd = directories.pop()
        elif program in {"source", "."} and len(words) > 1:
            activation = _path(words[1], cwd, local)
            if activation.name == "activate" and activation.parent.name == "bin":
                state["VIRTUAL_ENV"] = str(activation.parent.parent)
                state["PATH"] = str(activation.parent) + os.pathsep + state.get("PATH", "")
        elif program in _SHELLS:
            for i, word in enumerate(words[1:-1], 1):
                if word.startswith("-") and not word.startswith("--") and "c" in word:
                    if _unsafe(words[i + 1], cwd, local, shared, depth + 1):
                        return True
                    break
        elif program == "uv" and _uv_writes(words, cwd, local, shared):
            return True
        elif program.startswith(("pip", "python")):
            if program.startswith("python") and words[1:3] == ["-m", "uv"]:
                if _uv_writes(["uv", *words[3:]], cwd, local, shared):
                    return True
            pip_args = words[1:] if program.startswith("pip") else (
                words[words.index("pip") + 1:] if "-m" in words and "pip" in words else [])
            if _pip_writes(pip_args, cwd, local, shared, executable=words[0]):
                return True
    return False


def environment_gate(data: Mapping, env: Mapping[str, str]) -> dict | None:
    from .skill_hooks import _deny

    if env.get("BAWT_AGENT_BRIDGE") != "1" and not env.get(BROKER.file_env):
        return None
    command = (data.get("tool_input") or {}).get("command")
    if not isinstance(command, str):
        return None
    cwd = Path(data.get("cwd") or Path.cwd()).resolve()
    workspace = _path(env.get("BRIDGE_WORKSPACE_ROOT") or "~/dev", cwd, env)
    shared = (workspace / "llm-bawt" / ".venv").resolve()
    return _deny(DENY_REASON) if _unsafe(command, cwd, env, shared) else None
