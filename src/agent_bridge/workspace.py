"""Bridge-owned workspace operations. No credentials or Git diagnostics leave here."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .bridge_token import HEADER, accepts

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CLONE_TIMEOUT = 120


class WorkspaceError(Exception):
    def __init__(self, status: int, detail: str, code: str = "clone_failed"):
        self.status, self.detail, self.code = status, detail, code
        super().__init__(detail)


@dataclass(frozen=True)
class CloneInput:
    url: str
    name: str
    branch: str | None
    github: bool


def clone_input(body: object) -> CloneInput:
    """Validate before spawning Git; deliberately never interpolate user input."""
    invalid = WorkspaceError(400, "Invalid repository URL, folder name or branch.", "invalid_clone")
    if not isinstance(body, dict) or set(body) - {"url", "name", "branch"}:
        raise invalid
    url = body.get("url")
    if not isinstance(url, str) or not 1 <= len(url) <= 2048 or re.search(r"[\s\\?#\x00-\x1f\x7f]", url):
        raise invalid
    # Accept Git's conventional SCP spelling, not arbitrary remote helpers.
    if re.fullmatch(r"git@[A-Za-z0-9.-]+:[A-Za-z0-9._/-]+", url):
        host, remote_path = url[4:].split(":", 1)
        url = f"ssh://git@{host}/{remote_path}"
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        if (
            not host or parsed.scheme not in {"https", "ssh"} or parsed.password
            or (parsed.scheme == "https" and parsed.username)
            or (parsed.scheme == "ssh" and parsed.username != "git")
            or not parsed.path or parsed.path == "/"
            or any(part in {".", ".."} for part in parsed.path.split("/"))
            or re.search(r"%|[\x00-\x20\x7f]", parsed.netloc)
        ):
            raise invalid
    except ValueError:
        raise invalid from None
    github = host.lower() == "github.com"
    if github and parsed.scheme == "ssh":
        if port not in {None, 22}:
            raise invalid
        url = urlunsplit(("https", "github.com", parsed.path, "", ""))
    name = body.get("name")
    if name is None:
        name = parsed.path.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name.endswith("."):
        raise invalid
    branch = body.get("branch")
    if branch is not None and (
        not isinstance(branch, str) or not 1 <= len(branch) <= 255
        or branch.startswith(("-", "/")) or branch.endswith(("/", "."))
        or re.search(r"[\s~^:?*\[\\\x00-\x1f\x7f]", branch)
        or ".." in branch or "@{" in branch or "//" in branch or branch == "@"
        or any(part.startswith(".") or part.endswith(".lock") for part in branch.split("/"))
    ):
        raise invalid
    return CloneInput(url, name, branch, github)


class Workspace:
    def __init__(self, root: Path, timeout: float = CLONE_TIMEOUT):
        self.root = root
        self.timeout = timeout
        self._active = 0

    def listing(self) -> dict:
        if not self.root.is_dir():
            raise WorkspaceError(503, "Workspace is unavailable.", "workspace_unavailable")
        entries = []
        for child in sorted(self.root.iterdir()):
            if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
                continue
            entries.append({"name": child.name, "path": child.name, "isGitRepo": (child / ".git").is_dir()})
        return {"entries": entries}

    async def clone(self, item: CloneInput) -> dict:
        if self._active >= 2:
            raise WorkspaceError(503, "Clone service is busy. Try again shortly.", "clone_busy")
        self._active += 1
        target = self.root / item.name
        staging: Path | None = None
        claimed: os.stat_result | None = None
        proc = None
        try:
            # mkdir is the cross-process duplicate guard. Never replace a repo,
            # symlink, or another clone; cleanup only our exact empty reservation.
            try:
                target.mkdir()
                claimed = target.lstat()
            except FileExistsError:
                raise WorkspaceError(409, "That workspace folder already exists.", "folder_exists") from None
            staging = Path(tempfile.mkdtemp(prefix=".clone-", dir=self.root))
            env = {
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
                "GCM_INTERACTIVE": "never",
                "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15",
            }
            args = [
                "git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
                "-c", "protocol.ssh.allow=always", "-c", "core.hooksPath=/dev/null",
                "clone", "--template=", "--quiet",
            ]
            if item.branch:
                args += ["--branch", item.branch]
            args += ["--", item.url, str(staging)]
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=self.root, env=env, start_new_session=True,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), self.timeout)
            except TimeoutError:
                raise WorkspaceError(504, "Clone timed out.", "clone_timeout") from None
            if proc.returncode:
                raise WorkspaceError(502, "Git could not clone the repository. Check URL, branch and access.")
            current = target.lstat()
            if (current.st_dev, current.st_ino) != (claimed.st_dev, claimed.st_ino):
                raise WorkspaceError(409, "Workspace folder changed during clone.", "folder_exists")
            # mkdtemp keeps the in-flight checkout private (0700). Published
            # repos must be readable by the frontend's different UID through
            # its read-only workspace mount, without granting it write access.
            staging.chmod(0o755)
            staging.rename(target)  # replaces only the empty directory we reserved
            staging = None
            claimed = None
            return {"name": item.name, "path": item.name, **({"branch": item.branch} if item.branch else {})}
        except OSError:
            raise WorkspaceError(503, "Workspace is unavailable.", "workspace_unavailable") from None
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
            if staging is not None:
                await asyncio.to_thread(shutil.rmtree, staging, True)
            if claimed is not None:
                try:
                    current = target.lstat()
                    if (current.st_dev, current.st_ino) == (claimed.st_dev, claimed.st_ino):
                        target.rmdir()
                except OSError:
                    pass
            self._active -= 1


def add_workspace_routes(app: FastAPI, root: Path | None = None) -> None:
    workspace = Workspace(root or Path(os.getenv("BRIDGE_WORKSPACE_ROOT", "/home/bridge/dev")))

    @app.middleware("http")
    async def workspace_gate(request: Request, call_next):
        if request.url.path in {"/workspace", "/workspace/clone"}:
            if not accepts(request.headers.get(HEADER, "")):
                return JSONResponse({"detail": "Bad bridge token."}, status_code=401)
        return await call_next(request)

    @app.exception_handler(WorkspaceError)
    async def workspace_error(_request: Request, exc: WorkspaceError):
        return JSONResponse({"detail": exc.detail, "code": exc.code}, status_code=exc.status)

    @app.get("/workspace")
    async def listing():
        try:
            return await asyncio.to_thread(workspace.listing)
        except OSError:
            raise WorkspaceError(503, "Workspace is unavailable.", "workspace_unavailable") from None

    @app.post("/workspace/clone")
    async def clone(request: Request):
        # Stream with a bound rather than reflecting pydantic validation input,
        # which could contain an accidentally pasted credential-bearing URL.
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise WorkspaceError(400, "Invalid clone request.", "invalid_clone")
        import json
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            raise WorkspaceError(400, "Invalid clone request.", "invalid_clone") from None
        return await workspace.clone(clone_input(body))
