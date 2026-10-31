"""TASK-1474: the sealed git source check, run against REAL git on a scratch repository.

The fake Docker API keeps the sealed container's contract (one read-only
project bind, a scratch ``/tmp``, fixed argv, the session's env) but runs each
exec with the host's git, so every refusal here is git's own answer about a
real working tree. No live Docker or network.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib.error
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from llm_bawt.ops.compose_render import RENDER_DOCKER_CONFIG, ComposeError, ComposeRenderer
from llm_bawt.ops.release import GitHubReleaseVerifier, ReleaseVerificationError
from llm_bawt.ops.sealed_container import OPS_JOB_LABEL, SealedContainer, SealedRunError
from llm_bawt.ops.source_state import (
    SOURCE_CHECK_LABEL, SourceChecker, SourceStateError, validate_source_spec,
)

IMAGE = "docker.io/library/docker@sha256:" + "d" * 64
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required for the real-git source check")


def _frame(stream: int, data: bytes) -> bytes:
    return struct.pack(">BxxxI", stream, len(data)) + data


class GitDocker:
    """Engine API subset for :class:`SealedContainer` sessions; execs run host git."""

    def __init__(self, scratch: Path, *, image_present=True, fail_exec=False):
        self.scratch, self.image_present, self.fail_exec = scratch, image_present, fail_exec
        self.containers, self.execs, self.bodies, self.argv = {}, {}, [], []

    def request(self, method, path, *, body=None, headers=None, raw=False):
        url = urlsplit(path)
        parts = url.path.strip("/").split("/")
        if parts[0] == "images":
            if not self.image_present:
                raise RuntimeError("Docker HTTP 404: no such image")
            return {"Id": "sha256:" + "d" * 64}
        if parts[:2] == ["containers", "create"]:
            self.bodies.append(body)
            cid = f"{len(self.bodies):064x}"
            self.containers[cid] = {"body": body, "name": parse_qs(url.query)["name"][0], "running": False}
            return {"Id": cid}
        if parts[0] == "containers" and method == "DELETE":
            assert url.query == "force=1&v=1"
            self.containers.pop(parts[1])
            return None
        if parts[0] == "containers" and parts[2] == "start":
            self.containers[parts[1]]["running"] = True
            return None
        if parts[0] == "containers" and parts[2] == "exec":
            if self.fail_exec:
                raise RuntimeError("Docker HTTP 500: exec failed")
            assert self.containers[parts[1]]["running"]
            exec_id = f"x{len(self.execs)}"
            self.execs[exec_id] = {"container": parts[1], **body}
            return {"Id": exec_id}
        if parts[0] == "exec" and parts[2] == "start":
            assert raw
            return self._run(self.execs[parts[1]])
        if parts[0] == "exec" and parts[2] == "json":
            return {"Running": False, "ExitCode": self.execs[parts[1]]["code"]}
        raise AssertionError(f"unexpected {method} {path}")

    def attach_stream(self, container_id):  # pragma: no cover - sessions never attach
        raise AssertionError("a source check never attaches")

    def _run(self, record):
        body = self.containers[record["container"]]["body"]
        argv = record["Cmd"]
        assert argv[0] == "git"
        self.argv.append(argv)
        env = {}
        for item in [*body["Env"], *record["Env"]]:
            name, _, value = item.partition("=")
            env[name] = value.replace("/tmp/", f"{self.scratch}/")  # the container's tmpfs
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")  # the image's PATH
        proc = subprocess.run([GIT, *argv[1:]], cwd=body["WorkingDir"], env=env, capture_output=True,
                              check=False, timeout=60)
        record["code"] = proc.returncode
        return _frame(1, proc.stdout) + _frame(2, proc.stderr)


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent), "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run([GIT, *args], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = (tmp_path / "project").resolve()
    (root / "src" / "pkg").mkdir(parents=True)
    git(root, "init", "-q", "-b", "master")
    (root / "src" / "pkg" / "__init__.py").write_text("")
    (root / "src" / "pkg" / "module.py").write_text("VALUE = 1\n")
    (root / "server.sh").write_text("#!/bin/sh\nexec llm-service\n")
    (root / "server.sh").chmod(0o755)
    (root / "docker-compose.yml").write_text("services: {}\n")
    (root / "storage").mkdir()
    (root / "storage" / "payload.py").write_text("import os\n")
    (root / ".gitignore").write_text("storage/\n__pycache__/\n*.egg-info/\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def plan(**overrides):
    return {"paths": ["server.sh", "src"], "compose_files": ["docker-compose.yml"], "bytecode_ok": False,
            **overrides}


def check(repo, tmp_path, **plan_overrides):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    api = GitDocker(scratch)
    try:
        return SourceChecker(api, IMAGE, tag="t1").check(str(repo), plan(**plan_overrides)), api
    finally:
        assert api.containers == {}  # the session container is removed in every outcome


def refused(repo, tmp_path, **plan_overrides) -> SourceStateError:
    with pytest.raises(SourceStateError) as caught:
        check(repo, tmp_path, **plan_overrides)
    return caught.value


def commit(repo, message="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


# ── clean and the sealed container ──────────────────────────────────────────

def test_a_clean_checkout_is_its_head_commit(repo, tmp_path):
    head, api = check(repo, tmp_path)
    assert head == git(repo, "rev-parse", "HEAD").strip()
    body = api.bodies[0]
    host = body["HostConfig"]
    assert host["NetworkMode"] == "none" and body["NetworkDisabled"] is True
    assert host["Binds"] == [f"{repo}:{repo}:ro"] and host["ReadonlyRootfs"] is True
    assert host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=67108864"}
    assert host["CapDrop"] == ["ALL"] and host["SecurityOpt"] == ["no-new-privileges:true"]
    assert host["LogConfig"]["Type"] == "none" and host["Privileged"] is False
    assert body["Entrypoint"] == ["sleep"] and body["WorkingDir"] == str(repo)
    assert body["Labels"][SOURCE_CHECK_LABEL] == "1" and body["Labels"][OPS_JOB_LABEL] == "t1"
    assert body["Labels"]["com.docker.compose.project"] == "llm-bawt-ops-render"
    assert "HOME=/nonexistent" in body["Env"] and "GIT_CONFIG_NOSYSTEM=1" in body["Env"]
    for argv in api.argv:
        assert {"core.fsmonitor=false", "core.fileMode=true", "core.symlinks=true",
                f"safe.directory={repo}"} <= set(argv)
    assert ["hash-object", "--no-filters"] == api.argv[-1][api.argv[-1].index("hash-object"):][:2]


def test_untracked_files_outside_the_source_paths_are_not_its_concern(repo, tmp_path):
    (repo / "notes.txt").write_text("scratch\n")
    (repo / "storage" / "new.db").write_text("x")
    assert check(repo, tmp_path)[0] == git(repo, "rev-parse", "HEAD").strip()


def test_image_missing_or_docker_down_is_unverifiable(repo, tmp_path):
    for api in (GitDocker(tmp_path, image_present=False), GitDocker(tmp_path, fail_exec=True)):
        with pytest.raises(SourceStateError) as caught:
            SourceChecker(api, IMAGE, tag="t1").check(str(repo), plan())
        assert caught.value.code == "source_unverifiable" and api.containers == {}


def test_a_subdirectory_or_a_non_repository_is_refused(repo, tmp_path):
    assert refused(repo / "src", tmp_path).code == "source_tree_unsupported"
    plain = (tmp_path / "plain").resolve()
    plain.mkdir()
    assert refused(plain, tmp_path).code == "source_unverifiable"


# ── tracked changes ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("change", ["edit", "staged", "delete", "mode", "compose-edit", "outside-src"])
def test_any_tracked_change_in_the_repository_is_dirty(repo, tmp_path, change):
    module = repo / "src" / "pkg" / "module.py"
    if change == "edit":
        module.write_text("VALUE = 2\n")
    elif change == "staged":
        module.write_text("VALUE = 2\n")
        git(repo, "add", "src/pkg/module.py")  # the real index is never trusted
    elif change == "delete":
        module.unlink()
    elif change == "mode":
        module.chmod(0o755)
    elif change == "compose-edit":
        (repo / "docker-compose.yml").write_text("services: {app: {}}\n")
    else:
        (repo / ".gitignore").write_text("")
    caught = refused(repo, tmp_path)
    assert caught.code == "source_dirty"


def test_a_clean_filter_cannot_launder_source_bytes(repo, tmp_path):
    """r1 repro: a clean filter strips the injected line, so status is clean;
    the raw hash still differs from the HEAD blob."""
    (repo / ".gitattributes").write_text("*.py filter=strip\n")
    commit(repo, "attributes")
    git(repo, "config", "filter.strip.clean", "sed /INJECTED/d")
    module = repo / "src" / "pkg" / "module.py"
    module.write_text(module.read_text() + "import os; os.system('id')  # INJECTED\n")
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_path / "idx")}
    subprocess.run([GIT, "read-tree", "HEAD"], cwd=repo, env=env, check=True)
    status = subprocess.run([GIT, "status", "--porcelain", "--untracked-files=no"], cwd=repo, env=env,
                            check=True, capture_output=True, text=True).stdout
    assert status == ""  # the filter hides it from status ...
    caught = refused(repo, tmp_path)
    assert caught.code == "source_dirty" and "raw bytes differ" in str(caught)  # ... not from the raw hash
    assert caught.names == ["src/pkg/module.py"]


def test_an_eol_attribute_cannot_launder_source_bytes(repo, tmp_path):
    (repo / ".gitattributes").write_text("*.py text eol=lf\n")
    commit(repo, "attributes")
    module = repo / "src" / "pkg" / "module.py"
    module.write_bytes(b"VALUE = 1\r\n")
    assert refused(repo, tmp_path).code == "source_dirty"


@pytest.mark.parametrize("flag, tag", [("--assume-unchanged", "h"), ("--skip-worktree", "S")])
def test_index_flags_that_hide_edits_refuse(repo, tmp_path, flag, tag):
    git(repo, "update-index", flag, "src/pkg/module.py")
    assert git(repo, "ls-files", "-v", "src/pkg/module.py").startswith(tag)
    caught = refused(repo, tmp_path)
    assert caught.code == "source_index_flags" and caught.names == ["src/pkg/module.py"]


# ── untracked and ignored files under a source path ────────────────────────

@pytest.mark.parametrize("name", [
    "src/pkg/new_module.py", "src/evil.pth", "src/llm_bawt.egg-info/PKG-INFO", "src/pkg/module.pyc",
    "src/pkg/sub/__init__.py",
])
def test_untracked_or_ignored_files_under_source_refuse(repo, tmp_path, name):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n")
    caught = refused(repo, tmp_path)
    assert caught.code == "source_untracked" and len(caught.names) == 1
    assert name.startswith(caught.names[0].rstrip("/"))


def test_a_newly_staged_file_is_still_untracked_against_head(repo, tmp_path):
    (repo / "src" / "pkg" / "added.py").write_text("x\n")
    git(repo, "add", "src/pkg/added.py")
    assert refused(repo, tmp_path).code == "source_untracked"


def _forge_pyc(repo):
    cache = repo / "src" / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / f"module.cpython-{sys.version_info[0]}{sys.version_info[1]}.pyc").write_bytes(b"forged")


def test_bytecode_next_to_source_refuses_unless_python_keeps_it_elsewhere(repo, tmp_path):
    """r1 repro: a forged ``__pycache__`` .pyc whose header matches the source
    would be imported instead of it."""
    _forge_pyc(repo)
    caught = refused(repo, tmp_path)
    assert caught.code == "source_bytecode_unsafe" and caught.names == ["src/pkg/__pycache__/"]
    assert check(repo, tmp_path, bytecode_ok=True)[0] == git(repo, "rev-parse", "HEAD").strip()


def test_bytecode_tolerance_never_covers_other_untracked_files(repo, tmp_path):
    _forge_pyc(repo)
    (repo / "src" / "pkg" / "stray.py").write_text("x\n")
    assert refused(repo, tmp_path, bytecode_ok=True).code == "source_untracked"


def _retire_package_with_cache(repo, name="oldpkg"):
    package = repo / "src" / name
    package.mkdir()
    (package / "__init__.py").write_text("")
    commit(repo, "add package")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-test.pyc").write_bytes(b"leftover")
    (package / "__init__.py").unlink()
    commit(repo, "retire package")
    assert f"src/{name}/" in git(
        repo, "ls-files", "--others", "--directory", "--no-empty-directory", "--", "src"
    ).splitlines()
    return package


def test_retired_package_cache_is_bytecode_not_stray_source(repo, tmp_path):
    _retire_package_with_cache(repo)
    caught = refused(repo, tmp_path)
    assert caught.code == "source_bytecode_unsafe" and caught.names == ["src/oldpkg/"]
    assert check(repo, tmp_path, bytecode_ok=True)[0] == git(repo, "rev-parse", "HEAD").strip()


def test_retired_package_with_ignored_stray_file_still_refuses(repo, tmp_path):
    (repo / ".gitignore").write_text("storage/\n__pycache__/\n*.pth\n*.egg-info/\n")
    commit(repo, "ignore pth")
    package = _retire_package_with_cache(repo)
    (package / "evil.pth").write_text("import os\n")
    caught = refused(repo, tmp_path, bytecode_ok=True)
    assert caught.code == "source_untracked" and caught.names == ["src/oldpkg/"]


def test_each_retired_directory_is_classified_independently(repo, tmp_path):
    _retire_package_with_cache(repo, "old")
    package = _retire_package_with_cache(repo, "oldpkg")
    nested = package / "sub"
    nested.mkdir()
    (nested / "evil.py").write_text("import os\n")
    caught = refused(repo, tmp_path, bytecode_ok=True)
    assert caught.code == "source_untracked" and caught.names == ["src/oldpkg/"]


def test_retired_directory_with_glob_characters_is_matched_literally(repo, tmp_path):
    _retire_package_with_cache(repo, "old[abc]")
    assert check(repo, tmp_path, bytecode_ok=True)[0] == git(repo, "rev-parse", "HEAD").strip()


# ── tree entries ────────────────────────────────────────────────────────────

def test_a_tracked_symlink_into_a_data_mount_refuses(repo, tmp_path):
    """r1 repro: ``src/pkg/module.py -> ../../storage/payload.py``. The committed
    link is clean, but the import runs bytes from an unverified data mount."""
    module = repo / "src" / "pkg" / "module.py"
    module.unlink()
    module.symlink_to("../../storage/payload.py")
    commit(repo, "link")
    caught = refused(repo, tmp_path)
    assert caught.code == "source_unsupported_entry" and caught.names == ["src/pkg/module.py"]


def test_a_symlink_at_a_mount_root_refuses(repo, tmp_path):
    (repo / "server.sh").unlink()
    (repo / "server.sh").symlink_to("storage/payload.py")
    commit(repo, "link")
    assert refused(repo, tmp_path).code == "source_unsupported_entry"


def test_a_gitlink_under_source_refuses(repo, tmp_path):
    (repo / "src" / "vendored").mkdir()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},src/vendored")
    git(repo, "commit", "-q", "-m", "gitlink")
    assert refused(repo, tmp_path).code == "source_unsupported_entry"


def test_a_source_path_with_no_tracked_files_refuses(repo, tmp_path):
    (repo / "empty").mkdir()
    caught = refused(repo, tmp_path, paths=["empty", "src"])
    assert caught.code == "source_mount_invalid" and caught.names == ["empty"]


def test_compose_files_must_be_tracked_regular_files(repo, tmp_path):
    (repo / "docker-compose.override.yml").write_text("services: {}\n")
    caught = refused(repo, tmp_path, compose_files=["docker-compose.yml", "docker-compose.override.yml"])
    assert caught.code == "source_compose_untracked" and caught.names == ["docker-compose.override.yml"]


def test_names_are_sanitised_and_capped(repo, tmp_path):
    for i in range(7):
        (repo / "src" / f"x{i}.py").write_text("x\n")
    (repo / "src" / "we ird‮.py").write_text("x\n")
    caught = refused(repo, tmp_path)
    assert caught.code == "source_untracked" and len(caught.names) <= 5 and "8 paths" in str(caught)
    assert "‮" not in str(caught)


def test_the_check_reads_the_checkout_without_writing_to_it(repo, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in repo.rglob("*")}
    index = (repo / ".git" / "index").read_bytes()
    check(repo, tmp_path)
    assert (repo / ".git" / "index").read_bytes() == index
    assert {p: p.stat().st_mtime_ns for p in repo.rglob("*")} == before


# ── spec validation ─────────────────────────────────────────────────────────

GOOD = {"source_repository": "bawthub/llm-bawt", "source_branch": "master",
        "source_mounts": ["/app/src", "/app/server.sh"], "data_mounts": ["/app/storage", "/var/lib/x"],
        "env_names": ["XAI_API_KEY", "LLM_BAWT_SERVICE_PORT"]}


@pytest.mark.parametrize("key, value", [
    ("source_repository", "github.com/bawthub/llm-bawt"), ("source_repository", "bawthub"),
    ("source_branch", "../main"), ("source_branch", "main.lock"), ("source_branch", "a//b"),
    ("source_branch", ".hidden"), ("source_branch", "refs/heads/../x"),
    ("source_mounts", []), ("source_mounts", ["/app/src", "/app/src"]), ("source_mounts", ["/app/src/"]),
    ("source_mounts", ["app/src"]), ("source_mounts", ["/app/src", "/app/src/llm_bawt"]),
    ("source_mounts", ["/"]), ("data_mounts", ["/app"]), ("data_mounts", ["/app/src/data"]),
    ("data_mounts", ["/app/src"]), ("data_mounts", "/app/storage"),
    ("env_names", "XAI_API_KEY"), ("env_names", ["xai_api_key"]), ("env_names", ["A", "A"]), ("env_names", [1]),
    ("env_names", ["_A"]), ("env_names", ["A" * 129]), ("env_names", [f"K{i}" for i in range(257)]),
    ("env_names", ["PATH"]), ("env_names", ["PYTHONPATH"]), ("env_names", ["HF_HOME"]), ("env_names", ["BASH_ENV"]),
    ("env_names", ["LC_ALL"]), ("env_names", ["NODE_OPTIONS"]), ("env_names", ["PERL5OPT"]),
])
def test_source_spec_is_validated(key, value):
    validate_source_spec(GOOD)
    with pytest.raises(ValueError):
        validate_source_spec({**GOOD, key: value})


def test_source_errors_are_compose_errors_with_fixed_codes():
    error = SourceStateError("source_dirty", ["a.py"], detail="raw bytes differ from HEAD")
    assert isinstance(error, ComposeError) and str(error) == "source_dirty: a.py (raw bytes differ from HEAD)"


# ── the Compose renderer still runs one sealed attach/start/wait ───────────

def test_the_renderer_runs_in_the_shared_sealed_container():
    from tests.test_ops_image_deploy import COMPOSE_FILES, PROJECT_DIR, RENDERER, FakeDocker, SPEC as DEPLOY_SPEC

    api = FakeDocker()
    ComposeRenderer(api, DEPLOY_SPEC, tag="r1").render({"project_dir": PROJECT_DIR, "compose_files": COMPOSE_FILES},
                                                       "ghcr.io/bawthub/frontend@sha256:" + "a" * 64)
    body = api.render_bodies[0]
    assert body["HostConfig"]["NetworkMode"] == "none" and body["HostConfig"]["ReadonlyRootfs"] is True
    # Only the empty noexec DOCKER_CONFIG tmpfs (TASK-1476 B); no scratch /tmp.
    assert body["HostConfig"]["Tmpfs"] == {RENDER_DOCKER_CONFIG: "rw,noexec,nosuid,nodev,size=1048576"}
    assert body["Labels"]["com.docker.compose.service"] == "compose-render"
    assert body["Labels"][OPS_JOB_LABEL] == "r1" and not api.renderers_left()
    with pytest.raises(ValueError):
        SealedContainer(api, RENDERER, role="Bad Role", tag="r1", labels={})
    with pytest.raises(ValueError):
        SealedContainer(api, RENDERER, role="compose-render", tag="R-1", labels={})


def test_a_session_maps_an_oversized_exec_to_output_too_large(tmp_path):
    class Big(GitDocker):
        def request(self, method, path, *, body=None, headers=None, raw=False):
            if path.startswith("/exec/") and path.endswith("/start"):
                raise RuntimeError("Docker HTTP 200 (raw stream) exceeded 1048576 bytes")
            return super().request(method, path, body=body, headers=headers, raw=raw)

    api = Big(tmp_path)
    sealed = SealedContainer(api, IMAGE, role="source-check", tag="t1", labels={})
    with pytest.raises(SealedRunError) as caught:
        with sealed.session(env=[], directory=str(tmp_path)) as run:
            run(["git", "status"])
    assert caught.value.kind == "output_too_large" and api.containers == {}


# ── the published branch head (app side) ───────────────────────────────────

class _Resp:
    def __init__(self, body: bytes):
        self.status, self.headers, self._body = 200, {}, body

    def read(self, _n=-1):
        return self._body


class _Forge:
    def __init__(self, body=None, status=200):
        self.body, self.status, self.seen = body, status, []

    def open(self, request, timeout):
        self.seen.append((request.full_url, request.get_header("Authorization")))
        if self.status != 200:
            raise urllib.error.HTTPError(request.full_url, self.status, "x", {}, None)
        return _Resp(self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode())


SHA = "ab" * 20
REF = {"ref": "refs/heads/master", "object": {"type": "commit", "sha": SHA}}


def test_branch_head_reads_exactly_that_ref():
    forge = _Forge(REF)
    assert GitHubReleaseVerifier("tok", opener=forge).branch_head("bawthub/llm-bawt", "master") == SHA
    assert forge.seen == [("https://api.github.com/repos/bawthub/llm-bawt/git/ref/heads/master", "Bearer tok")]
    forge = _Forge(REF)
    GitHubReleaseVerifier("", opener=forge).branch_head("bawthub/llm-bawt", "master")
    assert forge.seen[0][1] is None  # a public repository needs no token


@pytest.mark.parametrize("forge, repo, branch", [
    (_Forge(status=404), "bawthub/llm-bawt", "master"),
    (_Forge(status=500), "bawthub/llm-bawt", "master"),
    (_Forge(b"not json"), "bawthub/llm-bawt", "master"),
    (_Forge([REF]), "bawthub/llm-bawt", "master"),
    (_Forge({**REF, "ref": "refs/heads/master-old"}), "bawthub/llm-bawt", "master"),
    (_Forge({**REF, "object": {"type": "tag", "sha": SHA}}), "bawthub/llm-bawt", "master"),
    (_Forge({**REF, "object": {"type": "commit", "sha": SHA.upper()}}), "bawthub/llm-bawt", "master"),
    (_Forge(REF), "bawthub/llm-bawt", "../master"),
    (_Forge(REF), "bawthub", "master"),
], ids=["404", "500", "not-json", "list", "other-ref", "not-commit", "bad-sha", "bad-branch", "bad-repo"])
def test_branch_head_fails_closed(forge, repo, branch):
    with pytest.raises(ReleaseVerificationError):
        GitHubReleaseVerifier("tok", opener=forge).branch_head(repo, branch)


# ── the immutable worker image carries these stdlib-only modules ───────────

def test_worker_modules_import_standalone_with_the_standard_library_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "Dockerfile.ops-worker").read_text()
    copy = re.search(r"^COPY (.+?) /ops/$", dockerfile, re.M | re.S).group(1)
    modules = [p for p in copy.replace("\\\n", " ").split() if p.endswith(".py")]
    assert "src/llm_bawt/ops/source_state.py" in modules and "src/llm_bawt/ops/sealed_container.py" in modules
    ops = tmp_path / "ops"
    ops.mkdir()
    for module in modules:
        shutil.copy(root / module, ops)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import worker, source_state, sealed_container, "
            "app_config_reconcile; assert 'llm_bawt' not in sys.modules")
    proc = subprocess.run([sys.executable, "-I", "-c", code, str(ops)], capture_output=True, text=True,
                          check=False, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]


# ── TASK-1831: a Compose-only plan (the guarded tagged reconcile) ───────────

def test_a_compose_only_plan_binds_the_compose_file_and_the_whole_tree(repo, tmp_path):
    from llm_bawt.ops.source_state import compose_source_plan
    only = compose_source_plan({"project_dir": str(repo), "compose_files": [f"{repo}/docker-compose.yml"]})
    assert only == {"paths": [], "compose_files": ["docker-compose.yml"], "bytecode_ok": True}
    (repo / "notes.txt").write_text("scratch\n")  # untracked elsewhere: not its concern
    assert check(repo, tmp_path, **only)[0] == git(repo, "rev-parse", "HEAD").strip()
    (repo / "docker-compose.yml").write_text("services: {redis: {}}\n")
    assert refused(repo, tmp_path, **only).code == "source_dirty"
    git(repo, "checkout", "--", "docker-compose.yml")
    (repo / "src" / "pkg" / "module.py").write_text("VALUE = 2\n")  # any tracked change, as for the app
    assert refused(repo, tmp_path, **only).code == "source_dirty"
