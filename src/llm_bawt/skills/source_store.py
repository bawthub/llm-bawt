"""On-disk skill sources: fetch, validate, pin and activate (TASK-2534).

Skills are installed and updated independently of the app and bridge images.
Each configured source (a Git repository, branch or tag, optional subdirectory)
is fetched into staging, validated, and copied to an immutable revision
directory. The bridges read one effective tree, ``<root>/current``, a symlink to
a generation built from every source's active revision plus the image's bundled
skills as the lowest-priority fallback. Activation swaps that symlink with one
``rename``, so a reader sees either the old tree or the new one, never a mix.

Layout under the store root (a volume outside the containers)::

    sources/<id>/revs/<revision>/<skill>/SKILL.md   immutable, one per pin
    generations/<n>/<skill>/...                      effective trees
    current -> generations/<n>

Fetching never runs repository code: Git hooks, templates and non-HTTPS/SSH
transports are disabled, submodules and LFS are not fetched, and the fetched
tree is read as data. A symlink anywhere in a fetched skill is refused.
Credentials reach Git through ``GIT_CONFIG_*`` environment entries, never argv.

A source can also be a published *archive*: an HTTPS manifest naming a tarball
and its sha256. The built-in BawtHub source uses this so tenants install the
reviewed product skills without any access to the private repository. The
archive is verified against the manifest hash and extracted as data (no links,
devices or paths outside the target); its revision is the hash's first 40 hex.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from agent_bridge.skill_sources import SKILL_FILE, split_frontmatter

REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 25_000_000
MAX_FILES = 5_000
KEEP_GENERATIONS = 4
OWNERS_FILE = ".sources.json"
LAYERS_FILE = ".layers.json"
FETCH_TIMEOUT = 180.0
MAX_ARCHIVE_BYTES = 30_000_000
MAX_EXPANDED_BYTES = 2 * MAX_TOTAL_BYTES
MAX_MANIFEST_BYTES = 64_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceError(Exception):
    """A refused or failed source operation; ``code`` is stable for callers."""

    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class GitSpec:
    url: str
    ref: str | None = None  # branch or tag; None = the remote's default branch
    subdir: str = ""
    github: bool = False


@dataclass(frozen=True)
class ArchiveSpec:
    """A published skills archive: ``url`` is its JSON manifest."""

    url: str
    subdir: str = ""
    github: bool = False  # never; keeps the credential check uniform
    # Trusted publisher keys (base64 raw Ed25519). When set, ``<url>.sig`` must
    # be a valid signature over the exact manifest bytes by one of them.
    keys: tuple[str, ...] = ()


Spec = GitSpec | ArchiveSpec


@dataclass(frozen=True)
class Fetched:
    revision: str
    skills: dict[str, str]  # name -> content digest


@dataclass
class Layer:
    """One input to a generation, highest priority first."""

    source_id: str
    path: Path
    skills: tuple[str, ...] = field(default_factory=tuple)


def validate_subdir(subdir: str | None) -> str:
    value = (subdir or "").strip()
    if value.startswith("/"):
        raise SourceError(400, "invalid_subdir", "Subdirectory must be relative to the repository root.")
    value = value.rstrip("/")
    if not value:
        return ""
    if len(value) > 255 or "\\" in value or "\x00" in value:
        raise SourceError(400, "invalid_subdir", "Subdirectory must be a relative POSIX path.")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
        raise SourceError(400, "invalid_subdir", "Subdirectory must be a relative path inside the repository.")
    return "/".join(parts)


def git_spec(url: str, ref: str | None, subdir: str | None) -> GitSpec:
    """Validate a user-supplied source with the workspace clone's rules."""
    from agent_bridge.workspace import WorkspaceError, clone_input

    body: dict[str, object] = {"url": url, "name": "skills"}
    if ref:
        body["branch"] = ref
    try:
        item = clone_input(body)
    except WorkspaceError as exc:
        raise SourceError(400, "invalid_source", "Invalid repository URL or branch/tag.") from exc
    return GitSpec(item.url, item.branch, validate_subdir(subdir), item.github)


def _allow_http() -> bool:
    # Disposable fixtures only (a LAN test server); real sources are HTTPS.
    return os.environ.get("SKILL_SOURCES_ALLOW_HTTP") == "1"


def _archive_url(url: str) -> str:
    parsed = urllib.parse.urlsplit((url or "").strip())
    schemes = {"https", "http"} if _allow_http() else {"https"}
    if parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password:
        raise SourceError(400, "invalid_source", "Archive sources need a credential-free HTTPS URL.")
    return urllib.parse.urlunsplit(parsed)


def archive_spec(url: str, subdir: str | None, keys: Iterable[str] = ()) -> ArchiveSpec:
    return ArchiveSpec(_archive_url(url), validate_subdir(subdir), keys=tuple(k for k in keys if k))


def verify_manifest(body: bytes, signature: bytes, keys: Iterable[str]) -> bool:
    """True when ``signature`` (base64 Ed25519) signs ``body`` under any trusted key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        raw_sig = base64.b64decode(signature.strip(), validate=True)
    except ValueError:
        return False
    for key in keys:
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(key, validate=True)).verify(raw_sig, body)
            return True
        except (InvalidSignature, ValueError):
            continue
    return False


def _skill_digest(skill_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in skill_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(skill_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def scan_skills(tree: Path) -> dict[str, str]:
    """Validate a fetched tree as data and return ``{skill: digest}``.

    A skill is a top-level directory with a ``SKILL.md`` whose frontmatter
    parses. Other top-level entries (README, dot directories) are ignored.
    Refuses symlinks, oversized files or trees, and non-UTF-8 ``SKILL.md``.
    """
    skills: dict[str, str] = {}
    files = 0
    total = 0
    for child in sorted(tree.iterdir()):
        if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
            continue
        if not (child / SKILL_FILE).is_file():
            continue
        if not REF.fullmatch(child.name):
            raise SourceError(422, "invalid_skill_name", f"Skill folder {child.name!r} is not a valid skill name.")
        for path in child.rglob("*"):
            if path.is_symlink():
                raise SourceError(422, "symlink_refused",
                                  f"{path.relative_to(tree).as_posix()} is a symlink; skill sources may not contain links.")
            if not path.is_file():
                continue
            size = path.stat().st_size
            files += 1
            total += size
            if size > MAX_FILE_BYTES:
                raise SourceError(422, "file_too_large",
                                  f"{path.relative_to(tree).as_posix()} is {size} bytes (limit {MAX_FILE_BYTES}).")
            if files > MAX_FILES or total > MAX_TOTAL_BYTES:
                raise SourceError(422, "source_too_large", "The source is larger than a skill source may be.")
        try:
            text = (child / SKILL_FILE).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SourceError(422, "invalid_skill", f"{child.name}/SKILL.md is not UTF-8 text.") from exc
        meta, _ = split_frontmatter(text)
        if not meta:
            raise SourceError(422, "invalid_skill", f"{child.name}/SKILL.md has no readable frontmatter.")
        skills[child.name] = _skill_digest(child)
    if not skills:
        raise SourceError(422, "no_skills", "No skills found (expected <name>/SKILL.md folders).")
    return skills


def diff_skills(old: dict[str, str] | None, new: dict[str, str]) -> dict[str, list[str]]:
    old = old or {}
    return {
        "added": sorted(set(new) - set(old)),
        "changed": sorted(name for name in set(new) & set(old) if new[name] != old[name]),
        "removed": sorted(set(old) - set(new)),
        "unchanged": sorted(name for name in set(new) & set(old) if new[name] == old[name]),
    }


class GitFetcher:
    """Hook-free, transport-restricted Git reads. ``token`` is an HTTPS secret."""

    def __init__(self, timeout: float = FETCH_TIMEOUT):
        self.timeout = timeout

    @staticmethod
    def _env(token: str | None) -> dict[str, str]:
        # Only the process basics are inherited: no GIT_* variable (trace,
        # alternates, exec paths) and no system or global Git config, so no
        # inherited filter driver, URL rewrite or credential helper applies.
        env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR",
                                                "SSL_CERT_FILE", "SSL_CERT_DIR") if key in os.environ}
        env.update({
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_SSH_COMMAND": "ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "XDG_CONFIG_HOME": "/nonexistent",
            "GIT_ATTR_NOSYSTEM": "1",
        })
        entries = [
            ("protocol.allow", "never"),
            ("protocol.https.allow", "always"),
            ("protocol.ssh.allow", "always"),
            ("core.hooksPath", "/dev/null"),
            ("core.attributesFile", "/dev/null"),
            ("submodule.recurse", "false"),
            ("credential.helper", ""),
            # A credentialed fetch never follows a redirect; others only the initial one.
            ("http.followRedirects", "false" if token else "initial"),
        ]
        if token:
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            entries.append(("http.https://github.com/.extraHeader", f"Authorization: Basic {basic}"))
        env["GIT_CONFIG_COUNT"] = str(len(entries))
        for index, (key, value) in enumerate(entries):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env

    def _run(self, args: list[str], *, token: str | None, cwd: Path | None = None) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=cwd, env=self._env(token), capture_output=True,
                text=True, timeout=self.timeout, check=False, start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise SourceError(504, "fetch_timeout", "Git timed out reading the source.") from exc
        except OSError as exc:
            raise SourceError(503, "git_unavailable", "Git is not available.") from exc
        if result.returncode:
            # Git's stderr can echo URLs; never return it to callers verbatim.
            raise SourceError(502, "fetch_failed", "Git could not read the source. Check URL, branch/tag and access.")
        return result.stdout

    def remote_revision(self, spec: GitSpec, token: str | None) -> str:
        """The revision the source's branch/tag points at now, without fetching content."""
        target = spec.ref or "HEAD"
        out = self._run(["ls-remote", "--", spec.url, target, f"{target}^{{}}"], token=token)
        rows = [line.split("\t") for line in out.splitlines() if "\t" in line]
        peeled = [sha for sha, name in rows if name.endswith("^{}")]
        candidates = peeled or [sha for sha, _ in rows]
        if not candidates or not REVISION.fullmatch(candidates[0]):
            raise SourceError(404, "ref_not_found", "That branch or tag does not exist in the repository.")
        return candidates[0]

    def fetch(self, spec: GitSpec, token: str | None, into: Path) -> str:
        """Shallow-clone ``spec`` into ``into`` (must not exist); return the commit."""
        args = ["clone", "--template=", "--quiet", "--depth", "1", "--no-tags",
                "--single-branch", "--no-recurse-submodules"]
        if spec.ref:
            args += ["--branch", spec.ref]
        self._run([*args, "--", spec.url, str(into)], token=token)
        revision = self._run(["rev-parse", "HEAD"], token=None, cwd=into).strip()
        if not REVISION.fullmatch(revision):
            raise SourceError(502, "fetch_failed", "Git returned an unexpected revision.")
        return revision


def layer_map(layers: Iterable[Layer]) -> dict[str, str]:
    return {layer.source_id: layer.path.name for layer in layers}


def _contained_subdir(checkout: Path, subdir: str) -> Path:
    """``checkout/subdir`` with no symlink anywhere on the path and inside the checkout."""
    tree = checkout
    for part in PurePosixPath(subdir).parts if subdir else ():
        tree = tree / part
        if tree.is_symlink():
            raise SourceError(422, "symlink_refused", "The skills folder path may not go through a link.")
    if not tree.is_dir():
        raise SourceError(422, "invalid_subdir", "That subdirectory does not exist in the repository.")
    if not tree.resolve().is_relative_to(checkout.resolve()):
        raise SourceError(422, "invalid_subdir", "The skills folder must be inside the repository.")
    return tree


class _NoDowngrade(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _archive_url(newurl)  # a redirect must satisfy the same scheme rules
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ArchiveFetcher:
    """Reads a manifest ``{"archive": url, "sha256": hex}`` and its verified tarball."""

    def __init__(self, timeout: float = FETCH_TIMEOUT):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoDowngrade())

    def _get(self, url: str, limit: int) -> bytes:
        try:
            with self._opener.open(urllib.request.Request(url, headers={"User-Agent": "bawthub-skills"}),
                                   timeout=self.timeout) as response:
                body = response.read(limit + 1)
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(502, "fetch_failed", "Could not download the skills archive.") from exc
        if len(body) > limit:
            raise SourceError(413, "too_large", "The skills archive is too large.")
        return body

    def _manifest(self, spec: ArchiveSpec) -> tuple[str, str]:
        body = self._get(spec.url, MAX_MANIFEST_BYTES)
        if spec.keys:
            # The signature binds version, archive and digest to the publisher;
            # a checksum served beside the archive alone authenticates nothing.
            signature = self._get(f"{spec.url}.sig", 4096)
            if not verify_manifest(body, signature, spec.keys):
                raise SourceError(502, "untrusted_manifest", "The archive manifest is not signed by a trusted publisher.")
        try:
            manifest = json.loads(body)
            digest = str(manifest["sha256"]).lower()
            version = str(manifest.get("version") or "")
            archive = urllib.parse.urljoin(spec.url, str(manifest["archive"]))
        except Exception as exc:
            raise SourceError(502, "invalid_manifest", "The archive manifest is not valid.") from exc
        if not SHA256.fullmatch(digest):
            raise SourceError(502, "invalid_manifest", "The archive manifest has no valid sha256.")
        if spec.keys and (not REF.fullmatch(version) or version not in urllib.parse.urlsplit(archive).path):
            # Signed releases are immutable, versioned files, never a moving "latest".
            raise SourceError(502, "invalid_manifest", "A signed manifest must name a versioned archive.")
        return _archive_url(archive), digest

    def remote_revision(self, spec: ArchiveSpec, token: str | None) -> str:
        return self._manifest(spec)[1][:40]

    def fetch(self, spec: ArchiveSpec, token: str | None, into: Path) -> str:
        archive, digest = self._manifest(spec)
        body = self._get(archive, MAX_ARCHIVE_BYTES)
        if hashlib.sha256(body).hexdigest() != digest:
            raise SourceError(502, "hash_mismatch", "The downloaded archive does not match its manifest.")
        into.mkdir(parents=True)
        blob = into.parent / "archive.tar.gz"
        blob.write_bytes(body)
        try:
            entries = expanded = 0
            with tarfile.open(blob, "r:*") as bundle:
                # Headers are read and limits enforced one entry at a time, so
                # nothing past a limit is ever written.
                for member in bundle:
                    entries += 1
                    expanded += max(member.size, 0)
                    if entries > MAX_FILES * 2 or expanded > MAX_EXPANDED_BYTES:
                        raise SourceError(413, "too_large", "The skills archive is too large when unpacked.")
                    if not (member.isfile() or member.isdir()):
                        raise SourceError(422, "symlink_refused", "The archive may only contain files and folders.")
                    bundle.extract(member, into, filter="data")
        except SourceError:
            raise
        except (tarfile.TarError, OSError, ValueError) as exc:
            raise SourceError(422, "invalid_archive", "The skills archive could not be unpacked safely.") from exc
        finally:
            blob.unlink(missing_ok=True)
        return digest[:40]


class SourceStore:
    """The filesystem half of skill sources. One writer at a time (``locked``)."""

    def __init__(self, root: Path | None, *, fallback: Iterable[Path] = (), fetcher: GitFetcher | None = None,
                 archive_fetcher: ArchiveFetcher | None = None):
        # None = not configured: sources are disabled, nothing is written anywhere.
        self.enabled = root is not None
        self.root = root if root is not None else Path("/nonexistent/skill-sources")
        self.fallback = tuple(fallback)
        self.fetcher = fetcher or GitFetcher()
        self.archive_fetcher = archive_fetcher or ArchiveFetcher()

    def fetcher_for(self, spec: Spec):
        return self.archive_fetcher if isinstance(spec, ArchiveSpec) else self.fetcher

    @property
    def current(self) -> Path:
        return self.root / "current"

    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            return os.access(self.root, os.W_OK)
        except OSError:
            return False

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def revision_dir(self, source_id: str, revision: str) -> Path:
        if not SOURCE_ID.fullmatch(source_id) or not REVISION.fullmatch(revision):
            raise SourceError(400, "invalid_revision", "Invalid source or revision.")
        return self.root / "sources" / source_id / "revs" / revision

    def stage(self, source_id: str, spec: Spec, token: str | None) -> Fetched:
        """Fetch, validate and store one revision. Idempotent per revision."""
        staging_root = self.root / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f"{source_id}-", dir=staging_root))
        try:
            checkout = work / "checkout"
            revision = self.fetcher_for(spec).fetch(spec, token, checkout)
            tree = _contained_subdir(checkout, spec.subdir)
            skills = scan_skills(tree)
            target = self.revision_dir(source_id, revision)
            if not target.is_dir():
                partial = work / "rev"
                partial.mkdir()
                for name in skills:
                    shutil.copytree(tree / name, partial / name, symlinks=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    partial.rename(target)
                except OSError:
                    if not target.is_dir():
                        raise
            return Fetched(revision, skills)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def skills_at(self, source_id: str, revision: str) -> dict[str, str]:
        return scan_skills(self.revision_dir(source_id, revision))

    def build(self, layers: list[Layer]) -> dict[str, str]:
        """Build and atomically activate a generation; return ``{skill: source_id}``."""
        new = self.prepare(layers)
        self.activate(new)
        return json.loads((new / OWNERS_FILE).read_text())

    def prepare(self, layers: list[Layer]) -> Path:
        """Build a generation without changing ``current``.

        A source layer whose revision files are missing fails the build rather
        than silently dropping that source's skills.
        """
        generations = self.root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        new = Path(tempfile.mkdtemp(prefix="gen-", dir=generations))
        owners: dict[str, str] = {}
        try:
            for layer in layers:
                if not layer.path.is_dir():
                    raise SourceError(500, "revision_missing",
                                      f"The files for source {layer.source_id} are missing; check it for updates again.")
            all_layers = [*layers, *(Layer("bundled", path) for path in self.fallback)]
            for layer in all_layers:
                if not layer.path.is_dir():
                    continue
                for child in sorted(layer.path.iterdir()):
                    name = child.name
                    if name in owners or not REF.fullmatch(name) or child.is_symlink():
                        continue
                    if not (child / SKILL_FILE).is_file():
                        continue
                    shutil.copytree(child, new / name, symlinks=True)
                    owners[name] = layer.source_id
            # Dot files are not skills; the catalog reads this for attribution.
            (new / OWNERS_FILE).write_text(json.dumps(owners, sort_keys=True))
            (new / LAYERS_FILE).write_text(json.dumps(layer_map(layers), sort_keys=True))
            new.chmod(0o755)
        except BaseException:
            shutil.rmtree(new, ignore_errors=True)
            raise
        return new

    def activate(self, new: Path) -> None:
        """Atomically make a prepared generation the effective tree."""
        generations = self.root / "generations"
        link = generations / f".current-{new.name}"
        link.unlink(missing_ok=True)
        link.symlink_to(Path("generations") / new.name)
        os.replace(link, self.current)  # atomic swap of the effective tree; the commit point
        try:  # best effort: a leftover generation is harmless, a failed swap is not hidden
            self._prune(generations, keep=new.name)
        except OSError:
            pass

    def discard(self, new: Path) -> None:
        """Remove a prepared generation, never the one ``current`` points at."""
        try:
            if self.current.resolve() == new.resolve():
                return
        except OSError:
            return
        shutil.rmtree(new, ignore_errors=True)

    def layers(self) -> dict[str, str] | None:
        """``{source_id: revision}`` the active tree was built from; None when unknown."""
        try:
            data = json.loads((self.current / LAYERS_FILE).read_text())
        except (OSError, ValueError):
            return None
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else None

    def owners(self) -> dict[str, str]:
        """``{skill: source_id}`` of the active tree; empty when there is none."""
        try:
            data = json.loads((self.current / OWNERS_FILE).read_text())
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _prune(self, generations: Path, keep: str) -> None:
        olds = sorted(
            (p for p in generations.iterdir() if p.is_dir() and not p.is_symlink() and p.name != keep),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for old in olds[KEEP_GENERATIONS - 1:]:
            shutil.rmtree(old, ignore_errors=True)

    def remove_source(self, source_id: str) -> None:
        if not SOURCE_ID.fullmatch(source_id):
            raise SourceError(400, "invalid_source", "Invalid source id.")
        shutil.rmtree(self.root / "sources" / source_id, ignore_errors=True)

    def prune_revisions(self, source_id: str, keep: Iterable[str]) -> None:
        revs = self.root / "sources" / source_id / "revs"
        keep_set = set(keep)
        if revs.is_dir():
            for path in revs.iterdir():
                if path.name not in keep_set:
                    shutil.rmtree(path, ignore_errors=True)
