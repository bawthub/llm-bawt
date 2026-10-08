"""Approval-time bindings for digest deploy/rollback operations (TASK-997).

``prepare_invocation`` calls :func:`bind_image_invocation` BEFORE an approval
snapshot is persisted. The result is embedded in the snapshot (hashed, no
secrets), so the operator approves an exact, verified release:

* deploy  — a completed, successful ``workflow_dispatch`` run of the pinned
  workflow on the canonical branch, whose receipt artifact binds
  version → source SHA → immutable digest, and whose git tag resolves to that
  SHA; plus a compare-and-swap on the image the target runs right now. The
  receipt is read from the run's LATEST attempt (``release-receipt-<attempt>``,
  TASK-1030) with a fallback to the pre-TASK-1030 single ``release-receipt``
  artifact. Only a ``complete`` receipt is deployable.
* rollback — a SUCCEEDED deploy job for the same target, its recorded
  last-known-good image, and a CAS that the target still runs what it deployed.

The worker independently re-verifies the pulled bytes (RepoDigest + release
labels + in-container /api/health). Both GitHub credentials live in the
encrypted CredentialStore (providers ``github-release`` / ``ghcr-pull``).
"""
from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .image_deploy import DEPLOY_ACTION, ROLLBACK_ACTION, parse_deployment

RECEIPT_SCHEMA = "bawthub.release-receipt/v1"
RECEIPT_ARTIFACT = "release-receipt"
DEPLOYABLE_RECEIPT_STATUSES = ("complete",)
# Public deploy input is a durable release ID. The four identity fields below
# are derived from the verified release row, never accepted from the caller.
_DEPLOY_ARGS = {"release_run_id": r"[0-9a-f]{32}"}
_VERIFIER_ARGS = {"workflow_run_id": r"[0-9]{1,20}", "digest": r"sha256:[0-9a-f]{64}",
                  "source_sha": r"[0-9a-f]{40}", "version": r"[0-9]+\.[0-9]+\.[0-9]+"}
_ROLLBACK_ARGS = {"deploy_job_id": r"[0-9a-f]{32}"}
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


class ReleaseVerificationError(ValueError):
    pass


def _require_args(args: dict, patterns: dict) -> dict:
    """Enforced in code: the operator-editable schema is not the only guard."""
    if set(args) != set(patterns):
        raise ReleaseVerificationError(f"arguments must be exactly {sorted(patterns)}")
    for key, pattern in patterns.items():
        if not isinstance(args[key], str) or not re.fullmatch(pattern, args[key]):
            raise ReleaseVerificationError(f"argument {key} is invalid")
    return args


class ReleaseVerifier(ABC):
    @abstractmethod
    def verify(self, spec: dict, args: dict) -> dict:
        """Return the verified release binding or raise ReleaseVerificationError."""


def _stored_release_token() -> str:
    from ..service.providers.github_deploy import GitHubReleaseAdapter
    from ..utils.config import Config
    try:
        return GitHubReleaseAdapter(Config()).api_key() or ""
    except Exception as exc:  # noqa: BLE001 - DB/key trouble must surface as "unverified"
        raise ReleaseVerificationError(f"release verification token unavailable: {type(exc).__name__}") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class GitHubReleaseVerifier(ReleaseVerifier):
    """Read-only GitHub API checks. Token: fine-grained, Actions+Contents READ
    on the pinned repository only, stored as the ``github-release`` provider
    credential (CredentialStore). Resolved per verification, so a reconnect
    takes effect without a restart."""

    API = "https://api.github.com"
    MAX_ARTIFACT_BYTES = 1024 * 1024

    def __init__(self, token: str | None = None, *, opener=None):
        self._fixed_token = token
        self._token = ""
        self._opener = opener or urllib.request.build_opener(_NoRedirect)

    def _fetch(self, url: str, *, auth: bool = True, limit: int = MAX_ARTIFACT_BYTES):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "llm-bawt-ops-release-verifier"}
        if auth:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._opener.open(urllib.request.Request(url, headers=headers), timeout=20)
            status, location, body = response.status, response.headers.get("Location"), response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            status, location, body = exc.code, exc.headers.get("Location"), b""
        except OSError as exc:
            raise ReleaseVerificationError(f"GitHub unreachable: {type(exc).__name__}") from exc
        if len(body) > limit:
            raise ReleaseVerificationError("GitHub response too large")
        return status, location, body

    def _json(self, path: str) -> dict:
        status, _location, body = self._fetch(self.API + path)
        if status != 200:
            raise ReleaseVerificationError(f"GitHub GET {path.split('?')[0]} returned HTTP {status}")
        return json.loads(body)

    def _artifact_receipt(self, repo: str, run_id: str, name: str) -> dict | None:
        """Return the named receipt, ``None`` if the run has no such artifact."""
        listing = self._json(f"/repos/{repo}/actions/runs/{run_id}/artifacts?name={name}")
        live = [a for a in listing.get("artifacts", []) if a.get("name") == name and not a.get("expired")]
        if not live:
            return None
        if len(live) != 1:
            raise ReleaseVerificationError(f"run must carry exactly one unexpired {name} artifact")
        status, location, body = self._fetch(live[0]["archive_download_url"])
        if status in (301, 302, 303, 307, 308) and location:
            # Signed blob URL: never forward the GitHub token to blob storage.
            status, _loc, body = self._fetch(location, auth=False)
        if status != 200:
            raise ReleaseVerificationError(f"receipt artifact download returned HTTP {status}")
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                return json.loads(archive.read("release.json"))
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ReleaseVerificationError("receipt artifact has no valid release.json") from exc

    def latest_receipt(self, repo: str, run_id: str, run_attempt) -> dict:
        """Receipt of the run's latest attempt; legacy single artifact fallback.

        A re-run ("Re-run failed jobs") uploads ``release-receipt-<attempt>``,
        so an earlier attempt's partial receipt can never be mistaken for the
        current one.
        """
        attempt = str(run_attempt or "").strip()
        if attempt.isdigit():
            receipt = self._artifact_receipt(repo, run_id, f"{RECEIPT_ARTIFACT}-{attempt}")
            if receipt is not None:
                if str(receipt.get("workflow_run_attempt", "")) != attempt:
                    raise ReleaseVerificationError(
                        f"release receipt workflow_run_attempt is {receipt.get('workflow_run_attempt')!r}, "
                        f"expected {attempt!r}")
                return receipt
        receipt = self._artifact_receipt(repo, run_id, RECEIPT_ARTIFACT)
        if receipt is None:
            raise ReleaseVerificationError(
                f"run must carry exactly one unexpired {RECEIPT_ARTIFACT}-{attempt or '<attempt>'} "
                f"(or legacy {RECEIPT_ARTIFACT}) artifact")
        return receipt

    def _tag_commit(self, repo: str, tag: str) -> str:
        obj = self._json(f"/repos/{repo}/git/ref/tags/{tag}")["object"]
        if obj.get("type") == "tag":
            obj = self._json(f"/repos/{repo}/git/tags/{obj['sha']}")["object"]
        if obj.get("type") != "commit":
            raise ReleaseVerificationError(f"tag {tag} does not point at a commit")
        return obj["sha"]

    def verify(self, spec, args):
        _require_args(args, _VERIFIER_ARGS)
        self._token = self._fixed_token if self._fixed_token is not None else _stored_release_token()
        if not self._token:
            raise ReleaseVerificationError(
                "release verification token not connected (Providers: GitHub release verification)")
        repo, run_id = spec["github_repository"], args["workflow_run_id"]
        run = self._json(f"/repos/{repo}/actions/runs/{run_id}")
        checks = {
            "status": (run.get("status"), "completed"),
            "conclusion": (run.get("conclusion"), "success"),
            "workflow path": ((run.get("path") or "").split("@")[0], spec["workflow_path"]),
            "branch": (run.get("head_branch"), spec["canonical_branch"]),
            "event": (run.get("event"), "workflow_dispatch"),
            "repository": ((run.get("repository") or {}).get("full_name"), repo),
        }
        for label, (got, want) in checks.items():
            if got != want:
                raise ReleaseVerificationError(f"workflow run {run_id} {label} is {got!r}, expected {want!r}")
        receipt = self.latest_receipt(repo, run_id, run.get("run_attempt"))
        if receipt.get("status") not in DEPLOYABLE_RECEIPT_STATUSES:
            raise ReleaseVerificationError(
                f"release receipt status is {receipt.get('status')!r}, expected one of {DEPLOYABLE_RECEIPT_STATUSES}")
        tag = f"v{args['version']}"
        expected = {"schema": RECEIPT_SCHEMA, "repository": repo,
                    "workflow_run_id": run_id, "source_sha": args["source_sha"], "version": args["version"],
                    "tag": tag, "digest": args["digest"], "image_repository": spec["image_repository"],
                    "base_sha": run.get("head_sha")}
        for key, want in expected.items():
            if receipt.get(key) != want:
                raise ReleaseVerificationError(f"release receipt {key} is {receipt.get(key)!r}, expected {want!r}")
        if self._tag_commit(repo, tag) != args["source_sha"]:
            raise ReleaseVerificationError(f"tag {tag} does not resolve to source_sha")
        return {"github_repository": repo, "workflow_run_id": run_id,
                "workflow_run_attempt": str(receipt.get("workflow_run_attempt", "")),
                "workflow_run_url": run.get("html_url", ""), "workflow_path": spec["workflow_path"],
                "trigger_sha": run.get("head_sha"), "source_sha": args["source_sha"], "version": args["version"],
                "tag": tag, "digest": args["digest"], "image_repository": spec["image_repository"],
                "image_ref": f"{spec['image_repository']}@{args['digest']}",
                "receipt_status": receipt["status"],
                "verified_at": datetime.now(timezone.utc).isoformat()}


def rollback_binding(store, spec: dict, args: dict) -> dict:
    _require_args(args, _ROLLBACK_ARGS)
    job = store.get_job(args["deploy_job_id"])
    if job is None or job.state != "succeeded" or not job.invocation_snapshot_json:
        raise ReleaseVerificationError("rollback requires a SUCCEEDED deploy job")
    deployed_spec = json.loads(job.invocation_snapshot_json).get("spec", {})
    if deployed_spec.get("action") != DEPLOY_ACTION or deployed_spec.get("container_name") != spec["container_name"]:
        raise ReleaseVerificationError("job is not a deploy of this rollback target")
    record = parse_deployment(job.output_tail)
    if not record or record.get("phase") != "done" or record.get("action") != "deploy":
        raise ReleaseVerificationError("deploy job has no complete deployment record")
    previous, deployed = record.get("previous") or {}, record.get("deployed") or {}
    if not _IMAGE_ID.fullmatch(previous.get("image_id") or "") or not _IMAGE_ID.fullmatch(deployed.get("image_id") or ""):
        raise ReleaseVerificationError("deployment record lacks image identities")
    return {"deploy_job_id": job.id, "from_image_id": deployed["image_id"], "from_image_ref": deployed.get("image_ref", ""),
            "to_image_id": previous["image_id"], "to_image_ref": previous.get("image_ref", ""),
            "to_release": previous.get("release")}


def bind_image_invocation(*, spec: dict, args: dict, store, releases, executor,
                          verifier: ReleaseVerifier) -> dict:
    """Bind a verified release ID to the existing immutable-image worker."""
    current = executor.inspect_target_image(spec)
    if spec["action"] == DEPLOY_ACTION:
        _require_args(args, _DEPLOY_ARGS)
        try:
            binding = releases.verified_binding(args["release_run_id"])
        except ValueError as exc:
            raise ReleaseVerificationError(str(exc)) from exc
        if (binding["github_repository"] != spec["github_repository"]
                or binding["workflow_path"] != spec["workflow_path"]
                or binding["image_repository"] != spec["image_repository"]):
            raise ReleaseVerificationError("release run is bound to another deploy target")
        derived = {key: binding[key] for key in _VERIFIER_ARGS}
        release = verifier.verify(spec, derived)
        for key in ("workflow_run_id", "workflow_run_attempt", "source_sha", "version",
                    "tag", "digest", "image_repository", "trigger_sha"):
            if str(release.get(key)) != str(binding.get(key)):
                raise ReleaseVerificationError(f"GitHub verification differs from release run: {key}")
        release["release_run_id"] = args["release_run_id"]
        release["expected_current_image_id"] = current
        return {"release": release}
    if spec["action"] == ROLLBACK_ACTION:
        binding = rollback_binding(store, spec, args)
        if current != binding["from_image_id"]:
            raise ReleaseVerificationError("target no longer runs the image that deploy installed; refusing rollback")
        return {"rollback": binding}
    raise ReleaseVerificationError(f"not an image action: {spec['action']}")


def binding_matches_args(snapshot: dict) -> bool:
    """Cheap consistency check for _verify_snapshot (no network)."""
    action, args = snapshot["spec"].get("action"), snapshot["resolved_args"]
    if action == DEPLOY_ACTION:
        release = snapshot.get("release") or {}
        return ("rollback" not in snapshot and set(args) == set(_DEPLOY_ARGS)
                and release.get("release_run_id") == args.get("release_run_id"))
    if action == ROLLBACK_ACTION:
        return "release" not in snapshot and (snapshot.get("rollback") or {}).get("deploy_job_id") == args.get("deploy_job_id")
    return "release" not in snapshot and "rollback" not in snapshot
