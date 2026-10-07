"""Typed GitHub Actions gateway for durable BawtHub release orchestration."""
from __future__ import annotations

import io
import json
import re
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from ..utils.config import Config

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SEMVER_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class GitHubWorkflowError(RuntimeError):
    """A classified GitHub transport or contract failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class GitHubWorkflowGateway(ABC):
    """Narrow remote-release API. No generic repository mutation methods."""

    @abstractmethod
    def resolve_branch_head(self, repository: str, branch: str) -> str: ...

    @abstractmethod
    def latest_semver_tag(self, repository: str) -> dict[str, str] | None: ...

    @abstractmethod
    def compare(self, repository: str, base: str, head: str) -> dict[str, Any]: ...

    @abstractmethod
    def dispatch_workflow(
        self,
        repository: str,
        workflow_path: str,
        branch: str,
        inputs: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def find_correlated_runs(
        self,
        repository: str,
        workflow_path: str,
        branch: str,
        release_request_id: str,
        *,
        created_after: datetime | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_run(self, repository: str, run_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def rerun_failed_jobs(self, repository: str, run_id: str) -> None: ...

    @abstractmethod
    def download_release_receipt(
        self,
        repository: str,
        run_id: str,
        run_attempt: int,
    ) -> dict[str, Any]: ...


class HttpGitHubWorkflowGateway(GitHubWorkflowGateway):
    """GitHub REST implementation using one Actions read/write credential."""

    API = "https://api.github.com"
    MAX_JSON_BYTES = 1024 * 1024
    MAX_ARTIFACT_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        token: str | None = None,
        *,
        token_loader: Callable[[], str] | None = None,
        client: httpx.Client | None = None,
    ):
        self._fixed_token = token
        self._token_loader = token_loader or _stored_dispatch_token
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=False)

    def _token(self) -> str:
        token = self._fixed_token if self._fixed_token is not None else self._token_loader()
        if not token:
            raise GitHubWorkflowError(
                "credential_unavailable",
                "GitHub release dispatch credential is not connected",
            )
        return token

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        json_body: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> httpx.Response:
        url = path_or_url if path_or_url.startswith("http") else self.API + path_or_url
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "llm-bawt-release-coordinator",
        }
        if auth:
            headers["Authorization"] = f"Bearer {self._token()}"
        try:
            response = self._client.request(method, url, headers=headers, json=json_body)
        except httpx.HTTPError as exc:
            raise GitHubWorkflowError(
                "github_unreachable",
                f"GitHub request failed: {type(exc).__name__}",
                retryable=True,
            ) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise GitHubWorkflowError(
                "github_transient",
                f"GitHub returned HTTP {response.status_code}",
                retryable=True,
            )
        if response.status_code in (401, 403):
            raise GitHubWorkflowError(
                "github_forbidden",
                f"GitHub rejected the release credential (HTTP {response.status_code})",
            )
        return response

    @staticmethod
    def _expect(response: httpx.Response, statuses: set[int], label: str) -> httpx.Response:
        if response.status_code not in statuses:
            raise GitHubWorkflowError(
                "github_contract",
                f"{label} returned HTTP {response.status_code}",
            )
        return response

    def _json(self, path: str) -> dict[str, Any]:
        response = self._expect(self._request("GET", path), {200}, f"GitHub GET {path.split('?')[0]}")
        if len(response.content) > self.MAX_JSON_BYTES:
            raise GitHubWorkflowError("response_too_large", "GitHub JSON response exceeded limit")
        try:
            value = response.json()
        except ValueError as exc:
            raise GitHubWorkflowError("github_contract", "GitHub returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise GitHubWorkflowError("github_contract", "GitHub JSON response is not an object")
        return value

    def _tag_commit(self, repository: str, tag: str) -> str:
        ref = self._json(f"/repos/{repository}/git/ref/tags/{tag}").get("object") or {}
        if ref.get("type") == "tag":
            ref = self._json(f"/repos/{repository}/git/tags/{ref.get('sha')}").get("object") or {}
        sha = ref.get("sha")
        if ref.get("type") != "commit" or not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise GitHubWorkflowError("github_contract", f"tag {tag} does not resolve to a commit")
        return sha

    def resolve_branch_head(self, repository: str, branch: str) -> str:
        obj = self._json(f"/repos/{repository}/git/ref/heads/{branch}").get("object") or {}
        sha = obj.get("sha")
        if obj.get("type") != "commit" or not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise GitHubWorkflowError("github_contract", f"branch {branch} has no valid commit SHA")
        return sha

    def latest_semver_tag(self, repository: str) -> dict[str, str] | None:
        response = self._expect(
            self._request("GET", f"/repos/{repository}/tags?per_page=100"),
            {200},
            "GitHub tags list",
        )
        try:
            rows = response.json()
        except ValueError as exc:
            raise GitHubWorkflowError("github_contract", "GitHub returned invalid tags JSON") from exc
        candidates: list[tuple[tuple[int, int, int], str]] = []
        for row in rows if isinstance(rows, list) else []:
            name = row.get("name") if isinstance(row, dict) else None
            match = _SEMVER_TAG.fullmatch(name or "")
            if match:
                candidates.append((tuple(int(part) for part in match.groups()), name))
        if not candidates:
            return None
        _version, name = max(candidates)
        return {"tag": name, "sha": self._tag_commit(repository, name)}

    def compare(self, repository: str, base: str, head: str) -> dict[str, Any]:
        value = self._json(f"/repos/{repository}/compare/{base}...{head}")
        return {
            "status": value.get("status"),
            "ahead_by": int(value.get("ahead_by") or 0),
            "behind_by": int(value.get("behind_by") or 0),
            "total_commits": int(value.get("total_commits") or 0),
        }

    def dispatch_workflow(
        self,
        repository: str,
        workflow_path: str,
        branch: str,
        inputs: dict[str, str],
    ) -> None:
        response = self._request(
            "POST",
            f"/repos/{repository}/actions/workflows/{workflow_path}/dispatches",
            json_body={"ref": branch, "inputs": inputs},
        )
        self._expect(response, {204}, "GitHub workflow dispatch")

    def find_correlated_runs(
        self,
        repository: str,
        workflow_path: str,
        branch: str,
        release_request_id: str,
        *,
        created_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        data = self._json(
            f"/repos/{repository}/actions/workflows/{workflow_path}/runs"
            f"?event=workflow_dispatch&branch={branch}&per_page=100"
        )
        expected_title = f"Release frontend {release_request_id}"
        after = created_after
        if after is not None and after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)
        matches = []
        for run in data.get("workflow_runs", []):
            if run.get("display_title") != expected_title:
                continue
            if run.get("event") != "workflow_dispatch" or run.get("head_branch") != branch:
                continue
            if (run.get("repository") or {}).get("full_name") != repository:
                continue
            if after is not None:
                try:
                    created = datetime.fromisoformat(str(run.get("created_at")).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if created < after:
                    continue
            matches.append(run)
        return matches

    def get_run(self, repository: str, run_id: str) -> dict[str, Any]:
        return self._json(f"/repos/{repository}/actions/runs/{run_id}")

    def rerun_failed_jobs(self, repository: str, run_id: str) -> None:
        response = self._request("POST", f"/repos/{repository}/actions/runs/{run_id}/rerun-failed-jobs")
        self._expect(response, {201}, "GitHub rerun failed jobs")

    def download_release_receipt(
        self,
        repository: str,
        run_id: str,
        run_attempt: int,
    ) -> dict[str, Any]:
        name = f"release-receipt-{run_attempt}"
        listing = self._json(f"/repos/{repository}/actions/runs/{run_id}/artifacts?name={name}")
        artifacts = [
            row
            for row in listing.get("artifacts", [])
            if row.get("name") == name and not row.get("expired")
        ]
        if not artifacts:
            # Distinct from ambiguity: the attempt produced no receipt (receipt
            # job never ran/uploaded), which a same-run rerun can repair.
            raise GitHubWorkflowError("receipt_missing", f"workflow run has no unexpired {name} artifact")
        if len(artifacts) != 1:
            raise GitHubWorkflowError(
                "receipt_ambiguous",
                f"workflow run must carry exactly one unexpired {name} artifact",
            )
        response = self._request("GET", artifacts[0]["archive_download_url"])
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise GitHubWorkflowError("github_contract", "artifact redirect has no location")
            response = self._request("GET", location, auth=False)
        self._expect(response, {200}, "GitHub receipt artifact download")
        if len(response.content) > self.MAX_ARTIFACT_BYTES:
            raise GitHubWorkflowError("response_too_large", "release receipt artifact exceeded limit")
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                raw = archive.read("release.json")
            value = json.loads(raw)
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise GitHubWorkflowError(
                "receipt_invalid",
                "release receipt artifact has no valid release.json",
            ) from exc
        if not isinstance(value, dict):
            raise GitHubWorkflowError("receipt_invalid", "release receipt is not an object")
        return value


def _stored_dispatch_token() -> str:
    from ..service.providers.github_deploy import GitHubReleaseDispatchAdapter

    try:
        return GitHubReleaseDispatchAdapter(Config()).api_key() or ""
    except Exception as exc:  # noqa: BLE001 - credential-store errors are surfaced safely
        raise GitHubWorkflowError(
            "credential_unavailable",
            f"GitHub release dispatch credential unavailable: {type(exc).__name__}",
        ) from exc


__all__ = [
    "GitHubWorkflowGateway",
    "HttpGitHubWorkflowGateway",
    "GitHubWorkflowError",
]
