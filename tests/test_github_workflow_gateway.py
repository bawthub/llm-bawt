"""TASK-1030: the release coordinator's GitHub gateway against a fake transport."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import httpx
import pytest

from llm_bawt.ops.github_workflow import GitHubWorkflowError, HttpGitHubWorkflowGateway

REPO = "bawthub/bawthub"
WF = "release-frontend.yml"
REQ = "release-" + "a" * 32
SHA = "1" * 40


def _zip(payload) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("release.json", payload if isinstance(payload, str) else json.dumps(payload))
    return buf.getvalue()


class Fake:
    def __init__(self, routes):
        self.routes = routes
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        key = (request.method, str(request.url))
        handler = self.routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"message": "not found"})
        return handler(request) if callable(handler) else handler


def _gateway(routes):
    fake = Fake(routes)
    client = httpx.Client(transport=httpx.MockTransport(fake), follow_redirects=False)
    return HttpGitHubWorkflowGateway("tok", client=client), fake


API = "https://api.github.com"


def _run(title=f"Release frontend {REQ}", **over):
    run = {
        "id": 99,
        "display_title": title,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "repository": {"full_name": REPO},
        "created_at": "2026-10-07T12:00:05Z",
    }
    run.update(over)
    return run


def test_correlation_matches_exact_title_branch_repo_and_window():
    url = f"{API}/repos/{REPO}/actions/workflows/{WF}/runs?event=workflow_dispatch&branch=main&per_page=100"
    runs = [
        _run(),
        _run(title=f"Release frontend {REQ}-suffix", id=1),
        _run(head_branch="feature", id=2),
        _run(event="push", id=3),
        _run(repository={"full_name": "evil/bawthub"}, id=4),
        _run(created_at="2026-10-07T11:00:00Z", id=5),
    ]
    gateway, fake = _gateway({("GET", url): httpx.Response(200, json={"workflow_runs": runs})})
    after = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
    found = gateway.find_correlated_runs(REPO, WF, "main", REQ, created_after=after)
    assert [run["id"] for run in found] == [99]
    assert fake.seen[0].headers["Authorization"] == "Bearer tok"


def test_dispatch_requires_204_and_rerun_requires_201():
    dispatch = f"{API}/repos/{REPO}/actions/workflows/{WF}/dispatches"
    rerun = f"{API}/repos/{REPO}/actions/runs/99/rerun-failed-jobs"
    gateway, fake = _gateway({("POST", dispatch): httpx.Response(204), ("POST", rerun): httpx.Response(201)})
    gateway.dispatch_workflow(REPO, WF, "main", {"release_request_id": REQ})
    assert json.loads(fake.seen[0].content) == {"ref": "main", "inputs": {"release_request_id": REQ}}
    gateway.rerun_failed_jobs(REPO, "99")
    gateway, _ = _gateway({("POST", dispatch): httpx.Response(200)})
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.dispatch_workflow(REPO, WF, "main", {})
    assert exc.value.code == "github_contract" and not exc.value.retryable


@pytest.mark.parametrize(
    "status, code, retryable",
    [(429, "github_transient", True), (502, "github_transient", True),
     (401, "github_forbidden", False), (403, "github_forbidden", False)],
)
def test_http_failures_are_classified(status, code, retryable):
    gateway, _ = _gateway({("GET", f"{API}/repos/{REPO}/actions/runs/99"): httpx.Response(status)})
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.get_run(REPO, "99")
    assert (exc.value.code, exc.value.retryable) == (code, retryable)


def test_network_failure_is_retryable_unreachable():
    def boom(request):
        raise httpx.ConnectError("down", request=request)

    gateway, _ = _gateway({("GET", f"{API}/repos/{REPO}/actions/runs/99"): boom})
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.get_run(REPO, "99")
    assert exc.value.code == "github_unreachable" and exc.value.retryable


def test_missing_credential_fails_closed_before_any_request():
    fake = Fake({})
    client = httpx.Client(transport=httpx.MockTransport(fake))
    gateway = HttpGitHubWorkflowGateway(token_loader=lambda: "", client=client)
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.get_run(REPO, "99")
    assert exc.value.code == "credential_unavailable" and fake.seen == []


def test_receipt_download_is_attempt_specific_and_never_forwards_token():
    listing = f"{API}/repos/{REPO}/actions/runs/99/artifacts?name=release-receipt-2"
    archive = f"{API}/repos/{REPO}/actions/artifacts/7/zip"
    blob = "https://blob.example/signed?sig=x"
    receipt = {"status": "complete", "workflow_run_attempt": "2"}
    gateway, fake = _gateway({
        ("GET", listing): httpx.Response(200, json={"artifacts": [
            {"name": "release-receipt-2", "expired": False, "archive_download_url": archive},
            {"name": "release-receipt-1", "expired": False, "archive_download_url": "x"},
            {"name": "release-receipt-2", "expired": True, "archive_download_url": "y"},
        ]}),
        ("GET", archive): httpx.Response(302, headers={"location": blob}),
        ("GET", blob): httpx.Response(200, content=_zip(receipt)),
    })
    assert gateway.download_release_receipt(REPO, "99", 2) == receipt
    blob_request = fake.seen[-1]
    assert str(blob_request.url) == blob and "Authorization" not in blob_request.headers


@pytest.mark.parametrize("artifacts, code", [
    ([], "receipt_missing"),
    ([{"name": "release-receipt-1", "expired": True}], "receipt_missing"),
    ([{"name": "release-receipt-1", "expired": False}] * 2, "receipt_ambiguous"),
])
def test_receipt_requires_exactly_one_matching_artifact(artifacts, code):
    listing = f"{API}/repos/{REPO}/actions/runs/99/artifacts?name=release-receipt-1"
    gateway, _ = _gateway({("GET", listing): httpx.Response(200, json={"artifacts": artifacts})})
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.download_release_receipt(REPO, "99", 1)
    assert exc.value.code == code


def test_receipt_rejects_invalid_archive():
    listing = f"{API}/repos/{REPO}/actions/runs/99/artifacts?name=release-receipt-1"
    archive = f"{API}/repos/{REPO}/actions/artifacts/7/zip"
    gateway, _ = _gateway({
        ("GET", listing): httpx.Response(200, json={"artifacts": [
            {"name": "release-receipt-1", "expired": False, "archive_download_url": archive}]}),
        ("GET", archive): httpx.Response(200, content=_zip("[1, 2]")),
    })
    with pytest.raises(GitHubWorkflowError) as exc:
        gateway.download_release_receipt(REPO, "99", 1)
    assert exc.value.code == "receipt_invalid"


def test_latest_semver_tag_ignores_non_release_tags_and_peels_annotated():
    tags = [{"name": n} for n in ("v0.1.9", "v0.1.44", "v0.1.44-rc1", "v01.2.3", "permission-probe-1", "v0.0.99")]
    tag_obj = "a" * 40
    gateway, _ = _gateway({
        ("GET", f"{API}/repos/bawthub/llm-bawt/tags?per_page=100"): httpx.Response(200, json=tags),
        ("GET", f"{API}/repos/bawthub/llm-bawt/git/ref/tags/v0.1.44"): httpx.Response(
            200, json={"object": {"type": "tag", "sha": tag_obj}}),
        ("GET", f"{API}/repos/bawthub/llm-bawt/git/tags/{tag_obj}"): httpx.Response(
            200, json={"object": {"type": "commit", "sha": SHA}}),
    })
    assert gateway.latest_semver_tag("bawthub/llm-bawt") == {"tag": "v0.1.44", "sha": SHA}


def test_branch_head_and_compare():
    gateway, _ = _gateway({
        ("GET", f"{API}/repos/{REPO}/git/ref/heads/main"): httpx.Response(
            200, json={"object": {"type": "commit", "sha": SHA}}),
        ("GET", f"{API}/repos/{REPO}/compare/v0.1.62...{SHA}"): httpx.Response(
            200, json={"status": "ahead", "ahead_by": 3, "behind_by": 0, "total_commits": 3}),
    })
    assert gateway.resolve_branch_head(REPO, "main") == SHA
    assert gateway.compare(REPO, "v0.1.62", SHA)["ahead_by"] == 3
