"""GitHub credentials for the BawtHub release deploy path (TASK-997).

Two static tokens, stored like every other API key: encrypted in the
:class:`CredentialStore` and connected through the Providers UI
(``POST /v1/providers/{id}/connect/api-key``). Nothing reads them from env.

- ``github-release`` — fine-grained token, Actions+Contents READ on the release
  repository. :class:`llm_bawt.ops.release.GitHubReleaseVerifier` uses it to
  verify a workflow run/receipt/tag before an approval snapshot exists.
- ``github-release-dispatch`` — separate fine-grained token, Actions READ+WRITE
  and Contents READ on the release repositories. The durable release coordinator
  uses it for remote preflight, workflow dispatch, status and safe failed-job reruns.
- ``ghcr-pull`` — classic token with ``read:packages``. The ops executor uses it
  to pull the approved ``repo@digest`` during deploy preflight, so the one-shot
  worker never holds a registry credential.
"""

from __future__ import annotations

import httpx

from .api_key import ApiKeyAdapter

_USER_URL = "https://api.github.com/user"


class _GitHubTokenAdapter(ApiKeyAdapter):
    """Validates a GitHub token against ``GET /user``; account = GitHub login."""

    category = "deploy"
    description = ""
    credential_help = ""
    setup_url = ""
    required_scope: str | None = None  # classic-token scope that must be granted

    def descriptor(self) -> dict:
        return {
            **super().descriptor(),
            "category": self.category,
            "credential_label": "GitHub token",
            "description": self.description,
            "credential_help": self.credential_help,
            "setup_url": self.setup_url,
        }

    @classmethod
    def _probe(cls, key: str) -> tuple[str | None, str | None]:
        try:
            resp = httpx.get(_USER_URL, timeout=15.0, headers={
                "Authorization": f"Bearer {key}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "llm-bawt"})
        except httpx.HTTPError as exc:
            return None, f"GitHub unreachable: {type(exc).__name__}"
        if resp.status_code in (401, 403):
            return None, "GitHub rejected the token"
        if resp.is_error:
            return None, f"GitHub token check returned HTTP {resp.status_code}"
        if cls.required_scope:
            granted = {s.strip() for s in resp.headers.get("X-OAuth-Scopes", "").split(",") if s.strip()}
            # write:packages implies read:packages.
            if not granted & {cls.required_scope, cls.required_scope.replace("read:", "write:")}:
                return None, f"token lacks the {cls.required_scope} scope (use a classic token)"
        try:
            login = str(resp.json().get("login") or "").strip()
        except ValueError:
            login = ""
        if not login:
            return None, "GitHub returned no account login for this token"
        return login, None


class GitHubReleaseAdapter(_GitHubTokenAdapter):
    id = "github-release"
    label = "GitHub release verification"
    description = ("Read-only check that a BawtHub release run, receipt and tag are genuine "
                   "before a production deploy approval is shown.")
    credential_help = ("Fine-grained token, resource owner bawthub, only the bawthub repository, "
                       "permissions Actions: Read and Contents: Read.")
    setup_url = "https://github.com/settings/personal-access-tokens/new"


class GitHubReleaseDispatchAdapter(_GitHubTokenAdapter):
    id = "github-release-dispatch"
    label = "GitHub release workflow dispatch"
    description = (
        "Runs and reconciles the durable BawtHub release workflow; separate from "
        "the read-only deployment verifier."
    )
    credential_help = (
        "Fine-grained token, resource owner bawthub, only the bawthub and llm-bawt "
        "repositories, permissions Actions: Read and write and Contents: Read."
    )
    setup_url = "https://github.com/settings/personal-access-tokens/new"


class GhcrPullAdapter(_GitHubTokenAdapter):
    id = "ghcr-pull"
    label = "GitHub container registry pull"
    description = "Pulls approved BawtHub release images (by digest) from ghcr.io during deploy."
    credential_help = "Classic token with only the read:packages scope (ghcr.io does not accept fine-grained tokens)."
    setup_url = "https://github.com/settings/tokens/new?scopes=read:packages&description=bawthub-prod-pull"
    required_scope = "read:packages"

    def pull_auth(self) -> dict | None:
        """Docker ``auth_config`` for ghcr.io, or None when not connected."""
        record = self.store.load(self.id)
        token = self.api_key()
        if not token or not record or not record.account:
            return None
        return {"username": record.account, "password": token}
