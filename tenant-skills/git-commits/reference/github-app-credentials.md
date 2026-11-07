# GitHub App git and gh credentials

`git-credential-bawthub-app` and `gh-bawthub-app` (in `scripts/`) let an agent
account push and open PRs as the GitHub App without a personal token on disk.
Each call mints a short-lived installation token in the llm-bawt app for the one
repository in use. The App private key stays in the app; the bridge only holds
the `X-Bridge-Token` gate file.

## Contract

- Helper request: `GET <api_url>/v1/providers/github/git-credential?repository=<owner>/<repo>`
  with `X-Bridge-Token` when a gate file is configured. The app resolves the
  App installation for that repository and mints a token restricted to it,
  without changing the stored installation selection. Without `repository` the
  endpoint keeps its legacy selected-installation behaviour, which this helper
  never uses. The reply is
  `username=…` / `password=…` lines.
- Before any request the helper refuses: a protocol other than `https`, a host
  other than exactly `github.com` (no port, suffix or userinfo), a missing or
  non-`owner/repo` path, duplicate or malformed attributes. No proxies, no
  redirects, no echoed error bodies, nothing written to disk.
- `gh-bawthub-app` resolves the repository (`-R`/`--repo`, a URL selector such
  as `gh pr view https://github.com/OWNER/REPO/pull/1` or `gh repo view
  OWNER/REPO`, `GH_REPO`, `gh api repos/OWNER/REPO/…`, then `origin`), mints
  through the helper and execs the real `gh` with `GH_TOKEN` and `GH_REPO` set
  only in the child. Sources naming different repositories are refused. It
  refuses non-github.com `-R`/URL selectors/`GH_REPO`/`GH_HOST`/`--hostname`,
  absolute `gh api` URLs other than `https://api.github.com`, any argument naming
  a host `gh` also sends `GH_TOKEN` to (`*.ghe.com`, `github.localhost`), and
  `gh auth`, `gh extension` and `gh alias`. Other tokens (`GITHUB_TOKEN`, `GH_ENTERPRISE_TOKEN`,
  `GITHUB_ENTERPRISE_TOKEN`) are dropped from the child.

## Per-account setup

Configuration lives under the account's home, outside every repository, so a
repository rsync or re-clone does not remove it.

```sh
mkdir -p ~/.config/bawthub-git && chmod 700 ~/.config/bawthub-git
cat > ~/.config/bawthub-git/credential.conf <<'CONF'
api_url=http://app:8642
token_file=/path/to/git-credential-gate-token
CONF
git config --global credential.https://github.com.helper ''
git config --global --add credential.https://github.com.helper bawthub-app
git config --global credential.https://github.com.useHttpPath true
```

`useHttpPath` is required; without it the helper cannot pick an installation
and fails closed. The empty first `helper` entry clears inherited helpers for
github.com only. Use `gh-bawthub-app` in place of `gh`; its state stays in
`~/.config/bawthub-git/gh`, so the account's own `gh` login and a human's
`~/.config/gh` are never read or changed. Environment overrides:
`BAWTHUB_GIT_CONFIG_DIR`, `BAWTHUB_GIT_API_URL`,
`BRIDGE_GIT_CREDENTIAL_TOKEN_FILE`, `BAWTHUB_GH_BIN`,
`BAWTHUB_GIT_CREDENTIAL_HELPER`.

## Limits

- One token covers one repository. A `gh` command that touches a second
  repository gets a 404 from GitHub; pass `-R` for it instead.
- A `gh` command with no repository context (outside a clone, no `-R`) is
  refused; there is no default account and no fallback to anyone's own login.
- Hosts other than `github.com` (including GitHub Enterprise) are not supported.
- This bounds where the token can be sent. It does not stop the same Unix user
  from calling the helper directly.
