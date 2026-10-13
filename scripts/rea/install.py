"""Install a frozen REA release without lifecycle scripts or service changes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
VERSION = "6.4.0"
INTEGRITY = "sha512-Vj1zWfSlna5bphGoaoI/TcVBuYvze9u+OyPC3Shz8YkBqaCTqB4COnnRoX7XDwMpwI24ZlFEel2yEj0k8LTWEQ=="


class Installer:
    def __init__(self, root: Path):
        self.root = root.expanduser().absolute()

    @staticmethod
    def run(*args: str, cwd: Path | None = None, timeout: int = 120) -> str:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr[-4000:]}")
        return result.stdout.strip()

    def smoke(self, release: Path) -> None:
        package = release / "node_modules/rea-agents"
        installed = json.loads((package / "package.json").read_text())
        if installed.get("version") != VERSION:
            raise RuntimeError("Installed REA version does not match the pin")
        output = self.run("node", str(package / "scripts/rea.mjs"), "--help", cwd=release)
        if not output:
            raise RuntimeError("REA CLI smoke returned no help")

    def install(self) -> Path:
        # Check before creating or changing any installation state.
        self.run("node", str(HERE / "check-node.mjs"))
        node = self.run("node", "--version")
        npm = self.run("npm", "--version")
        manifest = (HERE / "package.json").read_bytes()
        lock_bytes = (HERE / "package-lock.json").read_bytes()
        lock = json.loads(lock_bytes)
        dependency = lock["packages"]["node_modules/rea-agents"]
        if dependency["version"] != VERSION or dependency["integrity"] != INTEGRITY:
            raise RuntimeError("REA lock provenance differs from the reviewed release")
        for path, package in lock["packages"].items():
            if path and (not package.get("integrity") or not package.get("resolved", "").startswith("https://registry.npmjs.org/")):
                raise RuntimeError(f"Unpinned/non-registry dependency: {path}")
            if package.get("hasInstallScript"):
                raise RuntimeError(f"New lifecycle script requires review: {path}")
        digest = hashlib.sha256(manifest + b"\0" + lock_bytes).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / ".install.lock").open("a") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another REA installation is running") from exc
            releases = self.root / "releases"
            releases.mkdir(exist_ok=True)
            release = releases / f"{VERSION}-{digest}"
            if release.exists():
                receipt = json.loads((release / "receipt.json").read_text())
                if receipt.get("input_sha256") != digest:
                    raise RuntimeError("Existing release receipt differs; refusing to replace it")
                if (release / "package.json").read_bytes() != manifest or (release / "package-lock.json").read_bytes() != lock_bytes:
                    raise RuntimeError("Existing release inputs changed; refusing to replace it")
                self.smoke(release)
            else:
                stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=releases))
                try:
                    (stage / "package.json").write_bytes(manifest)
                    (stage / "package-lock.json").write_bytes(lock_bytes)
                    self.run("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund",
                             "--registry=https://registry.npmjs.org", cwd=stage, timeout=300)
                    self.smoke(stage)
                    receipt = {
                        "rea": VERSION, "integrity": INTEGRITY, "input_sha256": digest,
                        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                        "node": node, "npm": npm, "lifecycle_scripts": "disabled",
                        "installed_at": datetime.now(timezone.utc).isoformat(),
                        "smoke": "CLI --help; static/MCP acceptance recorded separately",
                    }
                    (stage / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                    stage.rename(release)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            current = self.root / "current"
            if current.exists() and not current.is_symlink():
                raise RuntimeError("current is not a symlink; refusing to overwrite")
            if current.is_symlink() and current.resolve() == release:
                return release
            # Each client resolves once; never overwrite files beneath running clients.
            with tempfile.TemporaryDirectory(prefix=".promote-", dir=self.root) as temp:
                pointer = Path(temp) / "current"
                pointer.symlink_to(release.relative_to(self.root))
                os.replace(pointer, current)
            return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("REA_INSTALL_ROOT", str(Path.home() / ".claude/tools/rea"))))
    args = parser.parse_args()
    try:
        release = Installer(args.root).install()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"REA install: {exc}", file=sys.stderr)
        return 1
    print(f"REA {VERSION} ready: {release}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
