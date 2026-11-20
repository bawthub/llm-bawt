"""The installation's dedicated SSH identity for host access (TASK-2520).

At bridge start, when ``~/.ssh`` is a writable directory (a tenant's
per-installation volume) and holds no ``id_ed25519``, generate one with no
passphrase. Settings > Host access then shows its public key and authorizes it
on the host. An existing key, or a read-only ``~/.ssh`` (an operator bind
mount), is left untouched. Never raises: host access is optional.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_ssh_identity(home: Path | None = None) -> str:
    """Returns ``present``, ``generated``, ``read_only`` or ``unavailable``."""
    ssh_dir = (home or Path.home()) / ".ssh"
    key = ssh_dir / "id_ed25519"
    try:
        if key.exists() and key.with_suffix(".pub").exists():
            return "present"
        if os.environ.get("BRIDGE_SSH_IDENTITY", "1") == "0" or not shutil.which("ssh-keygen"):
            return "unavailable"
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        if not os.access(ssh_dir, os.W_OK):
            return "read_only"
        if key.exists():  # private key without its .pub: derive, never replace
            pub = subprocess.run(["ssh-keygen", "-y", "-f", str(key)], capture_output=True,
                                 text=True, timeout=20, check=True).stdout
            key.with_suffix(".pub").write_text(pub)
            return "present"
        comment = f"bawthub-bots@{socket.gethostname()[:48]}"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(key)],
                       capture_output=True, timeout=30, check=True)
        os.chmod(ssh_dir, 0o700)
        os.chmod(key, 0o600)
        logger.info("Generated the installation SSH identity for host access")
        return "generated"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Host access SSH identity unavailable: %s", type(exc).__name__)
        return "unavailable"
