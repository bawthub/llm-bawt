"""Pinned MiniLM embedding model, fetched on the user's machine.

No model weights ship in images, exports or the installer. On first use the
bridge downloads exactly these files of the pinned upstream revision from
Hugging Face into a cache directory (a data volume), checks every SHA-256 and
size, then loads them with networking disabled. Later starts load the cache
local-only. A missing upstream file, a wrong hash, or an extra or edited cached
file fails closed with a clear error; it never falls back to a floating
download or an ambient cache.

Manual steps (optional; first use does the same)::

    python -m local_model_bridge.pinned_embed_model fetch /models/all-MiniLM-L6-v2
    python -m local_model_bridge.pinned_embed_model verify /models/all-MiniLM-L6-v2
"""

from __future__ import annotations

import hashlib
import os
import sys
import shutil
import tempfile
import threading
import urllib.request
from pathlib import Path

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_NAME = "all-MiniLM-L6-v2"
#: Cache directory inside the bridge's model data volume.
DEFAULT_DIR = "/models/all-MiniLM-L6-v2"
PINNED_DIR_ENV = "LOCAL_MODEL_EMBED_PINNED_DIR"

# path -> (sha256, bytes); safetensors only, no pickle/ONNX/TF weight copies.
FILES: dict[str, tuple[str, int]] = {
    "1_Pooling/config.json": ("4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23", 190),
    "README.md": ("dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0", 10502),
    "config.json": ("953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41", 612),
    "config_sentence_transformers.json": ("061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3", 116),
    "modules.json": ("84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf", 349),
    "sentence_bert_config.json": ("fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7", 53),
    "special_tokens_map.json": ("303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3", 112),
    "tokenizer.json": ("be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037", 466247),
    "tokenizer_config.json": ("acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b", 350),
    "vocab.txt": ("07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
    "model.safetensors": ("53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", 90868376),
}

_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


#: Serializes first-use fetches in this process (warmup and /embed can race).
_FIRST_USE_LOCK = threading.Lock()

class PinnedModelError(RuntimeError):
    """The pinned model directory does not hold exactly the pinned bytes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: str | os.PathLike[str]) -> Path:
    """Return ``directory`` when it holds exactly ``FILES``; raise otherwise."""
    root = Path(directory)
    if not root.is_dir():
        raise PinnedModelError(f"pinned model directory missing: {root}")
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    extra = sorted(present - set(FILES))
    missing = sorted(set(FILES) - present)
    if extra or missing:
        raise PinnedModelError(f"pinned model file set differs: missing={missing} extra={extra}")
    for rel, (sha, size) in FILES.items():
        path = root / rel
        if path.is_symlink():
            raise PinnedModelError(f"pinned model file is a symlink: {rel}")
        if path.stat().st_size != size or _sha256(path) != sha:
            raise PinnedModelError(f"pinned model file mismatch: {rel}")
    return root


def ensure(directory: str | os.PathLike[str]) -> Path:
    """Fetch the pinned files into an absent or empty cache, then verify it.

    An existing, non-empty cache is only verified: a mismatch is reported, not
    silently overwritten.
    """
    root = Path(directory)
    with _FIRST_USE_LOCK:  # one download per process; later callers verify the result
        if not root.exists() or not any(root.iterdir()):
            return fetch(root)
        return verify(root)


def load(directory: str | os.PathLike[str], device: str = "cpu"):
    """Load the pinned model (fetching it on first use) with Hugging Face networking disabled."""
    root = ensure(directory)
    os.environ.update(_OFFLINE_ENV)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(root), device=device, local_files_only=True)


def fetch(directory: str | os.PathLike[str]) -> Path:
    """Download the pinned revision's files into an absent or empty ``directory``.

    Files land in a sibling staging directory and are verified before it is
    moved into place, so an interrupted or mismatched download leaves no cache.
    """
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise PinnedModelError(f"fetch target is not empty: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        for rel, (sha, size) in FILES.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{rel}"
            try:
                with urllib.request.urlopen(url, timeout=600) as response, target.open("wb") as out:  # noqa: S310 - fixed https URL
                    shutil.copyfileobj(response, out, 1 << 20)
            except OSError as exc:
                raise PinnedModelError(f"could not download {REPO_ID}@{REVISION}/{rel}: {exc}") from exc
            if target.stat().st_size != size or _sha256(target) != sha:
                raise PinnedModelError(
                    f"downloaded {rel} does not match the pinned SHA-256 for {REPO_ID}@{REVISION}"
                )
        verify(staging)
        try:
            if root.exists():
                root.rmdir()  # empty, checked above
            staging.replace(root)
        except OSError:
            # Another process published first; use its cache if it verifies,
            # never overwrite it.
            if root.is_dir() and any(root.iterdir()):
                return verify(root)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("fetch", "verify"):
        print("usage: python -m local_model_bridge.pinned_embed_model fetch|verify DIR", file=sys.stderr)
        return 2
    try:
        (fetch if args[0] == "fetch" else verify)(args[1])
    except PinnedModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ok: {REPO_ID}@{REVISION} verified in {args[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
