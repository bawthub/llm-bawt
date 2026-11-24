"""Pinned, local-only MiniLM embedding model files.

The public text profile bakes exactly these upstream files into a fixed local
directory at image build time and loads them with networking disabled. Loading
fails closed: a missing, extra or mismatched file means no model, never a
Hugging Face download or an ambient cache.

Build step::

    python -m local_model_bridge.pinned_embed_model fetch /opt/models/all-MiniLM-L6-v2
    python -m local_model_bridge.pinned_embed_model verify /opt/models/all-MiniLM-L6-v2
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DIR = "/opt/models/all-MiniLM-L6-v2"
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


def load(directory: str | os.PathLike[str], device: str = "cpu"):
    """Load the verified directory with Hugging Face networking disabled."""
    root = verify(directory)
    os.environ.update(_OFFLINE_ENV)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(root), device=device, local_files_only=True)


def fetch(directory: str | os.PathLike[str]) -> Path:
    """Download the pinned revision's files into an empty ``directory`` and verify them."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise PinnedModelError(f"fetch target is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for rel, (sha, _size) in FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{rel}"
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310 - fixed https URL
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    tmp.write(chunk)
        Path(tmp.name).replace(target)
        if _sha256(target) != sha:
            raise PinnedModelError(f"downloaded file mismatch: {rel}")
    return verify(root)


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
