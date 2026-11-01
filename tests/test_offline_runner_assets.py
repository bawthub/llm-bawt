"""Finite bundle checks use only synthetic files, never host caches or services."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.offline_runner_assets import Refused, main, verify_bundle

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "runner-assets/manifest.v1.json"
VERSIONS = {"python": "3.12.13", "node": "24.21.0", "uv": "0.10.8", "corepack": "0.36.0", "pnpm": "10.32.1"}


@pytest.fixture
def staged(tmp_path):
    payload = b"image/webp webp\n"
    (tmp_path / "mime").mkdir()
    (tmp_path / "mime/mime.types").write_bytes(payload)
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    manifest["files"] = [{
        "path": "mime/mime.types", "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload), "origin": "synthetic fixture, not upstream asset",
    }]
    return tmp_path, manifest


def check(staged, versions=None):
    root, manifest = staged
    data = json.dumps(manifest).encode()
    return verify_bundle(root, data, hashlib.sha256(data).hexdigest(), json.dumps(versions or VERSIONS).encode())


def test_valid_finite_bundle_never_claims_gate_or_attestation(staged):
    result = check(staged)
    assert result["status"] == "verified_finite_files_and_declared_versions"
    assert result["file_count"] == 1
    assert result["declared_tool_versions"] == VERSIONS
    assert result["attestation"] is result["complete_offline_ci"] is result["gate_receipt"] is False


def test_same_size_modified_file_refused(staged):
    (staged[0] / "mime/mime.types").write_bytes(b"image/jpeg jpeg\n")
    with pytest.raises(Refused, match="SHA256 mismatch"):
        check(staged)


def test_missing_file_refused(staged):
    (staged[0] / "mime/mime.types").unlink()
    with pytest.raises(FileNotFoundError):
        check(staged)


def test_wrong_size_refused(staged):
    staged[1]["files"][0]["size_bytes"] += 1
    with pytest.raises(Refused, match="size mismatch"):
        check(staged)


@pytest.mark.parametrize("value", ["0" * 64, "A" * 64, "", "sha256:123", None])
def test_manifest_digest_must_match_before_parsing(staged, value):
    with pytest.raises(Refused, match="manifest SHA256"):
        verify_bundle(staged[0], b"not JSON", value, b"{}")


@pytest.mark.parametrize("relative", ["../outside", "/etc/mime.types", "mime/../mime.types", "mime//mime.types", "./mime.types", "", "C:/file", "mime\\file", "mime/"])
def test_path_escape_and_ambiguous_spelling_refused(staged, relative):
    staged[1]["files"][0]["path"] = relative
    with pytest.raises(Refused, match="relative path"):
        check(staged)


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_symlinks_refused_even_when_target_is_inside_bundle(staged, kind):
    root, _ = staged
    if kind == "file":
        (root / "mime/mime.types").rename(root / "saved.types")
        (root / "mime/mime.types").symlink_to(root / "saved.types")
    elif kind == "directory":
        (root / "mime").rename(root / "saved")
        (root / "mime").symlink_to(root / "saved", target_is_directory=True)
    else:
        link = root / "linked-root"
        link.symlink_to(root, target_is_directory=True)
        staged = link, staged[1]
    with pytest.raises(Refused, match="symlink|real directory"):
        check(staged)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_payload_refused_without_opening(staged, kind):
    path = staged[0] / "mime/mime.types"
    path.unlink()
    if kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(Refused, match="regular file"):
        check(staged)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", True), ("scope", "SHIP"), ("platform", "linux-arm64"), ("files", []), ("files", {}), ("limitations", []), ("limitations", [False])])
def test_unknown_or_broadened_manifest_refused(staged, field, value):
    staged[1][field] = value
    with pytest.raises(Refused):
        check(staged)


@pytest.mark.parametrize("field,value", [("sha256", "A" * 64), ("sha256", None), ("size_bytes", True), ("size_bytes", -1), ("origin", ""), ("path", None)])
def test_invalid_entry_refused(staged, field, value):
    staged[1]["files"][0][field] = value
    with pytest.raises(Refused):
        check(staged)


def test_duplicate_path_refused(staged):
    staged[1]["files"].append(copy.deepcopy(staged[1]["files"][0]))
    with pytest.raises(Refused, match="duplicate bundle path"):
        check(staged)


@pytest.mark.parametrize("data", [b'{"schema_version":1,"schema_version":1}', b'[]', b'{', b'{"python":"3.12.13","python":"3.12.13"}'])
def test_malformed_or_duplicate_json_refused(staged, data):
    with pytest.raises(ValueError):
        verify_bundle(staged[0], data, hashlib.sha256(data).hexdigest(), b"{}")


@pytest.mark.parametrize("tool,version", [("python", "3.14.0"), ("node", "24.20.0"), ("uv", "0.13.0"), ("corepack", "0.35.0"), ("pnpm", "10.32.0"), ("node", "v24.21.0"), ("python", "3.12")])
def test_wrong_or_inexact_tool_version_refused(staged, tool, version):
    versions = VERSIONS | {tool: version}
    with pytest.raises(Refused, match="version"):
        check(staged, versions)


def test_python_patch_is_not_falsely_claimed_as_ci_pin(staged):
    assert check(staged, VERSIONS | {"python": "3.12.99"})["declared_tool_versions"]["python"] == "3.12.99"


@pytest.mark.parametrize("target", ["declaration", "requirements"])
def test_missing_or_extra_tool_refused(staged, target):
    versions = VERSIONS.copy()
    if target == "declaration":
        versions["shell"] = "1.0.0"
    else:
        del staged[1]["tool_versions"]["node"]
    with pytest.raises(Refused, match="tool .*set"):
        check(staged, versions)


def test_unknown_version_rule_refused(staged):
    staged[1]["tool_versions"]["node"]["match"] = "at_least"
    with pytest.raises(Refused, match="unsupported version rule"):
        check(staged)


def test_duplicate_declared_version_refused(staged):
    data = json.dumps(staged[1]).encode()
    with pytest.raises(Refused, match="duplicate JSON"):
        verify_bundle(staged[0], data, hashlib.sha256(data).hexdigest(), b'{"python":"3.12.13","python":"3.14.0"}')


def test_cli_success_and_refusal_are_machine_readable(staged, capsys):
    root, manifest = staged
    manifest_data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(manifest_data)
    (root / "versions.json").write_text(json.dumps(VERSIONS))
    args = ["--bundle", str(root), "--manifest", str(root / "manifest.json"),
            "--manifest-sha256", hashlib.sha256(manifest_data).hexdigest(),
            "--declared-tool-versions", str(root / "versions.json")]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["gate_receipt"] is False
    (root / "mime/mime.types").unlink()
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "refused"


def test_committed_manifest_preserves_pins_and_explicit_limitations():
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    files = {entry["path"]: entry for entry in manifest["files"]}
    tokenizer = files["tiktoken/fb374d419588a4632f3f557e76b4b70aebbca790"]
    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    assert tokenizer["path"].split("/")[-1] == hashlib.sha1(url.encode()).hexdigest()
    assert tokenizer["sha256"] == "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    assert manifest["tool_versions"]["uv"]["version"] == "0.10.8"
    assert manifest["tool_versions"]["python"]["match"] == "major_minor"
    assert len(files) == 13
    assert all("hostcache" not in entry["origin"] for entry in files.values())
    assert any("NOT" in item and "attestation" in item for item in manifest["limitations"])
