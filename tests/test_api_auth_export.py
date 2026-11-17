import os
import stat

import pytest

from llm_bawt.api_auth import client, export, keys

ROOT = b"r" * 32


def _root():
    return ROOT


def test_parse_entries_and_rejects_bad():
    specs = export.parse("bawthub-web=/k/web:1000:1001, house=/k/house::50,cli-taurus=/k/cli")
    assert [(s.principal, str(s.directory), s.uid, s.gid) for s in specs] == [
        ("bawthub-web", "/k/web", 1000, 1001), ("house", "/k/house", None, 50),
        ("cli-taurus", "/k/cli", None, None)]
    for bad in ("x", "Bad=/k", "a=rel", "a=/k:x", "a=/k:1:2:3", "a=/k,a=/j"):
        with pytest.raises(ValueError):
            export.parse(bad)


def test_export_writes_verifiable_key_and_client_reads_it(tmp_path, monkeypatch):
    spec = export.ExportSpec("bawthub-web", tmp_path / "web", None, None)
    assert export.export_all([spec], root=_root) == {"bawthub-web": "written"}
    path = tmp_path / "web" / export.FILE_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.setenv(client.FILE_ENV, str(path))
    headers = client.add_auth_header({"Accept": "x"})
    check = keys.verify(headers[client.HEADER_NAME], root=_root)
    assert check.ok and check.principal == "bawthub-web"
    # A caller-set header is never overwritten.
    assert client.add_auth_header({"x-llm-bawt-key": "mine"}) == {"x-llm-bawt-key": "mine"}


def test_failed_source_keeps_last_copy(tmp_path):
    spec = export.ExportSpec("house", tmp_path, None, None)
    export.export_all([spec], root=_root)
    before = (tmp_path / export.FILE_NAME).read_text()

    def broken():
        raise RuntimeError("no master key")

    assert export.export_all([spec], root=broken) == {"house": "kept"}
    assert (tmp_path / export.FILE_NAME).read_text() == before


def test_unset_config_and_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv(export.EXPORTS_ENV, raising=False)
    assert export.export_all() == {}
    monkeypatch.setenv(export.EXPORTS_ENV, "not valid")
    assert export.export_all() == {}
    monkeypatch.setenv(client.FILE_ENV, str(tmp_path / "absent"))
    assert client.auth_headers() == {}
    monkeypatch.delenv(client.FILE_ENV)
    assert client.auth_headers() == {}


def test_replaced_file_seen_through_directory(tmp_path, monkeypatch):
    path = tmp_path / export.FILE_NAME
    export.export_all([export.ExportSpec("house", tmp_path, None, None)], root=_root)
    monkeypatch.setenv(client.FILE_ENV, str(path))
    first = client.header_value()
    inode = os.stat(path).st_ino
    export.export_all([export.ExportSpec("house", tmp_path, None, None)], root=lambda: b"s" * 32)
    assert os.stat(path).st_ino != inode
    assert client.header_value() != first
