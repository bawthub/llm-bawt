"""TASK-1270: the running-image vs uv.lock postflight check."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_running_lock", Path(__file__).resolve().parents[1] / "scripts" / "check_running_lock.py")
crl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crl)

LOCK = '''
version = 1
[[package]]
name = "Huggingface_Hub"
version = "0.36.2"
[[package]]
name = "numpy"
version = "2.2.6"
[[package]]
name = "numpy"
version = "2.3.1"
[[package]]
name = "llm-bawt"
source = { editable = "." }
[[package]]
name = "pywin32"
version = "311"
'''


def test_matching_install_passes_and_ignores_uninstalled_and_project():
    locked = crl.lock_versions(LOCK)
    report = crl.compare({"huggingface-hub": "0.36.2", "numpy": "2.3.1", "llm_bawt": "0.1.0"}, locked)
    assert report["ok"] and report["mismatched"] == {} and report["not_in_lock"] == []


def test_version_difference_and_unlocked_package_fail():
    report = crl.compare({"huggingface_hub": "1.31.0", "extra-pkg": "1.0"}, crl.lock_versions(LOCK))
    assert not report["ok"]
    assert report["mismatched"] == {"huggingface-hub": {"running": "1.31.0", "lock": ["0.36.2"]}}
    assert report["not_in_lock"] == ["extra-pkg==1.0"]


def test_unreadable_lock_exits_2(tmp_path, capsys):
    assert crl.main(["--lock", str(tmp_path / "missing.lock")]) == 2
    assert '"ok": false' in capsys.readouterr().out
