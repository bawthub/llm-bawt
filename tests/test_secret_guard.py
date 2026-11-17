"""TASK-2386: no secret value in an env template, compose file or repo .env."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from llm_bawt.secret_guard import main, scan_compose_text, scan_env_text, scan_root

ROOT = Path(__file__).resolve().parents[1]

# Synthetic values only.
SHORT = "Ab3$x!9"  # 7 characters
ENTROPY = "q8Zr2LkV9wTn4XbY7cMf1HsJ6pDg0QeA"


def codes(findings):
    return [(f.name, f.code) for f in findings]


@pytest.mark.parametrize("line", [
    f"DB_PASSWORD={SHORT}", f"DB_PASSWORD='{SHORT}'", f'export SOME_API_KEY="{ENTROPY}"',
    f"HF_TOKEN={ENTROPY}", f"X_CLIENT_SECRET={SHORT} # note", f"A_TOKEN=${{A_TOKEN:-{ENTROPY}}}",
])
def test_env_literal_fails_at_any_length(line):
    found = scan_env_text(line)
    assert [f.code for f in found] == ["secret_literal"]
    assert SHORT not in str(found[0]) and ENTROPY not in str(found[0])


@pytest.mark.parametrize("line", [
    f"DATABASE_URL=postgresql://app:{SHORT}@db:5432/x", f"REDIS_URL=redis://u:{ENTROPY}@redis:6379",
    f"ANYTHING=https://u:{SHORT}@example.com/",
])
def test_credential_url_fails(line):
    assert [f.code for f in scan_env_text(line)] == ["credential_url"]


@pytest.mark.parametrize("line", [
    "DB_PASSWORD=", "DB_PASSWORD=''", "A_TOKEN=${A_TOKEN}", "A_TOKEN=${A_TOKEN:-}", "A_TOKEN=$A_TOKEN",
    "A_TOKEN=${A_TOKEN:?set it}", "A_TOKEN_FILE=/run/secrets/a", "A_KEY_PATH=/x", "EXPORT_UID=1000",
    "A_API_KEY=<your-key>", "A_API_KEY=changeme", "A_API_KEY=your-key-here", "A_API_KEY=xxxx",
    "DATABASE_URL=postgresql://user:password@host:5432/db", "DATABASE_URL=postgresql://db:5432/x",
    "DATABASE_URL=postgresql://app:${DB_PASSWORD}@db/x", "TOKENIZERS_PARALLELISM=false",
    "GIT_CONFIG_KEY_0=credential.helper", "DB_PASSWORD= # required: set at bootstrap", "# DB_PASSWORD=" + SHORT, "LOG_LEVEL=INFO",
])
def test_pointers_placeholders_and_interpolation_pass(line):
    assert scan_env_text(line) == []


def test_compose_list_and_map_forms():
    text = "\n".join([
        "services:", "  app:", "    environment:",
        f"      - A_TOKEN={SHORT}", "      - B_TOKEN_FILE=/run/b", "      - C_TOKEN",
        f"      D_SECRET: {ENTROPY}", "      E_SECRET: ${E_SECRET:?required}",
        f"    command: [\"run\", \"--db\", \"postgresql://u:{SHORT}@db/x\"]",
        f"      # F_TOKEN={SHORT}",
    ])
    assert codes(scan_compose_text(text)) == [("A_TOKEN", "secret_literal"), ("D_SECRET", "secret_literal"),
                                              ("command", "credential_url")]


@pytest.mark.parametrize("text, name", [
    (f'services:\n  db:\n    environment:\n      "POSTGRES_PASSWORD": {SHORT}', "POSTGRES_PASSWORD"),
    (f"services:\n  db:\n    environment:\n      'POSTGRES_PASSWORD': '{SHORT}'", "POSTGRES_PASSWORD"),
    (f"services:\n  db:\n    environment: {{POSTGRES_PASSWORD: {SHORT}, LOG_LEVEL: info}}", "POSTGRES_PASSWORD"),
    (f'services:\n  db:\n    environment: [LOG_LEVEL=info, "A_TOKEN={SHORT}"]', "A_TOKEN"),
    (f"services:\n  db:\n    environment:\n      POSTGRES_PASSWORD: ${{ENABLE_DB:+{SHORT}}}", "POSTGRES_PASSWORD"),
    (f"A_TOKEN=${{ON:+{ENTROPY}}}", "A_TOKEN"),
])
def test_quoted_keys_flow_maps_and_alternate_values_fail(text, name):
    """Codex: the supported Compose forms a seven-character password can hide in."""
    found = scan_compose_text(text) if text.startswith("services") else scan_env_text(text)
    assert [(f.name, f.code) for f in found] == [(name, "secret_literal")]
    assert SHORT not in str(found[0]) and ENTROPY not in str(found[0])


def test_flow_forms_with_only_interpolation_pass():
    text = 'services:\n  db:\n    environment: {POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}", B_TOKEN: ${B:?x}}'
    assert scan_compose_text(text) == []
    assert scan_env_text("A_TOKEN=${ON:+${A_TOKEN}}") == []


def test_env_backups_fail_without_reading_them(tmp_path):
    (tmp_path / ".env.bak-1").write_text("NOTHING=1\n")
    (tmp_path / ".env").write_text(f"DB_PASSWORD={SHORT}\nOK=1\n")
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    assert sorted(f.code for f in scan_root(tmp_path)) == ["env_backup", "secret_literal"]
    assert main([str(tmp_path)]) == 1


def test_cli_runs_stdlib_only_and_prints_names_only(tmp_path):
    (tmp_path / ".env").write_text(f"DB_PASSWORD={SHORT}\n")
    out = subprocess.run([sys.executable, "-I", str(ROOT / "src/llm_bawt/secret_guard.py"), str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "DB_PASSWORD secret_literal" in out.stdout
    assert SHORT not in out.stdout + out.stderr
    (tmp_path / ".env").write_text("DB_PASSWORD=\n")
    assert subprocess.run([sys.executable, "-I", str(ROOT / "src/llm_bawt/secret_guard.py"), str(tmp_path)],
                          capture_output=True).returncode == 0


def test_tracked_files_are_clean():
    tracked = subprocess.run(["git", "ls-files", "--", "docker-compose*.yml", ".env*"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked
    findings = []
    for name in tracked:
        text = (ROOT / name).read_text()
        findings += (scan_compose_text if name.endswith(".yml") else scan_env_text)(text, name)
    assert [str(f) for f in findings] == []
