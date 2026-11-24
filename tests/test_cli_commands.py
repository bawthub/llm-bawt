"""
CLI Integration Tests — llm command.

Invokes the actual installed CLI entrypoints via subprocess.
No llm_bawt code is imported; all assertions are on stdout/stderr/exit codes.

Marks:
  @pytest.mark.service  — requires the llm-bawt service running at :8642
  @pytest.mark.llm_call — makes real LLM API calls (slow, uses API credits)

Run all:              pytest tests/test_cli_commands.py -v
Skip LLM API calls:   pytest tests/test_cli_commands.py -v -m "not llm_call"
Service tests only:   pytest tests/test_cli_commands.py -v -m service
"""

import re
import subprocess
import urllib.request

import pytest

# Every test in this module shells out to the real installed CLI entrypoints
# via subprocess, and many reach the live Postgres DB (bot registry, memory
# stats) or the running service at :8642. None of it is hermetic, so the whole
# module is marked `integration` and excluded from the default `pytest` run.
# Run it explicitly with `uv run pytest -m integration`. Per-test `service` /
# `llm_call` marks still apply for finer slicing.
pytestmark = [pytest.mark.integration, pytest.mark.service]


def assert_private_cli_target():
    # The installed CLI and hardcoded localhost:8642 may address the running
    # operational service, even when this pytest process has DBAPI guards.
    # Keep these tests blocked until an isolated CLI/service target is wired.
    pytest.fail("CLI integration needs a separately verified isolated service; localhost:8642 is not a test target")


@pytest.fixture(autouse=True)
def require_isolated_cli_service():
    assert_private_cli_target()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM = "llm"
BOT = "nova"
MODEL = "grok-4-fast"          # valid alias available in all environments
SAFE_BOT = "spark"             # historical fixture only; never wipe outside an isolated service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Only launch CLIs after proving their service/DB target is private."""
    assert_private_cli_target()
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )


def out(r: subprocess.CompletedProcess) -> str:
    """Return combined stdout + stderr, stripped."""
    return (r.stdout + r.stderr).strip()


def assert_ok(r: subprocess.CompletedProcess, label: str = "") -> str:
    """Assert exit 0. Return combined output for further assertions."""
    o = out(r)
    assert r.returncode == 0, (
        f"[{label}] expected exit 0, got {r.returncode}\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    return o


def assert_err(r: subprocess.CompletedProcess, label: str = "") -> str:
    """Assert non-zero exit. Return combined output for further assertions."""
    o = out(r)
    assert r.returncode != 0, (
        f"[{label}] expected non-zero exit, got 0\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    return o


def no_traceback(o: str) -> bool:
    """Return True if output contains no Python traceback."""
    return "Traceback (most recent call last)" not in o


def find_uuid(text: str) -> str | None:
    """Extract the first UUID from text."""
    m = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text
    )
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def service_up() -> bool:
    """Only probe a separately verified private service; never the live :8642."""
    assert_private_cli_target()  # Session fixtures run before function autouse fixtures.
    try:
        with urllib.request.urlopen("http://127.0.0.1:8642/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ===========================================================================
# llm — Info / Configuration
# ===========================================================================


class TestLLMInfo:
    def test_help(self):
        o = assert_ok(run(f"{LLM} --help"), "--help")
        assert "Query LLM models" in o
        assert "--list-models" in o
        assert "--add-bot" in o
        assert "--list-bots" in o
        assert no_traceback(o)

    def test_list_models(self):
        o = assert_ok(run(f"{LLM} --list-models"), "--list-models")
        assert "grok-4-fast" in o
        assert no_traceback(o)

    def test_list_models_shows_service_section(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --list-models"), "--list-models service section")
        assert "Service Models" in o
        assert no_traceback(o)

    def test_list_config(self):
        o = assert_ok(run(f"{LLM} --list-config"), "--list-config")
        assert "USE_SERVICE" in o
        assert "DEFAULT_BOT" in o
        assert "POSTGRES_HOST" in o
        assert no_traceback(o)

    def test_list_bots(self):
        o = assert_ok(run(f"{LLM} --list-bots"), "--list-bots")
        assert "nova" in o
        assert "spark" in o
        assert no_traceback(o)

    def test_list_bots_shows_default(self):
        o = assert_ok(run(f"{LLM} --list-bots"), "--list-bots default marker")
        # Default bot should be marked
        assert "⭐" in o or "default" in o.lower()


# ===========================================================================
# llm — Status
# ===========================================================================


class TestLLMStatus:
    @pytest.mark.service
    def test_status_via_service(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --status"), "--status")
        assert no_traceback(o)
        # Should contain service/memory info
        assert any(kw in o for kw in ["Service", "Memory", "Status", "service", "memory"])


    @pytest.mark.service
    def test_status_shows_postgres_info(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --status"), "--status postgres")
        # Service status should expose DB connection info
        assert any(kw in o for kw in ["postgres", "Postgres", "database", "Database", "connected", "Connected"])
        assert no_traceback(o)


# ===========================================================================
# llm — History
# ===========================================================================


class TestLLMHistory:
    @pytest.mark.service
    def test_print_history_default_pairs(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} -b {BOT} -ph", timeout=15), "-ph default")
        assert no_traceback(o)

    @pytest.mark.service
    def test_print_history_with_count(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} -b {BOT} -ph 3", timeout=15), "-ph 3")
        assert no_traceback(o)

    @pytest.mark.service
    def test_print_history_all(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} -b {BOT} -ph -1", timeout=15), "-ph -1")
        assert no_traceback(o)


    @pytest.mark.service
    def test_delete_history_safe_bot(self, service_up):
        """Delete history for spark (no memory bot — safe to wipe)."""
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} -b {SAFE_BOT} -dh", timeout=15), "-dh spark")
        assert no_traceback(o)


# ===========================================================================
# llm — Queries  (real LLM API calls)
# ===========================================================================


class TestLLMQuery:
    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_via_service(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} -m {MODEL} -b {BOT} --no-stream 'what is 2+2'", timeout=60),
            "query via service",
        )
        assert "4" in o
        assert no_traceback(o)


    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_no_stream(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} -m {MODEL} -b {BOT} --no-stream 'say only: pong'", timeout=60),
            "--no-stream",
        )
        assert no_traceback(o)

    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_plain_output(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} -m {MODEL} -b {BOT} --plain --no-stream 'say only: pong'", timeout=60),
            "--plain",
        )
        # Plain mode strips rich box decorations
        assert "╭" not in o
        assert "╰" not in o
        assert no_traceback(o)

    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_explicit_service_flag(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} --service -m {MODEL} -b {BOT} --no-stream 'ping'", timeout=60),
            "--service flag",
        )
        assert no_traceback(o)

    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_explicit_bot(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} -b {SAFE_BOT} -m {MODEL} --no-stream 'ping'", timeout=60),
            f"-b {SAFE_BOT}",
        )
        assert no_traceback(o)



# ===========================================================================
# llm — Jobs
# ===========================================================================


class TestLLMJobs:
    @pytest.mark.service
    def test_job_status(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --job-status", timeout=15), "--job-status")
        assert no_traceback(o)
        assert any(kw in o for kw in ["maintenance", "consolidation", "decay", "summarization", "Job", "job"])

    @pytest.mark.service
    def test_run_job_profile_maintenance(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --run-job profile_maintenance", timeout=20), "--run-job profile_maintenance")
        assert no_traceback(o)
        assert any(kw in o.lower() for kw in ["trigger", "job", "profile_maintenance"])

    @pytest.mark.service
    def test_run_job_memory_consolidation(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --run-job memory_consolidation", timeout=20), "--run-job memory_consolidation")
        assert no_traceback(o)

    @pytest.mark.service
    def test_run_job_memory_decay(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --run-job memory_decay", timeout=20), "--run-job memory_decay")
        assert no_traceback(o)

    @pytest.mark.service
    def test_run_job_history_summarization(self, service_up):
        """history_summarization was missing from old hardcoded list — regression test."""
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --run-job history_summarization", timeout=20), "--run-job history_summarization")
        assert no_traceback(o)

    def test_run_job_invalid_type(self):
        r = run(f"{LLM} --run-job totally_invalid_job_xyz", timeout=15)
        o = out(r)
        assert no_traceback(o)
        # Should mention valid types or "unknown" — never a traceback
        assert any(kw in o.lower() for kw in ["unknown", "invalid", "valid", "error"])


# ===========================================================================
# llm — Users / Profiles
# ===========================================================================


class TestLLMUsers:
    @pytest.mark.service
    def test_list_users(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --list-users", timeout=15), "--list-users")
        assert no_traceback(o)

    @pytest.mark.service
    def test_user_profile(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --user-profile", timeout=15), "--user-profile")
        assert no_traceback(o)

    @pytest.mark.service
    def test_user_profile_with_explicit_user(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --user operator --user-profile", timeout=15), "--user operator --user-profile")
        assert no_traceback(o)


# ===========================================================================
# llm — Runtime Settings
# ===========================================================================


class TestLLMSettings:
    @pytest.mark.service
    def test_settings_list_bot_scope(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM} --settings-list", timeout=15), "--settings-list")
        assert no_traceback(o)

    @pytest.mark.service
    def test_settings_list_global_scope(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} --settings-scope global --settings-list", timeout=15),
            "--settings-scope global --settings-list",
        )
        assert no_traceback(o)

    @pytest.mark.service
    def test_settings_list_explicit_bot(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM} -b {BOT} --settings-list", timeout=15),
            f"-b {BOT} --settings-list",
        )
        assert no_traceback(o)

    @pytest.mark.service
    def test_settings_set_and_delete(self, service_up):
        """Round-trip: set a test key, verify it appears, then delete it."""
        if not service_up:
            pytest.skip("Service not running")
        key = "test_cli_integration_key"
        val = "test_value_42"

        set_r = run(f"{LLM} -b {BOT} --settings-set {key} {val}", timeout=15)
        assert_ok(set_r, "--settings-set")

        list_o = assert_ok(run(f"{LLM} -b {BOT} --settings-list", timeout=15), "--settings-list after set")
        assert key in list_o

        del_r = run(f"{LLM} -b {BOT} --settings-delete {key}", timeout=15)
        assert_ok(del_r, "--settings-delete")

        list_o2 = assert_ok(run(f"{LLM} -b {BOT} --settings-list", timeout=15), "--settings-list after delete")
        assert key not in list_o2


# ===========================================================================
# llm — Error handling
# ===========================================================================


class TestLLMErrors:
    def test_missing_default_alias_missing_default_alias(self):
        """Missing DEFAULT_MODEL_ALIAS should be handled cleanly (error or safe fallback)."""
        r = run(
            f"LLM_BAWT_DEFAULT_MODEL_ALIAS=nonexistent_model_xyz {LLM} 'hello'",
            timeout=15,
        )
        o = out(r)
        # Must never crash with traceback; implementation may either hard-fail
        # or gracefully fall back to another available model.
        assert no_traceback(o)
        if r.returncode != 0:
            assert any(kw in o.lower() for kw in ["model", "invalid", "unknown", "not available"])

    @pytest.mark.service
    def test_service_unavailable_when_explicitly_set(self, service_up):
        """When USE_SERVICE=True (env) and service is down, must hard-error.

        This test only runs when service IS up to validate normal behavior;
        the inverse (service down) is hard to test without stopping the service.
        Verified manually: exits with clear message when service unreachable.
        """
        if not service_up:
            pytest.skip("Service not running — cannot test normal service path")
        # If service is up, --service flag should succeed
        o = assert_ok(
            run(f"{LLM} --service --list-bots", timeout=15),
            "--service flag with service up",
        )
        assert no_traceback(o)
