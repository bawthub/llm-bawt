"""
CLI Integration Tests — llm and llm-memory commands.

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
LLM_MEMORY = "llm-memory"
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


@pytest.fixture(scope="session")
def nova_msg_id(service_up) -> str | None:
    """A real nova message ID, fetched once per session."""
    if not service_up:
        return None
    r = run(f"{LLM_MEMORY} --bot {BOT} --msg --limit 1")
    return find_uuid(r.stdout + r.stderr)


@pytest.fixture(scope="session")
def nova_summary_id(service_up) -> str | None:
    """A real nova summary ID, fetched once per session."""
    if not service_up:
        return None
    r = run(f"{LLM_MEMORY} --bot {BOT} --msg-summaries --limit 1")
    return find_uuid(r.stdout + r.stderr)


@pytest.fixture(scope="session")
def nova_memory_id(service_up) -> str | None:
    """A real nova memory ID, fetched once per session."""
    if not service_up:
        return None
    r = run(f"{LLM_MEMORY} --bot {BOT} --list-memories --limit 1")
    return find_uuid(r.stdout + r.stderr)


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

    def test_status_local_mode(self):
        o = assert_ok(run(f"{LLM} --status --local"), "--status --local")
        assert no_traceback(o)

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

    def test_print_history_local_requires_service(self):
        """--local + -ph should fail clearly (no local DB fallback anymore)."""
        r = run(f"{LLM} -b {BOT} -ph 3 --local", timeout=15)
        o = out(r)
        assert no_traceback(o)
        # Either it exits with error about needing service, or succeeds (filesystem history)
        if r.returncode != 0:
            assert any(kw in o.lower() for kw in ["service", "history", "require"])

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

    @pytest.mark.llm_call
    def test_query_local_mode(self):
        o = assert_ok(
            run(f"{LLM} --local -m {MODEL} 'what is 2+2'", timeout=60),
            "--local query",
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

    @pytest.mark.service
    @pytest.mark.llm_call
    def test_query_with_command_flag(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(
                f'echo "hello world" | {LLM} --local -m {MODEL}'
                f' -c "echo hello world" "what did that output?" --no-stream',
                timeout=60,
            ),
            "-c command flag",
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
        o = assert_ok(run(f"{LLM} --user nick --user-profile", timeout=15), "--user nick --user-profile")
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
    def test_invalid_bot_fails(self):
        """In local mode, unknown bot slugs should fail with a clear error."""
        r = run(f"{LLM} --local -b nonexistent_bot_xyz 'hello'", timeout=15)
        o = assert_err(r, "invalid bot")
        assert no_traceback(o)
        assert any(kw in o.lower() for kw in ["unknown", "invalid", "not found", "list-bots"])

    def test_invalid_model_local_fails(self):
        r = run(f"{LLM} --local -m nonexistent_model_xyz 'hello'", timeout=15)
        o = assert_err(r, "invalid model --local")
        assert no_traceback(o)

    def test_local_with_missing_default_alias(self):
        """Missing DEFAULT_MODEL_ALIAS should be handled cleanly (error or safe fallback)."""
        r = run(
            f"LLM_BAWT_DEFAULT_MODEL_ALIAS=nonexistent_model_xyz {LLM} --local 'hello'",
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


# ===========================================================================
# llm-memory — Info
# ===========================================================================


class TestLLMMemoryInfo:
    def test_help(self):
        o = assert_ok(run(f"{LLM_MEMORY} --help"), "llm-memory --help")
        assert "Memory and message history" in o
        assert "--stats" in o
        assert "--list-memories" in o
        assert "--msg" in o

    @pytest.mark.service
    def test_stats(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --stats", timeout=15), "--stats")
        assert no_traceback(o)
        assert any(kw in o for kw in ["Memories", "Messages", "Statistics", "stats", "Bot:"])

    @pytest.mark.service
    def test_stats_contains_counts(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --stats", timeout=15), "--stats counts")
        # nova has real history — should show non-zero counts
        assert re.search(r"\d+", o), "Expected numeric counts in --stats output"
        assert no_traceback(o)

    def test_no_bot_without_tty_fails(self):
        """Without --bot and no interactive TTY, should exit with clear error."""
        r = run(f"{LLM_MEMORY} --stats", timeout=10)
        o = assert_err(r, "no --bot")
        assert no_traceback(o)
        assert "bot" in o.lower()


# ===========================================================================
# llm-memory — Memory search
# ===========================================================================


class TestLLMMemorySearch:
    @pytest.mark.service
    def test_search_default(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} 'test search'", timeout=30), "default search")
        assert no_traceback(o)

    @pytest.mark.service
    def test_search_method_embedding(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --method embedding 'test'", timeout=30), "--method embedding")
        assert no_traceback(o)
        # Output may be "No results" in clean environments; command should still
        # execute successfully and present query/search output.
        assert any(kw in o for kw in ["Query:", "No results", "Embedding", "embedding"])

    @pytest.mark.service
    def test_search_method_text(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --method text 'test'", timeout=30), "--method text")
        assert no_traceback(o)

    @pytest.mark.service
    def test_search_method_high_importance(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --method high-importance 'test'", timeout=30), "--method high-importance")
        assert no_traceback(o)

    @pytest.mark.service
    def test_search_method_all(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --method all 'test'", timeout=30), "--method all")
        assert no_traceback(o)
        # "all" runs both embedding and high-importance — should show both sections
        assert "Embedding" in o or "High" in o or "embedding" in o

    @pytest.mark.service
    def test_search_with_min_importance(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(
            run(f"{LLM_MEMORY} --bot {BOT} --method embedding --min-importance 0.8 'test'", timeout=30),
            "--min-importance 0.8",
        )
        assert no_traceback(o)

    @pytest.mark.service
    def test_search_with_limit(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --limit 3 'test'", timeout=30), "--limit 3")
        assert no_traceback(o)


# ===========================================================================
# llm-memory — Memory management
# ===========================================================================


class TestLLMMemoryManagement:
    @pytest.mark.service
    def test_list_memories(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-memories", timeout=15), "--list-memories")
        assert no_traceback(o)

    @pytest.mark.service
    def test_list_memories_with_limit(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-memories --limit 5", timeout=15), "--list-memories --limit 5")
        assert no_traceback(o)

    @pytest.mark.service
    def test_consolidate_dry_run(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --consolidate-dry-run", timeout=30), "--consolidate-dry-run")
        assert no_traceback(o)

    def test_delete_memory_nonexistent_id(self):
        """Deleting a nonexistent memory ID should fail cleanly, never traceback."""
        r = run(f"{LLM_MEMORY} --bot {BOT} --yes --delete-memory 00000000-0000-0000-0000-000000000000", timeout=15)
        o = out(r)
        assert no_traceback(o)

    @pytest.mark.service
    def test_regenerate_embeddings_dry_run(self, service_up):
        """--regenerate-embeddings is safe to call — service handles it."""
        if not service_up:
            pytest.skip("Service not running")
        # This is a potentially slow operation but read-safe; just check it starts
        r = run(f"{LLM_MEMORY} --bot {SAFE_BOT} --regenerate-embeddings --yes", timeout=30)
        o = out(r)
        assert no_traceback(o)


# ===========================================================================
# llm-memory — Entity profiles
# ===========================================================================


class TestLLMMemoryProfiles:
    @pytest.mark.service
    def test_list_profiles(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-profiles", timeout=15), "--list-profiles")
        assert no_traceback(o)

    @pytest.mark.service
    def test_list_attrs_default(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-attrs", timeout=15), "--list-attrs")
        assert no_traceback(o)

    @pytest.mark.service
    def test_list_attrs_for_user(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-attrs nick", timeout=15), "--list-attrs nick")
        assert no_traceback(o)

    @pytest.mark.service
    def test_list_attrs_for_bot(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --list-attrs {BOT}", timeout=15), f"--list-attrs {BOT}")
        assert no_traceback(o)

    def test_delete_attr_nonexistent_id(self):
        """Deleting a nonexistent attribute ID should fail cleanly."""
        r = run(f"{LLM_MEMORY} --bot {BOT} --yes --delete-attr 00000000-0000-0000-0000-000000000000", timeout=15)
        o = out(r)
        assert no_traceback(o)


# ===========================================================================
# llm-memory — Message history
# ===========================================================================


class TestLLMMemoryMessages:
    @pytest.mark.service
    def test_msg_list(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg", timeout=15), "--msg")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_list_with_limit(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg --limit 5", timeout=15), "--msg --limit 5")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_search(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg-search 'hello'", timeout=15), "--msg-search")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_get_valid_id(self, service_up, nova_msg_id):
        if not service_up:
            pytest.skip("Service not running")
        if not nova_msg_id:
            pytest.skip("No message IDs found in nova history")
        o = assert_ok(
            run(f"{LLM_MEMORY} --bot {BOT} --msg-get {nova_msg_id}", timeout=15),
            f"--msg-get {nova_msg_id}",
        )
        assert no_traceback(o)
        assert nova_msg_id[:8] in o or "role" in o.lower() or "content" in o.lower()

    def test_msg_get_invalid_id(self):
        """Invalid ID should handle gracefully — may exit 0 with 'not found' message."""
        r = run(f"{LLM_MEMORY} --bot {BOT} --msg-get nonexistent-id-xyz", timeout=15)
        o = out(r)
        assert no_traceback(o)
        # Either exits with error or prints a not-found message — no crash is the requirement

    @pytest.mark.service
    def test_msg_restore(self, service_up):
        """--msg-restore with --yes should not prompt and complete without crashing."""
        if not service_up:
            pytest.skip("Service not running")
        # Use SAFE_BOT (spark) with --yes to avoid touching nova's forgotten messages
        r = run(f"{LLM_MEMORY} --bot {SAFE_BOT} --msg-restore --yes", timeout=15)
        o = out(r)
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_restore_non_interactive(self, service_up):
        """--msg-restore without --yes and no TTY should cancel gracefully (not EOFError)."""
        if not service_up:
            pytest.skip("Service not running")
        r = run(f"{LLM_MEMORY} --bot {BOT} --msg-restore", timeout=15)
        o = out(r)
        assert no_traceback(o)
        # Should cancel (no TTY → treat as 'N') or say nothing to restore
        assert "EOFError" not in o

    @pytest.mark.service
    def test_msg_summarize_preview(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg-summarize-preview", timeout=30), "--msg-summarize-preview")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_forget_and_restore(self, service_up):
        """Forget 1 message from spark (safe bot), then restore it."""
        if not service_up:
            pytest.skip("Service not running")
        # First check spark has any messages — if not, skip
        r_check = run(f"{LLM_MEMORY} --bot {SAFE_BOT} --msg --limit 1", timeout=15)
        if not find_uuid(r_check.stdout + r_check.stderr):
            pytest.skip(f"{SAFE_BOT} has no messages to forget")

        forget_r = run(f"{LLM_MEMORY} --bot {SAFE_BOT} --yes --msg-forget 1", timeout=15)
        o = assert_ok(forget_r, "--msg-forget 1")
        assert no_traceback(o)

        restore_r = run(f"{LLM_MEMORY} --bot {SAFE_BOT} --yes --msg-restore", timeout=15)
        o2 = assert_ok(restore_r, "--msg-restore after forget")
        assert no_traceback(o2)


# ===========================================================================
# llm-memory — Summaries
# ===========================================================================


class TestLLMMemorySummaries:
    @pytest.mark.service
    def test_msg_summaries(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg-summaries", timeout=15), "--msg-summaries")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_summaries_with_limit(self, service_up):
        if not service_up:
            pytest.skip("Service not running")
        o = assert_ok(run(f"{LLM_MEMORY} --bot {BOT} --msg-summaries --limit 3", timeout=15), "--msg-summaries --limit 3")
        assert no_traceback(o)

    @pytest.mark.service
    def test_msg_summary_get_valid_id(self, service_up, nova_summary_id):
        if not service_up:
            pytest.skip("Service not running")
        if not nova_summary_id:
            pytest.skip("No summary IDs found in nova history")
        o = assert_ok(
            run(f"{LLM_MEMORY} --bot {BOT} --msg-summary-get {nova_summary_id}", timeout=15),
            f"--msg-summary-get {nova_summary_id}",
        )
        assert no_traceback(o)

    def test_msg_summary_get_invalid_id(self):
        r = run(f"{LLM_MEMORY} --bot {BOT} --msg-summary-get nonexistent-id-xyz", timeout=15)
        o = out(r)
        assert no_traceback(o)

    def test_msg_delete_summary_nonexistent(self):
        """Deleting a nonexistent summary should fail cleanly."""
        r = run(f"{LLM_MEMORY} --bot {BOT} --yes --msg-delete-summary 00000000-0000-0000-0000-000000000000", timeout=15)
        o = out(r)
        assert no_traceback(o)


# ===========================================================================
# llm-memory — Error handling
# ===========================================================================


class TestLLMMemoryErrors:
    def test_nonexistent_bot_stats(self):
        """A bot with no history should return stats cleanly (not crash)."""
        r = run(f"{LLM_MEMORY} --bot totally_nonexistent_bot_xyz --stats", timeout=15)
        o = out(r)
        assert no_traceback(o)
        # Either succeeds with 0 counts or fails cleanly
        if r.returncode != 0:
            assert any(kw in o.lower() for kw in ["error", "not found", "unavailable", "bot"])

    def test_nonexistent_bot_list_memories(self):
        r = run(f"{LLM_MEMORY} --bot totally_nonexistent_bot_xyz --list-memories", timeout=15)
        o = out(r)
        assert no_traceback(o)

    def test_nonexistent_bot_msg(self):
        r = run(f"{LLM_MEMORY} --bot totally_nonexistent_bot_xyz --msg", timeout=15)
        o = out(r)
        assert no_traceback(o)
