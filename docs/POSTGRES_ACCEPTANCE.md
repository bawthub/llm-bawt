# Isolated PostgreSQL acceptance boundary (TASK-1149)

`make test-pg-acceptance` is the **required** SQL behavior gate. It starts
pinned pgvector/PostgreSQL containers with `--network none`, no published
ports, and a pytest-owned Unix socket. The suite-wide DBAPI guard prevents
connection to operational PostgreSQL; the app-owned schema bootstrap runs
only after selecting and verifying a `*_test` database. Missing Docker/DB,
missing selected suites, zero passes, skips, and xfails fail this gate.
Do not run integration tests against an inherited DB URL or mistake the
marker-deselected unit suite for SQL acceptance.

The separate dependent source change establishes the active-session index
on **fresh** `sessions` tables. An existing unindexed table fails bootstrap
and requires the separately invoked `upgrade_sessions_one_active_index(engine)`
(concurrent index build, no data repair or invalid-index deletion). The
private tests exercise both paths, including concurrent first writes, NULL
user identities, archived coexistence, conflicting definitions, and duplicate
active-row failure. These tests never upgrade the operational database.

## Cases still outside the gate

Collection-only inventory (`pytest -o addopts='' --collect-only -q -m
integration`): 221 integration cases in 16 files. The required gate executes
129; the remaining **92** are:

| File | Cases | Reason for exclusion / required route |
| --- | ---: | --- |
| `test_history_timeline.py::TestLivePostgres` | 13 | 12 date/DST SQL cases use `require_test_db_config()` but have not been wired to the private fixture; the full-scan parity case additionally requires an existing `snark` partition with >=50 real history rows, so it skips on a fresh DB. Replace that corpus with deterministic synthetic, test-owned history before requiring it. Never use the operational history as test input. |
| `test_talk_reply_outbox_pg.py` | 3 | These tests require `POSTGRES_HOST == 127.0.0.1`, then create stores or issue SQL through a TCP-oriented engine. That route is not the private Unix socket fixture. Port them to `private_app_pg`, verify schema creation and cleanup, and include them only after a zero-skip private run. |
| `test_cli_commands.py` | 76 | All cases intentionally fail closed before launching an installed CLI or probing `127.0.0.1:8642`. Of these, 53 carry a `service` mark. This is a **temporary safety hold, not CLI coverage**: allocate a process/service bound to an independently verified isolated address and private database, route the CLI explicitly to it, prevent inherited operational config, and only then replace the hold with tested cleanup. No production service probes. |

Collection-only `-m service` selects 53 tests, all within the blocked CLI
module. Other tests may depend on optional external resources (`llm_call`)
and are not claimed by this PostgreSQL gate.

## CI and operational limits

`.github/workflows/tests.yml` defines distinct unit and private-PG jobs, but
it has not yet run on GitHub, and has **not** been verified
as a required branch-protection status. Earlier observed CI was BawtHub CI,
not llm-bawt CI. A local echo-host disposable-container pass does not establish
GitHub CI status or authorize production DDL, cleanup, restart, or release.
