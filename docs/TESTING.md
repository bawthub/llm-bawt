# Test suites (TASK-1189)

Use the Make targets below on a Unix runner with Python 3.12+, `uv`, and Docker
(for PostgreSQL). `make` and CI invoke the same `scripts/test_suite.py` runner.
It rejects concurrent suites on the same host and exits nonzero on failed
selection, missing PostgreSQL suites, PostgreSQL/local skips, or a slow unit.
Do not use `pytest -m ""`: that bypasses the external-target boundary.

| Command | Selection | Requirements |
| --- | --- | --- |
| `make test` | Hermetic unit tests; **setup + call + teardown <1 s** per case | Python only |
| `make test-pg-acceptance` | All manifest-listed private SQL tests, zero skips | Docker with pinned pgvector image |
| `make test-offline-integration` | Real local HTTP/thread lifecycle and real SDK wire integration, zero skips | Loopback/in-memory transport |
| `make test-sdk-probe` | Optional installed SDK subprocess against fake localhost upstream, zero skips | Installed SDK CLI; never operational bot/provider |
| `make test-integration` | Private SQL, then local HTTP/thread tests | Docker + loopback |
| `make test-all` | **Every offline test exactly once**: unit, SQL, local | Docker + loopback |
| `make test-cov` | Hermetic coverage (not a latency gate) | Python only |
| `make test-profile` | Detailed hermetic setup/call/teardown timings | Python only |
| `make test-target NODES=tests/test_file.py::test_name` | Explicit offline target, no skips | Relevant private fixtures |
| `make test-service` / `make test-paid` | **Refuse** until a disposable service/DB and credential scope are provisioned and verified | Never run against live :8642 |

The required offline gate compares pytest's safe collection with the union of
three disjoint suites and reports selected counts before executing any test.
`tests/suite_manifest.py` owns the SQL selectors used by both pytest's
`--require-pg-acceptance` guard and the runner. Unknown new integration tests
make the gate fail until assigned to a suite. `service`, `llm_call`, and
`external` are excluded; optional `sdk_probe` cases run with `make test-sdk-probe`
using fake local providers and never run implicitly. The unit gate now requires
zero skips; its former ten skips were opt-in SDK probes misclassified as units.
Raw pytest still has its historical hermetic default but is not a full gate.

The PostgreSQL fixture runs one pinned, networkless Docker container per pytest
session (one per xdist worker if workers are used), bound only to a
pytest-owned Unix socket with **no published port**. Each module receives its
own randomly named `_test` database; function-level schema/DB tests retain
independent transactions/databases. Only the `postgres` maintenance database
is used for CREATE/DROP DATABASE and connection identity checks. The DBAPI
connection guard and test-specific configuration remain mandatory; module DBs
are force-dropped after engine disposal, then the container stops. No test may
use an operational service or an inherited PostgreSQL address.

Current external-only tests are the CLI module (its subprocess targets are
hardcoded for an unverified service and can mutate real bot data) and the
read-only live-bot timeline parity class. Their `service`/`external` markers
make them explicit omissions from offline coverage. A paid-LLM target must
also be provisioned with separately scoped credentials before that suite can
be enabled. An explicit command that fails safely is preferable to one that
appears to pass by skipping every case or that contacts production.

For profiling, compare `make test-profile` and the unit gate on a quiet runner.
The initial default-suite sample (3,830 passed, 10 skipped, 404 deselected,
156.97 s) had 60 cases above 1 s when fixture time was included. A later
isolated host sample (3,832 passed, 10 skipped, 404 deselected, 46.93 s) had
only three: two real HTTP shutdowns (~2 s each, correctly classified as
local integration) and one delivery poll (~1 s, now driven by an event).
A real OpenAI-SDK-over-mock-HTTP wire test sometimes exceeded 1 s in later
runs; it is also local integration, not a pure unit.
Transient load should be measured again rather than treating a slow run as
proof that each test's own work takes a second.
