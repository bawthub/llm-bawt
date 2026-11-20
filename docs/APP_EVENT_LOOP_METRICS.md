# App event-loop watchdog

Each serving app process runs a 50 ms monotonic heartbeat and one daemon watchdog thread. A heartbeat more than 250 ms late counts as one stall, not one per polling tick. The thread captures the serving thread's Python frames while the stall is in progress and emits a single `event_loop_stall` warning with a JSON payload (`lag_ms_at_capture`, `frames`). Only module, basename, line and function are retained; no locals, request data, source text, or full paths. Warnings are limited to one per 30 seconds per process; this does not limit the counters. The watchdog writes its warning to stderr directly, without taking any of the app's shared logging locks. If stderr backs up, stack reporting can pause while heartbeat accounting continues. If the watchdog cannot run (for example a native call holding the GIL), the heartbeat measures the delay when it gets control back, but there may be no useful stack. A sampled frame is evidence of where the loop was, not by itself proof of the underlying cause.

## Prometheus scrape

Scrape `GET /metrics` on the published app port from a separate LAN host. This is the standard Prometheus text endpoint from `prometheus-fastapi-instrumentator`, not JSON. The app does not run or contact a collector. Probe health and external service latency separately from the collector host; an app that is down or blocked cannot report its own outage.

`/metrics` is an explicit exception to the app API-key gate, including in enforce mode. It exposes only process/runtime statistics, bounded HTTP metrics, and watchdog samples, never IDs, request contents, credentials, or stacks. Keep the app port and scrape endpoint on the trusted LAN; do not route `/metrics` through a public ingress. The host guard still applies: include the scrape target's hostname or address in `LLM_BAWT_ALLOWED_HOSTS` when it isn't already allowed.

The registry has the standard process and Python GC collectors. Default HTTP metrics include `http_requests_total`, `http_request_duration_seconds`, and `http_requests_inprogress`. Handler labels derive from route templates; unmatched paths, arbitrary HTTP methods, and scrapes themselves are excluded. Auth and host guard rejections before routing are not counted. Serving is single-process (`llm-service` explicitly uses one Uvicorn worker). Each replica is a separate scrape target; do not set `PROMETHEUS_MULTIPROC_DIR` (or its deprecated alias `prometheus_multiproc_dir`) or run multiple ASGI workers behind one target.

| Metric | Meaning |
|---|---|
| `event_loop_lag_max_seconds` | Maximum scheduling delay recovered in the last 60 s, or the current delay if larger. |
| `event_loop_lag_p99_seconds` | Nearest-rank p99 of heartbeat delays recovered within that window. |
| `event_loop_stalls_total` | Counter of stall episodes at least 250 ms late since process startup. |

Scrape more often than the rolling window. Scrapes do not reset counters or erase a recovered spike. Alert on the **maximum** above `0.5` seconds, not the p99, to catch an isolated stall. Compare times with `event_loop_stall` warnings in app logs. Do not inject a blocking request on a shared service to test the alert; use the isolated ASGI regression and verify the external collector and alert separately.

For the migration, the rebuilt app retains the existing JSON format and auth policy until the startup setting `LLM_BAWT_PROMETHEUS_METRICS=1` is passed to the app container. Each process serves only one format at the same path. Have the Prometheus target and queries ready before enabling the setting and recreating the app. Verify fresh samples externally before retiring the JSON consumer; unset the setting and recreate the app to restore JSON if the handoff fails. Delete the temporary compatibility path once the external collector is accepted.
