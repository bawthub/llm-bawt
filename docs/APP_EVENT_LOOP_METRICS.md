# App event-loop watchdog

Each serving app process runs a 50 ms monotonic heartbeat and one daemon watchdog thread. A heartbeat more than 250 ms late counts as one stall, not one per polling tick. The thread captures the serving thread's Python frames while the stall is in progress and emits a single `event_loop_stall` warning with a JSON payload (`lag_ms_at_capture`, `frames`). Only module, basename, line and function are retained; no locals, request data, source text, or full paths. Warnings are limited to one per 30 seconds per process; this does not limit the counters. The watchdog writes its warning to stderr directly, without taking any of the app's shared logging locks. If stderr backs up, stack reporting can pause while heartbeat accounting continues. If the watchdog cannot run (for example a native call holding the GIL), the heartbeat measures the delay when it gets control back, but there may be no useful stack. A sampled frame is evidence of where the loop was, not by itself proof of the underlying cause.

`GET /metrics` follows the existing API authentication policy and returns no-store numeric JSON. App liveness does not depend on a collector, and the route does not contact other services. It returns 503 before the serving lifespan is active.

| Field | Meaning |
|---|---|
| `event_loop_lag_max_ms` | Maximum scheduling delay recovered in the last 60 s, or the current delay if larger. |
| `event_loop_lag_p99_ms` | Nearest-rank p99 of heartbeat delays recovered within that window. |
| `event_loop_stalls_total` | Stall episodes exceeding 250 ms since this process started; resets on restart. |
| `event_loop_window_seconds` | Length of the rolling lag window. |

Scrape more often than the rolling window and let the time-series store retain the results. Scrapes do not reset counters or erase a recovered spike. A scrape cannot answer while the same app loop is stalled; scrape failures and process liveness are separate signals. Alert on the **maximum**, not the p99, to catch an isolated stall exceeding 500 ms. Compare times with `event_loop_stall` warnings in app logs. Do not inject a blocking request on a shared service to test the alert; use the isolated ASGI regression and validate the collector and alert query separately.
