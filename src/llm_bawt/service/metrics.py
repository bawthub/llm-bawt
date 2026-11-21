"""Bounded, process-local instrumentation for the app's LAN scrape endpoint."""

import os
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_fastapi_instrumentator import routing
from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
)
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_bawt.retrieval import metrics as retrieval_metrics

# The library labels requests by method as well as by handler. HTTP allows
# arbitrary method tokens, even on a known route that responds with 405.
_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT"})
PROMETHEUS_METRICS_ENV = "LLM_BAWT_PROMETHEUS_METRICS"


class _BoundedHTTPMetrics(PrometheusInstrumentatorMiddleware):
    def __init__(self, app: ASGIApp, *, registry: CollectorRegistry) -> None:
        super().__init__(
            app,
            should_ignore_untemplated=True,
            inprogress_labels=True,
            excluded_handlers=(r"^/metrics/?$",),
            registry=registry,
        )
        # 8.1 creates its in-progress gauge in the global registry even when
        # given a custom registry. Use the stock middleware's inc/dec lifecycle
        # with the same gauge registered alongside the other app metrics.
        self.inprogress = Gauge(
            self.inprogress_name,
            "Number of HTTP requests in progress.",
            labelnames=("method", "handler"),
            registry=registry,
        )
        self.should_instrument_requests_inprogress = True

    def _get_handler(self, request: Request) -> tuple[str, bool]:
        # 8.1's public get_route_name returns a raw URL path on slash redirects.
        # Only accept an actual route template; redirects and misses are ignored.
        template = routing._get_route_name(request.scope, request.app.routes)
        return (template, True) if template else ("", False)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in _HTTP_METHODS:
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


class _LoopMetrics:
    def __init__(self, app: FastAPI):
        self.app = app

    def collect(self) -> Iterator[Metric]:
        monitor = getattr(self.app.state, "loop_watchdog", None)
        if monitor is None:
            return
        snapshot = monitor.snapshot()
        yield GaugeMetricFamily(
            "event_loop_lag_max_seconds", "Maximum event loop delay over the last 60 seconds, including any active delay.",
            value=snapshot["event_loop_lag_max_ms"] / 1000,
        )
        yield GaugeMetricFamily(
            "event_loop_lag_p99_seconds", "99th percentile recovered event loop delay over the last 60 seconds.",
            value=snapshot["event_loop_lag_p99_ms"] / 1000,
        )
        yield CounterMetricFamily(
            "event_loop_stalls", "Event loop stall episodes since this process started.",
            value=snapshot["event_loop_stalls_total"],
        )


async def _legacy_scrape(request: Request) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    monitor = getattr(request.app.state, "loop_watchdog", None)
    if monitor is None:
        return JSONResponse({"error": "metrics_unavailable"}, status_code=503, headers=headers)
    return JSONResponse(monitor.snapshot(), headers=headers)


def install(app: FastAPI) -> bool:
    """Register exactly one scrape route; select its format at app startup."""
    if os.getenv(PROMETHEUS_METRICS_ENV) != "1":
        # Temporary rollback/staging path until the external collector is verified.
        app.add_api_route("/metrics", _legacy_scrape, methods=["GET"], include_in_schema=False)
        return False
    # prometheus_client also recognizes the deprecated lowercase spelling.
    for name in ("PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir"):
        if name in os.environ:
            raise RuntimeError(f"App metrics require single-process serving; unset {name}")

    # Use the same standard collectors as prometheus_client.REGISTRY, without
    # sharing HTTP series or app lifespans across ASGI app instances.
    registry = CollectorRegistry()
    GCCollector(registry=registry)
    PlatformCollector(registry=registry)
    ProcessCollector(registry=registry)
    registry.register(_LoopMetrics(app))
    retrieval_metrics.register(registry)

    app.add_middleware(_BoundedHTTPMetrics, registry=registry)
    # A sync endpoint collects in FastAPI's worker pool, off the app event loop.
    def scrape() -> Response:
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    app.add_api_route("/metrics", scrape, methods=["GET"], include_in_schema=False)
    return True
