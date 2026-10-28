"""Cutover credential probes, run inside the app container (TASK-1476 §3).

``python -m llm_bawt.service.cutover_probe <id>`` prints exactly one
:func:`llm_bawt.ops.probe_contract.emit` line and exits 0, whatever the
outcome: the ops worker reads the line, never the exit code or stderr. An
unknown id, bad argv or a crash inside a probe is still one schema line with
code ``error``.

Every credential comes from its stored record through
:func:`llm_bawt.service.connection_secrets.read_connection`, never from env:
the cutover is proving that the record alone works once the env names are
gone. The Crawl4AI probes are the exception by design: that token is
derived from the bootstrap root, never stored (TASK-1604).

Each request is the provider adapter's own key check (``_probe_request``) or
the integration client's status call, so the probe tests what the app
actually sends. Nothing spends model tokens: a "completion" probe is the
provider's free key check, because the catalog has no endpoint to complete
against and a probe may not pick a model of its own.

Output is booleans, an HTTP status and a closed code. No value, body, URL or
exception text reaches stdout: library output is diverted to stderr while a
probe runs.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from llm_bawt.ops import probe_contract as pc

logger = logging.getLogger(__name__)

TIMEOUT_S = 15.0
PROVIDER = "provider"
INTEGRATION = "integration"


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status: int | None
    code: str
    records: dict[str, dict[str, bool]] | None = None

    @classmethod
    def passed(cls, status: int | None = None) -> ProbeResult:
        return cls(True, status, pc.OK)

    @classmethod
    def failed(cls, code: str, status: int | None = None) -> ProbeResult:
        return cls(False, status, code)


class ProbeFailure(Exception):
    """Stops a probe with ``code`` (and the HTTP status, when there was one)."""

    def __init__(self, code: str, status: int | None = None):
        super().__init__(code)
        self.code = code
        self.status = status


# A credential reader: (kind, connection id, FieldSpec) -> ConnectionSecret | None,
# raising ConnectionUnavailable. Defaults to read_connection on the app config.
Reader = Callable[[str, str, Any], Any]


@dataclass
class ProbeContext:
    """What every probe may use. Tests pass fakes; :func:`default_context` the real ones."""

    http: httpx.Client
    read: Reader
    environ: Mapping[str, str]
    derive_crawl4ai_token: Callable[[], str]
    s3_client: Callable[[Any], Any]


class Probe(ABC):
    id: ClassVar[str]

    def __init__(self, ctx: ProbeContext):
        self.ctx = ctx

    @abstractmethod
    def run(self) -> ProbeResult:
        ...

    # --- shared helpers -------------------------------------------------------
    def credential(self, kind: str, connection_id: str, spec: Any) -> Any:
        """The stored record's fields, or ``not_configured``; a store outage is ``error``."""
        from llm_bawt.service.connection_secrets import ConnectionUnavailable  # noqa: PLC0415

        try:
            found = self.ctx.read(kind, connection_id, spec)
        except ConnectionUnavailable as error:
            raise ProbeFailure(pc.ERROR if error.code == "store_unavailable" else pc.NOT_CONFIGURED) from None
        if found is None:
            raise ProbeFailure(pc.NOT_CONFIGURED)
        return found

    def get(self, url: str, *, headers: Mapping[str, str] | None = None,
            params: Mapping[str, Any] | None = None) -> httpx.Response:
        try:
            return self.ctx.http.get(url, headers=dict(headers or {}), params=params, timeout=TIMEOUT_S)
        except httpx.TimeoutException:
            raise ProbeFailure(pc.TIMEOUT) from None
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            # InvalidURL/ValueError: a stored URL httpx can't use.
            raise ProbeFailure(pc.UNREACHABLE) from None


class HttpCheckProbe(Probe):
    """One authenticated GET; status classifies it, :meth:`accept` judges a 2xx."""

    rejected_statuses: ClassVar[frozenset[int]] = frozenset({401, 403})

    @abstractmethod
    def request(self) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        """``(url, headers, params)``, raising :class:`ProbeFailure` when it can't be built."""

    def accept(self, response: httpx.Response) -> str:
        """The code for a 2xx answer."""
        return pc.OK

    def run(self) -> ProbeResult:
        url, headers, params = self.request()
        response = self.get(url, headers=headers, params=params)
        status = response.status_code
        if status in self.rejected_statuses:
            return ProbeResult.failed(pc.UNAUTHORIZED, status)
        if not response.is_success:
            return ProbeResult.failed(pc.HTTP_ERROR, status)
        code = self.accept(response)
        return ProbeResult(code == pc.OK, status, code)


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        raise ProbeFailure(pc.UNEXPECTED_RESPONSE, response.status_code) from None


class ApiKeyProbe(HttpCheckProbe):
    """A provider's own key check (``adapter._probe_request``) with its stored key."""

    adapter: ClassVar[type]

    def request(self) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        from llm_bawt.service.connection_secrets import FieldSpec  # noqa: PLC0415

        found = self.credential(PROVIDER, self.adapter.id, FieldSpec(secret=("api_key",)))
        url, headers = self._request_for(found.secret["api_key"])
        return url, headers, None

    def _request_for(self, key: str) -> tuple[str, dict[str, str]]:
        return self.adapter._probe_request(key)


class AnthropicModelsProbe(ApiKeyProbe):
    id = "anthropic_models"

    @property
    def adapter(self) -> type:  # type: ignore[override]
        from llm_bawt.service.providers.anthropic_key import AnthropicKeyAdapter  # noqa: PLC0415

        return AnthropicKeyAdapter

    def accept(self, response: httpx.Response) -> str:
        data = _json(response)
        models = data.get("data") if isinstance(data, dict) else None
        return pc.OK if isinstance(models, list) and models else pc.UNEXPECTED_RESPONSE


class ZaiProbe(ApiKeyProbe):
    id = "zai_completion"
    # Same statuses the adapter's key check treats as a rejected key.
    rejected_statuses = frozenset({400, 401, 403})

    @property
    def adapter(self) -> type:  # type: ignore[override]
        from llm_bawt.service.providers.bridge_keys import ZaiKeyAdapter  # noqa: PLC0415

        return ZaiKeyAdapter

    def accept(self, response: httpx.Response) -> str:
        from llm_bawt.service.providers.bridge_keys import (  # noqa: PLC0415
            ZaiEnvelopeError,
            is_exact_zai_no_plan_body,
        )

        # z.ai rejects a key inside an HTTP 200 body. "No Coding Plan" also
        # arrives as a 200 failure envelope, but it proves the stored key
        # authenticated, which is all this probe exists to show. It stays
        # ``ok``: the probe line's ``ok`` is true exactly when code is ``ok``,
        # a contract the ops worker image enforces.
        payload = _json(response)
        if is_exact_zai_no_plan_body(response.status_code, payload):
            return pc.OK
        failure = ZaiEnvelopeError.from_payload(payload)
        if failure is None:
            return pc.OK
        return pc.UNAUTHORIZED if failure.unauthorized else pc.UNEXPECTED_RESPONSE


class KimiProbe(ApiKeyProbe):
    id = "kimi_completion"
    rejected_statuses = frozenset({400, 401, 403})

    @property
    def adapter(self) -> type:  # type: ignore[override]
        from llm_bawt.service.providers.bridge_keys import KimiCodingKeyAdapter  # noqa: PLC0415

        return KimiCodingKeyAdapter


class SearchKeyProbe(ApiKeyProbe):
    """Brave / NewsAPI: the search adapter's key check and its own rejected statuses."""

    adapter_name: ClassVar[str]

    @property
    def adapter(self) -> type:  # type: ignore[override]
        from llm_bawt.service.providers import search_keys  # noqa: PLC0415

        return getattr(search_keys, self.adapter_name)

    @property
    def rejected_statuses(self) -> frozenset[int]:  # type: ignore[override]
        return self.adapter.rejected_statuses


class BraveProbe(SearchKeyProbe):
    id = "brave_search"
    adapter_name = "BraveSearchKeyAdapter"


class NewsApiProbe(SearchKeyProbe):
    id = "newsapi_search"
    adapter_name = "NewsApiKeyAdapter"


class IntegrationProbe(HttpCheckProbe):
    """A media integration's status call with its stored ``url`` + ``api_key``."""

    integration_id: ClassVar[str]

    def request(self) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        from llm_bawt.service.connection_secrets import integration_spec  # noqa: PLC0415

        found = self.credential(INTEGRATION, self.integration_id, integration_spec(self.integration_id))
        return self.status_request(found.meta["url"].strip().rstrip("/"), found.secret["api_key"])

    @abstractmethod
    def status_request(self, base_url: str, api_key: str) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        ...

    def accept(self, response: httpx.Response) -> str:
        return pc.OK if isinstance(_json(response), dict) else pc.UNEXPECTED_RESPONSE


class ArrStatusProbe(IntegrationProbe):
    client: ClassVar[str]

    def status_request(self, base_url: str, api_key: str) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        from llm_bawt.media.arr import radarr, sonarr  # noqa: PLC0415

        prefix = {"sonarr": sonarr.SonarrClient, "radarr": radarr.RadarrClient}[self.integration_id].api_prefix
        return f"{base_url}{prefix}/system/status", {"X-Api-Key": api_key, "Accept": "application/json"}, None


class SonarrProbe(ArrStatusProbe):
    id = "sonarr_status"
    integration_id = "sonarr"


class RadarrProbe(ArrStatusProbe):
    id = "radarr_status"
    integration_id = "radarr"


class SabnzbdProbe(IntegrationProbe):
    id = "sabnzbd_status"
    integration_id = "sabnzbd"

    def status_request(self, base_url: str, api_key: str) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        # ``version`` answers without a key, so ask for the (empty-limit) queue.
        return (f"{base_url}/api", {"Accept": "application/json"},
                {"mode": "queue", "limit": 0, "output": "json", "apikey": api_key})

    def accept(self, response: httpx.Response) -> str:
        data = _json(response)
        if isinstance(data, dict) and data.get("status") is False:
            # SABnzbd answers a bad key with HTTP 200 {"status": false, "error": ...}.
            return pc.UNAUTHORIZED
        return pc.OK if isinstance(data, dict) and "queue" in data else pc.UNEXPECTED_RESPONSE


class Crawl4aiProbe(HttpCheckProbe):
    """GET ``/schema``: Crawl4AI's cheapest endpoint that requires the bearer."""

    def base_url(self) -> str:
        from llm_bawt.integrations.web_fetch.client import DEFAULT_CRAWL4AI_URL  # noqa: PLC0415

        # Same resolution as the web_fetch tool's client.
        return ((self.ctx.environ.get("LLM_BAWT_CRAWL4AI_URL") or "").strip() or DEFAULT_CRAWL4AI_URL).rstrip("/")


class Crawl4aiBearerProbe(Crawl4aiProbe):
    """The derived bearer is accepted (200) AND no bearer is refused (401)."""

    id = "crawl4ai_bearer"

    def request(self) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        try:
            token = self.ctx.derive_crawl4ai_token()
        except Exception:  # noqa: BLE001 - any key problem means "no derived token"
            raise ProbeFailure(pc.NOT_CONFIGURED) from None
        if not token:
            raise ProbeFailure(pc.NOT_CONFIGURED)
        return f"{self.base_url()}/schema", {"Authorization": f"Bearer {token}"}, None

    def accept(self, response: httpx.Response) -> str:
        anonymous = self.get(f"{self.base_url()}/schema")
        # Anything but 401 without a bearer means the gate is open (or not this service).
        return pc.OK if anonymous.status_code == 401 else pc.UNEXPECTED_RESPONSE


class WebFetchProbe(Crawl4aiProbe):
    """The web_fetch tool's own token read (the exported file) is accepted by Crawl4AI."""

    id = "web_fetch"

    def request(self) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        from agent_bridge import crawl4ai_token  # noqa: PLC0415

        headers = crawl4ai_token.auth_headers(self.ctx.environ)
        if not headers:
            raise ProbeFailure(pc.NOT_CONFIGURED)
        return f"{self.base_url()}/schema", headers, None


class S3HeadProbe(Probe):
    """HEAD the stored object-store record's bucket."""

    id = "s3_head"

    def run(self) -> ProbeResult:
        from llm_bawt.media.object_store import S3Config  # noqa: PLC0415
        from llm_bawt.service.connection_secrets import integration_spec  # noqa: PLC0415

        found = self.credential(INTEGRATION, "object-store", integration_spec("object-store"))
        cfg = S3Config(
            endpoint=found.meta["endpoint"].strip(),
            region=found.meta["region"].strip(),
            bucket=found.meta["bucket"].strip(),
            access_key=found.secret["access_key_id"].strip(),
            secret_key=found.secret["secret_access_key"].strip(),
            max_attempts=1,
        )
        try:
            from botocore.exceptions import (  # noqa: PLC0415
                ClientError,
                ConnectTimeoutError,
                EndpointConnectionError,
                ReadTimeoutError,
            )
        except ImportError:
            return ProbeResult.failed(pc.ERROR)
        try:
            response = self.ctx.s3_client(cfg).head_bucket(Bucket=cfg.bucket)
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            status = status if isinstance(status, int) and not isinstance(status, bool) else None
            return ProbeResult.failed(pc.UNAUTHORIZED if status in (401, 403) else pc.HTTP_ERROR, status)
        except (ConnectTimeoutError, ReadTimeoutError):
            return ProbeResult.failed(pc.TIMEOUT)
        except EndpointConnectionError:
            return ProbeResult.failed(pc.UNREACHABLE)
        status = (response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
        if not isinstance(status, int) or isinstance(status, bool):
            # A HEAD that returned no status proves nothing about the bucket.
            return ProbeResult.failed(pc.HTTP_ERROR)
        if not 200 <= status < 300:
            return ProbeResult.failed(pc.HTTP_ERROR, status)
        return ProbeResult.passed(status)


def record_sources() -> dict[str, str]:
    """``{record key: connection kind}`` for every key the ``records`` probe reports."""
    from llm_bawt.env_import import SPECS  # noqa: PLC0415

    return {**{spec.destination: spec.kind for spec in SPECS}, "anthropic-api": PROVIDER, "claude": PROVIDER}


class RecordsProbe(Probe):
    """``configured`` (a row exists) and ``connected`` (connected and readable) per record."""

    id = pc.RECORDS_PROBE

    def _spec(self, kind: str, key: str) -> Any:
        from llm_bawt.service.connection_secrets import FieldSpec, integration_spec  # noqa: PLC0415

        if kind == INTEGRATION:
            return integration_spec(key)
        # The claude record is an OAuth bundle, not an API key: connected is its status.
        return FieldSpec(secret=()) if key == "claude" else FieldSpec(secret=("api_key",))

    def run(self) -> ProbeResult:
        from llm_bawt.service.connection_secrets import ConnectionUnavailable  # noqa: PLC0415

        records: dict[str, dict[str, bool]] = {}
        for key, kind in sorted(record_sources().items()):
            try:
                found = self.ctx.read(kind, key, self._spec(kind, key))
            except ConnectionUnavailable as error:
                if error.code == "store_unavailable":
                    return ProbeResult.failed(pc.ERROR)
                records[key] = {"configured": True, "connected": False}
                continue
            records[key] = {"configured": found is not None, "connected": found is not None}
        return ProbeResult(True, None, pc.OK, records)


PROBES: dict[str, type[Probe]] = {cls.id: cls for cls in (
    RecordsProbe, Crawl4aiBearerProbe, AnthropicModelsProbe, ZaiProbe, KimiProbe, BraveProbe, NewsApiProbe,
    S3HeadProbe, SonarrProbe, RadarrProbe, SabnzbdProbe, WebFetchProbe,
)}
#: The one id this module doesn't serve: the claude-code bridge runs it.
BRIDGE_PROBES = frozenset({"broker"})


def default_context() -> ProbeContext:
    from llm_bawt.media.object_store import S3BlobBackend  # noqa: PLC0415
    from llm_bawt.service.connection_secrets import read_connection  # noqa: PLC0415
    from llm_bawt.service.crawl4ai_token_export import _derive  # noqa: PLC0415
    from llm_bawt.utils.config import Config  # noqa: PLC0415

    config = Config()
    return ProbeContext(
        http=httpx.Client(follow_redirects=False),
        read=lambda kind, connection_id, spec: read_connection(config, kind, connection_id, spec),
        environ=os.environ,
        derive_crawl4ai_token=_derive,
        s3_client=lambda cfg: S3BlobBackend(cfg).client,
    )


def error_line(probe_id: str) -> str:
    """The ``error`` line for ``probe_id``; an id outside the contract reports as ``records``.

    The worker only runs ids it knows and matches ``probe`` against the id it
    asked for, so an unknown id still yields one schema line, never a traceback.
    """
    return pc.emit(probe_id if probe_id in pc.PROBE_IDS else pc.RECORDS_PROBE, False, None, pc.ERROR)


def run_probe(probe_id: str, ctx_factory: Callable[[], ProbeContext] | None = None) -> str:
    """The emit() line for ``probe_id``. Never raises."""
    ctx_factory = ctx_factory or default_context
    cls = PROBES.get(probe_id)
    if cls is None:
        # Unknown, or ``broker`` (the bridge's probe): this container can't answer it.
        return error_line(probe_id)
    try:
        result = cls(ctx_factory()).run()
        return pc.emit(probe_id, result.ok, result.status, result.code, records=result.records)
    except ProbeFailure as failure:
        return pc.emit(probe_id, False, failure.status if pc.is_status(failure.status) else None, failure.code)
    except Exception as error:  # noqa: BLE001 - reported by type on stderr only
        logger.warning("cutover probe %s failed: %s", probe_id, type(error).__name__)
        return error_line(probe_id)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    real_stdout = sys.stdout
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    # Any library print goes to stderr; only the probe line reaches stdout.
    with contextlib.redirect_stdout(sys.stderr):
        line = run_probe(args[0]) if len(args) == 1 else error_line(pc.RECORDS_PROBE)
    real_stdout.write(line + "\n")
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
