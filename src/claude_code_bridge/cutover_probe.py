"""The ``broker`` cutover probe, run inside the claude-code bridge (TASK-1476 §3).

``python -m claude_code_bridge.cutover_probe broker`` makes one Anthropic API
key fetch through the same broker call the bridge's ``anthropic-api`` path
uses (:func:`.provider_key_broker.broker_response`). It prints exactly one
probe-contract line, with the broker's HTTP status and never the key, and
exits 0 whatever the outcome.

The bridge can't import the ``llm_bawt`` package (its ``__init__`` needs app
dependencies), so the stdlib-only contract module is loaded by file path.
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx

logger = logging.getLogger(__name__)

PROBE_ID = "broker"
#: The connected record the broker serves for the bridge's ``anthropic-api`` path.
PROVIDER_ID = "anthropic-api"
_CONTRACT = Path(__file__).resolve().parents[1] / "llm_bawt" / "ops" / "probe_contract.py"


def load_contract() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_cutover_probe_contract", _CONTRACT)
    if spec is None or spec.loader is None:
        raise ImportError("probe contract not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = load_contract()


def classify(fetch: Callable[[str], httpx.Response]) -> tuple[bool, int | None, str]:
    """``(ok, status, code)`` for one broker fetch made with ``fetch``."""
    from .provider_key_broker import BrokerKeyError  # noqa: PLC0415

    try:
        response = fetch(PROVIDER_ID)
    except BrokerKeyError:
        # The only pre-request failure: LLM_BAWT_API_URL is unset.
        return False, None, pc.ERROR
    except httpx.TimeoutException:
        return False, None, pc.TIMEOUT
    except httpx.HTTPError:
        return False, None, pc.UNREACHABLE
    status = response.status_code if pc.is_status(response.status_code) else None
    if status == 503:
        return False, status, pc.NOT_CONFIGURED
    if status in (401, 403):
        return False, status, pc.UNAUTHORIZED
    if not response.is_success:
        return False, status, pc.HTTP_ERROR
    try:
        payload = response.json()
    except ValueError:
        return False, status, pc.UNEXPECTED_RESPONSE
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token.strip():
        return False, status, pc.UNEXPECTED_RESPONSE
    return True, status, pc.OK


def run_probe(argv: list[str], fetch: Callable[[str], httpx.Response] | None = None) -> str:
    """The emit() line for ``argv``. Never raises."""
    if argv != [PROBE_ID]:
        # This container serves only the broker probe.
        probe = argv[0] if len(argv) == 1 and argv[0] in pc.PROBE_IDS else pc.RECORDS_PROBE
        return pc.emit(probe, False, None, pc.ERROR)
    try:
        if fetch is None:
            from .provider_key_broker import broker_response as fetch  # noqa: PLC0415
        ok, status, code = classify(fetch)
    except Exception as error:  # noqa: BLE001 - reported by type on stderr only
        logger.warning("broker probe failed: %s", type(error).__name__)
        ok, status, code = False, None, pc.ERROR
    return pc.emit(PROBE_ID, ok, status, code)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    real_stdout = sys.stdout
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    # Any library print goes to stderr; only the probe line reaches stdout.
    with contextlib.redirect_stdout(sys.stderr):
        line = run_probe(list(args))
    real_stdout.write(line + "\n")
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
