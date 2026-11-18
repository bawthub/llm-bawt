"""TASK-2406: the Redis URL (it carries the password) reaches each service under one name.

The ops reconcile refuses a declared compose secret repeated under another
variable (compose_render.check_compose_secrets), so a ``REDIS_URL=${LLM_BAWT_REDIS_URL}``
copy blocks the catalog reconcile of that service and of every house that
extends it. TASK-2439 removed the local bridge's copy; this keeps them all out.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECRET = "LLM_BAWT_REDIS_URL"


def _environment(service: dict) -> list[tuple[str, str | None]]:
    env = service.get("environment") or []
    if isinstance(env, dict):
        return [(k, None if v is None else str(v)) for k, v in env.items()]
    pairs = []
    for item in env:
        name, sep, value = str(item).partition("=")
        pairs.append((name, value if sep else None))
    return pairs


def test_no_service_copies_the_redis_secret_under_another_name():
    offenders = []
    for path in sorted(ROOT.glob("docker-compose*.yml")):
        services = (yaml.safe_load(path.read_text()) or {}).get("services") or {}
        for name, service in services.items():
            for key, value in _environment(service or {}):
                if key != SECRET and value and SECRET in value:
                    offenders.append(f"{path.name}:{name}:{key}")
    assert offenders == []


def test_agent_bridges_read_the_secret_name_first():
    for rel in ("src/claude_code_bridge/__main__.py", "src/codex_bridge/__main__.py",
                "src/local_model_bridge/__main__.py"):
        source = (ROOT / rel).read_text()
        assert 'os.getenv("LLM_BAWT_REDIS_URL") or os.getenv("REDIS_URL"' in source, rel
