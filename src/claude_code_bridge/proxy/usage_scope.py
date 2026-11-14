"""Non-secret account scope shared by app and bridge quota snapshots."""
import hashlib

SNAPSHOT_PREFIX = "llm_bawt:usage:openai_chatgpt:snapshot"


def account_scope(account_id: str | None) -> str | None:
    if not isinstance(account_id, str) or not account_id.strip():
        return None
    return hashlib.sha256(account_id.strip().encode()).hexdigest()


def snapshot_key(scope: str) -> str:
    return f"{SNAPSHOT_PREFIX}:{scope}"
