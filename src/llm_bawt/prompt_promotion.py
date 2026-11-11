"""Portable-field validation for explicitly approved prompt fixture exports.

This is a conservative preflight, not proof that prose is free of private facts.
Operators supply installation terms and review the selected diff before shipping.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any

from .prompt_template_text import extract_placeholders

PRIVATE_CATEGORIES = frozenset({"persona", "install", "user", "bot", "secret", "credential", "environment"})
PRIVATE_PREFIXES = ("persona.", "user.", "bot.", "secret.", "credential.", "environment.")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEY = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,254}")
_CATEGORY = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_DOCUMENTATION_NETS = tuple(ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))


def private_content_reasons(text: str) -> list[str]:
    """Return categories, never the matching credential/private text itself."""
    reasons = []
    patterns = {
        "credential-like content": (
            r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"
            r"|(?i:\bBearer\s+[A-Za-z0-9_.~-]{16,})"
            r"|\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}"
            r"|(?i:\b(?:api[_-]?key|password|client[_-]?secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_./+-]{12,})"
        ),
        "personal filesystem path": r"(?:/home/|/Users/)[A-Za-z0-9_.-]+(?:/|\b)",
        "credential-bearing URL": r"https?://[^\s/@:]+:[^\s/@]+@",
    }
    for reason, pattern in patterns.items():
        if re.search(pattern, text):
            reasons.append(reason)
    for value in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", text):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if address.is_private and not any(address in net for net in _DOCUMENTATION_NETS):
            reasons.append("private network address")
            break
    return reasons


def validate_promotion_item(item: dict[str, Any]) -> list[str]:
    """Validate the allowlisted public fields before any fixture file is written."""
    errors: list[str] = []
    key = item.get("key")
    if not isinstance(key, str) or not _KEY.fullmatch(key) or key.lower().startswith(PRIVATE_PREFIXES):
        errors.append("invalid or private prompt key")
    if item.get("scope_type") != "global" or item.get("scope_id") != "*":
        errors.append("only exact global prompts may be promoted")
    if item.get("sync_class") != "product":
        errors.append("prompt is install-only")
    category = item.get("category")
    if not isinstance(category, str) or not _CATEGORY.fullmatch(category) or category.lower() in PRIVATE_CATEGORIES:
        errors.append("invalid or private category")
    title = item.get("title")
    if not isinstance(title, str) or not title.strip() or len(title) > 255:
        errors.append("invalid title")
    if item.get("format") not in {"plain_text", "json_instruction"}:
        errors.append("unsupported prompt format")
    body = item.get("body")
    if not isinstance(body, str) or not body.strip():
        errors.append("prompt body cannot be empty")
    required = item.get("required_vars")
    if (not isinstance(required, list)
            or any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in required)):
        errors.append("required_vars must be a list of identifiers")
    elif len(set(required)) != len(required):
        errors.append("duplicate required_vars")
    # Match PromptResolver.validate: an empty required list is an open optional
    # variable contract (e.g. tag-context consumers), not "no variables allowed".
    elif required and isinstance(body, str) and set(required) != set(extract_placeholders(body)):
        errors.append("placeholders do not match required_vars")
    # Only these fields are exported; writer/history/arbitrary tenant metadata
    # are intentionally absent from this scan and from the resulting fixture.
    public_text = "\n".join(value for value in (key, title, category, body) if isinstance(value, str))
    errors.extend(private_content_reasons(public_text))
    return errors
