"""The agent vault's master password (TASK-2386).

``bw-secret`` unlocks the agents' Vaultwarden account with this password. It
used to sit in ``.env`` where every bot could read it; now it is an encrypted
record like every other credential, and ``bw-secret`` fetches it from the
vault route (``GET /v1/providers/vaultwarden/password``), which only the
always-enforced ``vault`` bridge gate opens.

Bitwarden can't check a master password without a full client login, so the
connect step stores the value as given; ``bw-secret`` is the live check.
"""

from __future__ import annotations

from .api_key import ApiKeyAdapter


class VaultwardenAdapter(ApiKeyAdapter):
    id = "vaultwarden"
    label = "Agent vault (Vaultwarden)"
    operator_only = True
    # Only the vault-gated /password route returns it; the generic token
    # broker's BROKER gate (which an operator may leave open) must not.
    generic_broker = False

    def descriptor(self) -> dict:
        return {
            **super().descriptor(),
            "category": "deploy",
            "credential_label": "Vault master password",
            "description": "Unlocks the agents' Vaultwarden account for bw-secret.",
        }

    @classmethod
    def _probe(cls, key: str) -> tuple[str | None, str | None]:
        return "vault", None
