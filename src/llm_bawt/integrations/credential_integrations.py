"""Integrations configured by entering credentials (TASK-1476).

Some integrations have no login flow: the operator types a URL and an API key.
Each one is a :class:`CredentialIntegration` whose fields say which values are
plain settings (stored as cleartext ``meta``) and which are secrets (stored
encrypted). The record lives in the same ``integration_connection:<id>`` row
that ``GET /v1/internal/connections/integration/<id>`` reads, so the
definition here is also where that route's field contract comes from.

Secrets are write-only: a descriptor reports whether each one is set, never
its value. A save that leaves a secret blank keeps the stored one, so changing
a URL doesn't mean retyping the key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore, RowSnapshot
from llm_bawt.service.providers.base import STATUS_CONNECTED, ConnectionRecord

AUTH_METHOD = "credentials"
MAX_VALUE_LENGTH = 4096
# Compare-and-set retries before a save or removal reports ``conflict``.
WRITE_ATTEMPTS = 3

FieldKind = Literal["url", "text", "secret"]


@dataclass(frozen=True)
class CredentialField:
    name: str
    label: str
    kind: FieldKind
    required: bool = True
    placeholder: str = ""

    @property
    def secret(self) -> bool:
        return self.kind == "secret"


@dataclass(frozen=True)
class CredentialIntegration:
    id: str
    label: str
    description: str
    fields: tuple[CredentialField, ...]
    # At least one of these (non-required) fields must be set.
    requires_one_of: tuple[str, ...] = ()
    # Whether BawtHub's server may read it through the internal connection
    # route. Integrations only llm-bawt itself uses stay off that allowlist.
    bawthub_readable: bool = True

    def field(self, name: str) -> CredentialField | None:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def secret_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.secret)

    @property
    def required_meta(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if not f.secret and f.required)

    @property
    def optional_meta(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if not f.secret and not f.required)


CREDENTIAL_INTEGRATIONS: dict[str, CredentialIntegration] = {
    definition.id: definition
    for definition in (
        CredentialIntegration(
            id="unraid",
            label="Unraid",
            description="Container status from your Unraid server's API.",
            fields=(
                CredentialField("url", "Server URL", "url", placeholder="https://unraid.example"),
                CredentialField("api_key", "API key", "secret"),
            ),
        ),
        CredentialIntegration(
            id="object-store",
            label="Object storage",
            description="S3-compatible bucket for uploads and media.",
            fields=(
                CredentialField("endpoint", "Endpoint URL", "url", placeholder="https://s3.example"),
                CredentialField("region", "Region", "text", placeholder="us-east-1"),
                CredentialField("bucket", "Bucket", "text"),
                CredentialField("access_key_id", "Access key ID", "secret"),
                CredentialField("secret_access_key", "Secret access key", "secret"),
            ),
        ),
        CredentialIntegration(
            id="azure-speech",
            label="Azure Speech",
            description="Text-to-speech voices from Azure AI Speech.",
            fields=(
                CredentialField("region", "Region", "text", required=False, placeholder="eastus"),
                CredentialField("endpoint", "Endpoint URL", "url", required=False),
                CredentialField("key", "Key", "secret"),
            ),
            requires_one_of=("region", "endpoint"),
        ),
        CredentialIntegration(
            id="home-assistant",
            label="Home Assistant",
            description="Weather and home state from Home Assistant.",
            fields=(
                CredentialField("url", "Home Assistant URL", "url", placeholder="https://homeassistant.example"),
                CredentialField("token", "Long-lived access token", "secret"),
            ),
        ),
        *(
            CredentialIntegration(
                id=service_id,
                label=label,
                description=description,
                fields=(
                    CredentialField("url", "Server URL", "url", placeholder=placeholder),
                    CredentialField("api_key", "API key", "secret"),
                ),
                bawthub_readable=False,
            )
            for service_id, label, description, placeholder in (
                ("sonarr", "Sonarr", "TV library and downloads for the media tools.", "https://sonarr.example"),
                ("radarr", "Radarr", "Movie library and downloads for the media tools.", "https://radarr.example"),
                ("sabnzbd", "SABnzbd", "Download queue for the media tools.", "https://sabnzbd.example"),
            )
        ),
    )
}


def _clean(value: object) -> str:
    if not isinstance(value, str):
        raise IntegrationError("invalid_field")
    value = value.strip()
    if len(value) > MAX_VALUE_LENGTH:
        raise IntegrationError("field_too_long")
    return value


def _valid_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname) and not parts.username and not parts.password


class CredentialIntegrations:
    """Save, describe and remove credential-entry integrations."""

    def __init__(self, config, *, store: IntegrationStore | None = None):
        self.store = store if store is not None else IntegrationStore(config)

    @staticmethod
    def definition(integration_id: str) -> CredentialIntegration:
        definition = CREDENTIAL_INTEGRATIONS.get(integration_id)
        if definition is None:
            raise IntegrationError("unknown_integration")
        return definition

    @staticmethod
    def _foreign(snapshot: RowSnapshot) -> bool:
        """A row this form didn't write (another auth method, or unreadable)."""
        return snapshot.exists and (snapshot.record is None or snapshot.record.auth_method != AUTH_METHOD)

    def _owned(self, integration_id: str) -> RowSnapshot:
        snapshot = self.store.snapshot(integration_id)
        if self._foreign(snapshot):
            # Never overwrite or delete a connection something else manages.
            raise IntegrationError("managed_elsewhere")
        return snapshot

    def descriptor(self, integration_id: str) -> dict:
        definition = self.definition(integration_id)
        snapshot = self.store.snapshot(integration_id)
        foreign = self._foreign(snapshot)
        record = None if foreign else snapshot.record
        meta = record.meta if record else {}
        secret = record.secret if record else {}
        return {
            "id": definition.id,
            "label": definition.label,
            "description": definition.description,
            "auth_method": AUTH_METHOD,
            "requirements": "",
            "limitations": "",
            "setup_url": "",
            "scopes": [],
            "admin_scopes": [],
            "configured": record is not None,
            "managed_elsewhere": foreign,
            "has_unverified_credentials": False,
            "reauthorization_required": False,
            "client": {"client_id": "", "redirect_uri": "", "has_client_secret": False},
            "fields": [
                {
                    "name": f.name,
                    "label": f.label,
                    "kind": f.kind,
                    "required": f.required,
                    "placeholder": f.placeholder,
                    # Secrets report only whether they're stored.
                    **({"set": bool(secret.get(f.name))} if f.secret else {"value": meta.get(f.name, "")}),
                }
                for f in definition.fields
            ],
            "requires_one_of": list(definition.requires_one_of),
            "connection": (record or ConnectionRecord(provider=integration_id)).public(),
        }

    @staticmethod
    def _validated(definition: CredentialIntegration, values: dict[str, object]) -> dict[str, str]:
        if any(definition.field(name) is None for name in values):
            raise IntegrationError("invalid_field")
        cleaned = {name: _clean(raw) for name, raw in values.items()}
        for name, value in cleaned.items():
            if value and definition.field(name).kind == "url" and not _valid_url(value):
                raise IntegrationError("invalid_url")
        return cleaned

    @staticmethod
    def _merged(definition: CredentialIntegration, integration_id: str,
                previous: ConnectionRecord | None, values: dict[str, str]) -> ConnectionRecord:
        meta = dict(previous.meta) if previous else {}
        secret = dict(previous.secret) if previous else {}
        for name, value in values.items():
            if definition.field(name).secret:
                if value:  # blank keeps the stored secret
                    secret[name] = value
            elif value:
                meta[name] = value
            else:
                meta.pop(name, None)
        known_meta = {f.name for f in definition.fields if not f.secret}
        meta = {k: v for k, v in meta.items() if k in known_meta}
        secret = {k: v for k, v in secret.items() if k in definition.secret_names}
        missing = [f.name for f in definition.fields if f.required
                   and not (secret if f.secret else meta).get(f.name)]
        if missing:
            raise IntegrationError("missing_required_field")
        if definition.requires_one_of and not any(meta.get(n) for n in definition.requires_one_of):
            raise IntegrationError("missing_required_field")
        return ConnectionRecord(
            provider=integration_id, status=STATUS_CONNECTED, auth_method=AUTH_METHOD,
            meta=meta, secret=secret,
            connected_at=previous.connected_at if previous else None,
        )

    def configure(self, integration_id: str, values: dict[str, object]) -> dict:
        definition = self.definition(integration_id)
        if not self.store.available:
            raise IntegrationError("storage_unavailable")
        cleaned = self._validated(definition, values)
        # Read, merge and compare-and-set the exact row. A concurrent save makes
        # the write miss; re-merging onto the newer row keeps its secrets, so a
        # blank-secret save can never restore a key that was just replaced.
        for _ in range(WRITE_ATTEMPTS):
            snapshot = self._owned(integration_id)
            record = self._merged(definition, integration_id, snapshot.record, cleaned)
            if self.store.replace(integration_id, snapshot, record):
                return self.descriptor(integration_id)
        raise IntegrationError("conflict")

    def disconnect(self, integration_id: str) -> bool:
        self.definition(integration_id)
        if not self.store.available:
            raise IntegrationError("storage_unavailable")
        for _ in range(WRITE_ATTEMPTS):
            snapshot = self._owned(integration_id)
            if not snapshot.exists:
                return False
            if self.store.remove(integration_id, snapshot):
                return True
        raise IntegrationError("conflict")


__all__ = [
    "AUTH_METHOD",
    "CREDENTIAL_INTEGRATIONS",
    "CredentialField",
    "CredentialIntegration",
    "CredentialIntegrations",
]
