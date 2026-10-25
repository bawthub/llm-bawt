"""BawtHub service-caller auth contract (TASK-1472, for BawtHub auth v1 TASK-1467).

One token, the BawtHub service token, authenticates llm-bawt to BawtHub and
BawtHub to llm-bawt's internal secret API. llm-bawt derives it from its master
key and exports it to a file BawtHub reads; an operator may override it by env
or secret file. It never appears in source.
"""

from .service_token import (
    DERIVATION_LABEL,
    EXPORT_FILE_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    BawtHubServiceAuth,
    ExportResult,
    ServiceTokenSource,
    TokenCheck,
    derive_service_token,
    service_auth,
    service_token_source,
)
from .token_export import EXPORT_UID_ENV, export_service_token

__all__ = [
    "DERIVATION_LABEL",
    "EXPORT_FILE_ENV",
    "EXPORT_UID_ENV",
    "TOKEN_ENV",
    "TOKEN_FILE_ENV",
    "BawtHubServiceAuth",
    "ExportResult",
    "ServiceTokenSource",
    "TokenCheck",
    "derive_service_token",
    "export_service_token",
    "service_auth",
    "service_token_source",
]
