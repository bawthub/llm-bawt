"""BawtHub service-caller auth contract (TASK-1472, for BawtHub auth v1 TASK-1467).

One deployment-bootstrap secret, the BawtHub service token, authenticates
llm-bawt to BawtHub and BawtHub to llm-bawt's internal secret API. Both sides
read it from the same env var or secret file; it never appears in source.
"""

from .service_token import (
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    BawtHubServiceAuth,
    ServiceTokenSource,
    TokenCheck,
    service_auth,
    service_token_source,
)

__all__ = [
    "TOKEN_ENV",
    "TOKEN_FILE_ENV",
    "BawtHubServiceAuth",
    "ServiceTokenSource",
    "TokenCheck",
    "service_auth",
    "service_token_source",
]
