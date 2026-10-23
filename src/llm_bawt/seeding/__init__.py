"""Virgin-tenant seeding (TASK-350 / TASK-355).

Public surface:

- :func:`llm_bawt.seeding.engine.seed_tenant` — idempotent infrastructure
  seed (catalog schema, standard access paths, prompt defaults). Called by
  the installer AND the setup wizard.
- :meth:`llm_bawt.seeding.engine.TenantSeeder.register_model` — registers a
  model + endpoint (insert-if-missing). The setup wizard does not use it: it
  adds models through the admin catalog routes (TASK-1297).
- ``python -m llm_bawt.seeding`` — thin CLI wrapper around ``seed_tenant``.

No bots or model rows are seed data: the wizard creates the first bot only
after a provider credential exists (see :mod:`llm_bawt.seeding.data`).
"""

from .data import FIRST_BOT_AGENT_BACKEND
from .engine import TenantSeeder, seed_tenant

__all__ = [
    "FIRST_BOT_AGENT_BACKEND",
    "TenantSeeder",
    "seed_tenant",
]
