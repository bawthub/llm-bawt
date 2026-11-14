# Prompt defaults and tenant control

The owning implementation/runbook is llm-bawt `docs/PRODUCT_DEFAULTS.md`.
Use it for the domain inventory, migration guarantees and exact commands.

- Read and edit the live registry once for an immediate tenant change. Ordinary
  saves, including no-op saves, are overrides; body equality never cancels that
  intent. DB edit history and product content versions are different.
- Proven default-managed rows follow shipped upgrades. Ambiguous/edited legacy
  rows and scoped customizations remain overrides. Reset inherits the latest
  fallback now and future releases; empty is not a reset/disable signal.
- Product promotion is explicit and portable: selected global keys, approved
  source API, one export command, then the existing source review/publication
  workflow. Never export credentials, private/persona/bot/user/environment data
  or maintain the same body in a second seed. A product/install flag alone does
  not publish a prompt or change tenant override intent.
- Export is read-only to the DB. Reset the source tenant only after the approved
  default ships if it should follow again. Neither a task card nor a fixture
  export independently authorizes live writes or service changes.
- Offline fixture checks validate source, not live state. Read-only live drift
  separates stale managed defaults, intentional overrides and promotion
  candidates. Verify both upgrade propagation and retained tenant control.

Prompt/session pinning remains a separate activation boundary; use
[agent system prompt](agent-system-prompt.md) when diagnosing adoption.
