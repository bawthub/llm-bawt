---
name: skill-creation
description: Create, edit and publish skills in agent-skills. Covers concise instructions, one authoritative policy source, isolated editing, validation and skills-sync.
---

# Creating and updating skills

Author skills in the configured skills repository, not the generated plugin. Use an isolated worktree and the [git workflow](../git-commits/SKILL.md); do not author in the live checkout.

Keep each skill brief: when it applies, the few constraints that matter, and links to technical references. Trust the model to choose routine implementation details. Prefer replacing outdated guidance to appending another exception. Do not turn an incident into a universal prohibition.

Use valid YAML frontmatter with a matching kebab-case `name` and a specific description under 450 characters. Keep operational details out of unrelated skills. Each policy has one authoritative home; other skills link to it.

Check frontmatter with `python3 scripts/test_skill_frontmatter.py` and run relevant script tests when helpers change. Instruction-only changes need consistency review, not an application full-suite run. Publish with `skills-sync <SHA>`, which refuses to push a SHA whose frontmatter fails that check, and verify both remote and live checkout.

Factual corrections and consolidation are routine maintenance. Changes that add mandatory gates or broaden decision authority require the operator's approval. The `product-skills.json` allowlist defines portable source. Put installation facts and personal workflows in the private overlay, never in product source. Run the product export, secret scan, and private denylist checks before publishing; retain their dated receipt.
