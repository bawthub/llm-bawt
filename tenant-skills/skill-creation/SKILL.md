---
name: skill-creation
description: Create, edit and publish skills in agent-skills. Covers concise instructions, one authoritative policy source, isolated editing, validation and skills-sync.
---

# Creating and updating skills

Skills live in `~/dev/agent-skills` and are linked into agent environments. Use an isolated worktree and the [git workflow](../git-commits/SKILL.md); do not author in the live checkout.

Keep each skill brief: when it applies, the few constraints that matter, and links to technical references. Trust the model to choose routine implementation details. Prefer replacing outdated guidance to appending another exception. Do not turn an incident into a universal prohibition.

Use valid YAML frontmatter with a matching kebab-case `name` and a specific description under 450 characters. Keep operational details out of unrelated skills. Each policy has one authoritative home; other skills link to it.

Check frontmatter with `python3 scripts/test_skill_frontmatter.py` and run relevant script tests when helpers change. Instruction-only changes need consistency review, not an application full-suite run. Publish with `skills-sync <SHA>` and verify both remote and live checkout.

Factual corrections and consolidation are routine maintenance. Changes that add mandatory gates or broaden decision authority require the user's approval. Preserve private-data and secret boundaries in public exports. Skills in `product-skills.json` ship to tenants: keep installation facts (hosts, addresses, host users) generic or inside `<!-- deployment -->` blocks, and run `python3 scripts/test_export_product_skills.py` after editing one.
