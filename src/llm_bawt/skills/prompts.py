"""Prompt-registry text for the injected-skills section (TASK-1309).

The ``DEFAULT_*`` constants are seed data only; the section renderer resolves
both keys per bot at call time. Placeholders are filled in one pass by plain
substitution, never ``.format()``: skill bodies and edited templates hold
literal braces, and a body must never be re-scanned for placeholders.

No side effects, so the prompt registry can import this module.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

PREAMBLE_KEY = "agents.injected_skills.preamble"
ENTRY_KEY = "agents.injected_skills.entry"
ENTRY_VARS = ("ref", "base_dir", "body")

DEFAULT_PREAMBLE = (
    "## Injected Skills\n"
    "The skills below are already loaded in full. Follow them directly; don't call the "
    "Skill tool for them. A relative path inside a skill resolves against that skill's "
    "base directory."
)

DEFAULT_ENTRY = (
    "### Skill: {ref}\n"
    "Base directory: {base_dir}\n\n"
    "{body}"
)

_PLACEHOLDER = re.compile(r"\{(" + "|".join(ENTRY_VARS) + r")\}")


def render_entry(template: str, *, ref: str, base_dir: str, body: str) -> str:
    values = {"ref": ref, "base_dir": base_dir, "body": body.strip()}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def render_section(preamble: str, entry_template: str,
                   skills: Sequence[tuple[str, str, str]]) -> str:
    """``skills`` is (ref, base_dir, body) in prompt order; empty renders nothing.

    An entry template that drops ``{body}`` would inject headings without the
    skills, so it renders nothing instead.
    """
    if not skills or "{body}" not in entry_template:
        return ""
    parts = [preamble.strip()] if preamble.strip() else []
    parts += [render_entry(entry_template, ref=r, base_dir=d, body=b) for r, d, b in skills]
    return "\n\n".join(parts)


__all__ = ["DEFAULT_ENTRY", "DEFAULT_PREAMBLE", "ENTRY_KEY", "ENTRY_VARS", "PREAMBLE_KEY",
           "render_entry", "render_section"]
