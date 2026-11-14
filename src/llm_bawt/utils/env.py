from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def read_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE .env file."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env_file(path: Path, override: bool = False) -> Dict[str, str]:
    """Load .env values into os.environ for non-prefixed keys."""
    values = read_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def set_env_value(path: Path, key: str, value: str) -> bool:
    """Set or update a raw KEY=VALUE entry in a .env file."""
    lines = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    new_line = f"{key}={value}\n"
    updated = False
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
            updated_lines.append(new_line)
            updated = True
        else:
            updated_lines.append(line)

    if not updated:
        if updated_lines and not updated_lines[-1].endswith(("\n", "\r")):
            updated_lines[-1] += "\n"
        updated_lines.append(new_line)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(updated_lines), encoding="utf-8")
        return True
    except OSError:
        return False
