#!/usr/bin/env bash
# Package install only: never changes settings, installs Node, or restarts services.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$here/install.py" "$@"
