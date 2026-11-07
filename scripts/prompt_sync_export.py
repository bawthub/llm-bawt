"""Run llm_bawt.prompt_sync from a source checkout without app dependencies (TASK-2127).

Loads only ``prompt_sync`` and ``prompt_synced_defaults`` (stdlib-only) under a
bare ``llm_bawt`` package so the package ``__init__`` and its deps are skipped.
Usage: python3 -I scripts/prompt_sync_export.py <src-dir> check|export [args]
"""

import sys
import types
from pathlib import Path

src = Path(sys.argv[1]).resolve()
package = types.ModuleType("llm_bawt")
package.__path__ = [str(src / "llm_bawt")]
sys.modules["llm_bawt"] = package

from llm_bawt.prompt_sync import main  # noqa: E402

raise SystemExit(main(sys.argv[2:]))
