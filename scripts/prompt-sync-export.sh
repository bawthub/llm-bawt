#!/usr/bin/env bash
# Export live product prompt drift into a task worktree as synced defaults (TASK-2127).
#
#   scripts/prompt-sync-export.sh <llm-bawt-worktree> [api-base]
#
# Writes src/llm_bawt/prompt_defaults/synced/ in the given worktree from the
# live install's /v1/prompts/sync/drift, using that worktree's own code. Exit 0
# with "in sync" when there is nothing to export; exit 3 when files changed and
# need the normal commit -> review -> gate -> publish path; exit 4 when nothing
# changed but prompts were held for naming install terms (bot names from the
# API, plus PROMPT_SYNC_INSTALL_TERMS). Never commits.
set -euo pipefail

wt="${1:?usage: prompt-sync-export.sh <llm-bawt-worktree> [api-base]}"
api="${2:-http://app:8642}"
root="$wt/src/llm_bawt/prompt_defaults/synced"

[ -d "$wt/.git" ] || [ -f "$wt/.git" ] || { echo "not a git checkout: $wt" >&2; exit 2; }

runner=(python3 -I "$wt/scripts/prompt_sync_export.py" "$wt/src")
if "${runner[@]}" check --api "$api"; then
  echo "in sync"
  exit 0
fi

held=0
"${runner[@]}" export --api "$api" --root "$root" || { rc=$?; [ "$rc" -eq 4 ] || exit "$rc"; held=1; }
[ "$held" -eq 0 ] || echo "held: some prompts name install-specific terms; reword the live row or mark it install" >&2

if [ -z "$(git -C "$wt" status --porcelain -- "$root")" ]; then
  echo "drift reported but no file changed (already exported, awaiting publish)"
  [ "$held" -eq 0 ] || exit 4
  exit 0
fi
git -C "$wt" status --short -- "$root"
exit 3
