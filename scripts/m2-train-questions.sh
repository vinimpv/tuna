#!/usr/bin/env bash
# Milestone two: training questions per repo, never on a test-set file.
#
#   scripts/m2-train-questions.sh [REPO...]        # default: saleor zulip posthog
#
# Same writer, prompt and filter as the v2 test set (pi on the served writer model,
# question-v2, v2 filter), seeded, 8 parallel writers. Several questions per
# file, one per N eligible symbols up to a cap per file (per repo below, sized
# to the symbols left once test-set files are held out). 10% of questions,
# grouped by file, become the validation split. Each finished set is checked
# against the held-out list.
set -euo pipefail
cd "$(dirname "$0")/.."
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
REPOS=("$@"); [ ${#REPOS[@]} -gt 0 ] || REPOS=(saleor zulip posthog)

declare -A PER_FILE=([saleor]=10 [zulip]=20 [posthog]=8)
declare -A SYMS_PER_Q=([saleor]=3 [zulip]=1.5 [posthog]=4)

for repo in "${REPOS[@]}"; do
  echo "=== $repo heldout $(date -u +%H:%M:%S)"
  $TUNA heldout --repo "$repo"
  echo "=== $repo questions $(date -u +%H:%M:%S)"
  $TUNA questions --repo "$repo" --name qwen-train1 --prompt question-v2 --n 1500 --workers 8 \
    --exclude-files "$OUT/$repo/questions/heldout-files.txt" \
    --per-file "${PER_FILE[$repo]}" --symbols-per-question "${SYMS_PER_Q[$repo]}" \
    --val-fraction 0.1 --timeout 90 --max-tool-calls 40
  echo "=== $repo check $(date -u +%H:%M:%S)"
  $TUNA heldout --repo "$repo" --check qwen-train1
done
echo "=== done $(date -u +%H:%M:%S)"
