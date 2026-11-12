#!/bin/bash
# The Claude-written held-out test set claude-m3 (Claude Code through scripts/claude-ask, no GPU).
#
#   scripts/claude-test-set-m3.sh m3 REPO          # claude-m3-new: 300 test questions on held-out files,
#                                                   # merged with claude-m2 into claude-m3 (400)
#
# Writer: Claude Code (scripts/claude-ask), model pinned to claude-opus-5-5 as in claude-m2. Reruns resume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src CLAUDE_ASK_MODEL=${CLAUDE_ASK_MODEL:-claude-opus-5-5}
PY=${PY:-.venv/bin/python}
what=$1 repo=$2
Q=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}/$repo/questions
common=(--repo "$repo" --writer claude-code --seed 20261111 --workers 4 --timeout 90 --max-tool-calls 40)
case $what in
  m3)
    $PY -m tuna questions "${common[@]}" --name claude-m3-new --prompt question-v2 --n 300 \
      --only-files "$Q/heldout-files.txt" --used-golds-from claude-m2 qwen-m2 qwen-m1 issues-v1 issues-v1-noleak \
      --per-file 3 --symbols-per-question 3 --max-cost 40
    $PY scripts/merge-question-sets.py --repo "$repo" --name claude-m3 claude-m2 claude-m3-new ;;
  *) echo "usage: $0 m3 REPO" >&2; exit 2 ;;
esac
