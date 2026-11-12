#!/usr/bin/env bash
# Milestone two, one repo end to end:
#   1. held-out list; training questions (qwen-train1, qwen-bug1) and qwen-m4-400 unless written
#      (needs the question-writer server running; see scripts/pi-ask)
#   2. leak check; tuned models (m2-train-models.sh); scoring (m2-score.sh) (needs the GPU)
#
#   scripts/m2-repo.sh REPO
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: m2-repo.sh REPO}
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
declare -A PER_FILE=([saleor]=10 [zulip]=20 [posthog]=8)
declare -A SYMS_PER_Q=([saleor]=3 [zulip]=1.5 [posthog]=4)
# qwen-bug1: issue-style training questions (prompt question-bug1), about 500 per repo,
# spread over more files (own seed), same exclusions, filter and split rules
declare -A BUG_PER_FILE=([saleor]=3 [zulip]=5 [posthog]=3)
declare -A BUG_SYMS_PER_Q=([saleor]=3 [zulip]=1.5 [posthog]=4)
QWORKERS=${QWORKERS:-8}   # parallel writers; match your server's concurrency
WRITE_BUG=${WRITE_BUG:-1}  # 0: leave qwen-bug1 for a later writer run (bug reports write slowly)
WRITE_M4=${WRITE_M4:-1}    # 0: leave qwen-m4-400 for a later writer run
# qwen-m4-400: Qwen writes one question for each of claude-m3's golds (same files and targets) with the
# gold-targeted prompt, so the two writers compare at equal size.
log() { echo "=== $REPO $* $(date -u +%H:%M:%S)"; }

finished() { python3 -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['finished'] else 1)" \
              "$OUT/$REPO/questions/$1.manifest.json" 2>/dev/null; }
if ! finished qwen-train1 || { [ "$WRITE_BUG" = 1 ] && ! finished qwen-bug1; }; then
  log "heldout"
  $TUNA heldout --repo "$REPO"
fi
if ! finished qwen-train1; then
  log "questions qwen-train1"
  $TUNA questions --repo "$REPO" --name qwen-train1 --prompt question-v2 --n 1500 --workers "$QWORKERS" \
    --exclude-files "$OUT/$REPO/questions/heldout-files.txt" \
    --per-file "${PER_FILE[$REPO]}" --symbols-per-question "${SYMS_PER_Q[$REPO]}" \
    --val-fraction 0.1 --timeout 90 --max-tool-calls 40
fi
if [ "$WRITE_BUG" = 1 ] && ! finished qwen-bug1; then
  log "questions qwen-bug1"
  $TUNA questions --repo "$REPO" --name qwen-bug1 --prompt question-bug1 --n 500 --seed 20261113 \
    --workers "$QWORKERS" --exclude-files "$OUT/$REPO/questions/heldout-files.txt" \
    --per-file "${BUG_PER_FILE[$REPO]}" --symbols-per-question "${BUG_SYMS_PER_Q[$REPO]}" \
    --val-fraction 0.1 --timeout 240 --max-tool-calls 40   # bug reports reason longer: 240 s per attempt
fi
if [ "$WRITE_M4" = 1 ] && [ -f "$OUT/$REPO/questions/claude-m3.jsonl" ] && ! finished qwen-m4-400; then
  log "questions qwen-m4-400 (claude-m3's golds)"
  $TUNA questions --repo "$REPO" \
    --name qwen-m4-400 --prompt question-v2-gold --gold-list "$OUT/$REPO/questions/claude-m3.jsonl" --n 400 \
    --writer pi --workers "$QWORKERS" --timeout 90 --max-tool-calls 40
fi
if finished qwen-bug1; then
  .venv/bin/python scripts/merge-sets.py --repo "$REPO" --sets qwen-train1 qwen-bug1 --name qwen-trainbug1 >/dev/null
  $TUNA heldout --repo "$REPO" --check qwen-bug1
  $TUNA heldout --repo "$REPO" --check qwen-trainbug1
fi
log "leak check"
$TUNA heldout --repo "$REPO" --check qwen-train1
scripts/m2-train-models.sh "$REPO"
scripts/m2-score.sh "$REPO"
log "done"
