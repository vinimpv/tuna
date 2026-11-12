#!/usr/bin/env bash
# Transfer-only test set for a test-only repo (Dispatch, Home Assistant): 400 Claude-written questions on
# random files, with claude-m3's writer, prompt, filter and settings (Claude Code via scripts/claude-ask,
# claude-opus-5-5, question-v2 + v2 filter, seed 20261111, 3 per file, 1 per 3 symbols, 90 s, 40 tool
# calls, 3 attempts), then a 25-question spot-check file. Test-only: the repo's config has
# test_only = true and tuna train refuses it. Costs go to the set's manifest.
#
# When a writer call fails on a usage or rate limit, tuna questions stops; this script backs off
# (10, 20, 30 ... up to 60 min) and resumes, until done.
#
#   scripts/claude-test-set.sh REPO
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: claude-test-set.sh REPO}
NAME=claude-t400
export CLAUDE_ASK_MODEL=${CLAUDE_ASK_MODEL:-claude-opus-5-5}
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
M=$OUT/$REPO/questions/$NAME.manifest.json
finished() { python3 -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['finished'] else 1)" "$M" 2>/dev/null; }
wait_s=600
for try in $(seq 1 40); do
  finished && break
  echo "=== $REPO $NAME try $try $(date -u +%H:%M:%S)"
  .venv/bin/tuna questions --repo "$REPO" --name "$NAME" --writer claude-code --prompt question-v2 --n 400 \
    --seed 20261111 --workers 4 --timeout 90 --max-tool-calls 40 --per-file 3 --symbols-per-question 3
  finished && break
  echo "=== $REPO $NAME stopped before 400 (writer error or limit); backing off ${wait_s}s"
  sleep "$wait_s"
  wait_s=$(( wait_s < 3600 ? wait_s + 600 : 3600 ))
done
finished || { echo "=== $REPO $NAME not finished after 40 tries" >&2; exit 1; }
.venv/bin/tuna review --repo "$REPO" --name "$NAME" --n 25 --seed 20261112
python3 -I -c "import json,sys; m=json.load(open(sys.argv[1])); print('cost_usd', m.get('cost_usd'), 'accepted', m.get('accepted'))" "$M"
echo "=== $REPO $NAME done $(date -u +%H:%M:%S)"
