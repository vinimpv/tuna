#!/usr/bin/env bash
# LFM3-1.7B-written test set (lfm3-t400), 400 per repo. Writer: scripts/pi-ask against an OpenAI-compatible
# vLLM server serving LiquidAI/LFM3-1.7B (set PI_CODING_AGENT_DIR to a pi config that points at it). Same target sampling as claude-m3's new part: held-out
# files only, golds of earlier test sets excluded, 3 per file, 1 per 3 symbols, prompt question-v2 with the
# v2 filter, seed 20261111, 90 s and 40 tool calls per attempt. Test-only: held-out files never train.
set -euo pipefail
cd "$(dirname "$0")/.."
export PI_CODING_AGENT_DIR=${PI_CODING_AGENT_DIR:?set to a pi config for the LFM3 server}
export TUNA_WRITER_ENGINE=${TUNA_WRITER_ENGINE:-"vLLM 0.33.0rc1.dev908+gc96e5f60b, LFM3-1.7B@0dc4268b"}
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
mkdir -p "$OUT/runs"
pids=()
for r in saleor zulip posthog; do
  Q=$OUT/$r/questions
  .venv/bin/tuna questions --repo "$r" --name lfm3-t400 --writer pi --prompt question-v2 --n 400 --seed 20261111 \
    --workers 8 --timeout 90 --max-tool-calls 40 --only-files "$Q/heldout-files.txt" \
    --used-golds-from claude-m3 qwen-m2 qwen-m1 issues-v1 issues-v1-noleak --per-file 3 --symbols-per-question 3 \
    > "$OUT/runs/lfm3-t400-$r.log" 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for r in saleor zulip posthog; do .venv/bin/tuna review --repo "$r" --name lfm3-t400 --n 25 --seed 20261112 || true; done
.venv/bin/python scripts/reject-shares.py --write | grep -E 'lfm3|claude-m3-new' || true
exit $rc
