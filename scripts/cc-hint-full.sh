#!/usr/bin/env bash
# Full hint-card run: all 321 issues-v1 questions, five arms (A, H-stock, H-tuned, H-bm25, H-fused),
# each issue's arms back to back in a seeded random order. Canary per repo on all five arms first.
# A canary failure or a harness stop (rate-limit guard 0.985, failures) ends the chain; finished rows stay.
set -u
cd "$(dirname "$0")/.."
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
mkdir -p "$OUT/runs"
L=$OUT/runs/cc-hint-full.log
P=.venv/bin/python
ARMS="A H-stock H-tuned H-bm25 H-fused"
step() { echo "== $* $(date -u +%FT%TZ)" >> "$L"; $P scripts/cc-bench.py "$@" >> "$L" 2>&1 || { echo "CHAIN STOP: $* exited $?" >> "$L"; exit 3; }; }
for r in saleor zulip posthog; do
  step canary --repo $r --name hint-full --arms $ARMS
  step run --repo $r --name hint-full --arms $ARMS --workers 4
done
echo "== done $(date -u +%FT%TZ)" >> "$L"
