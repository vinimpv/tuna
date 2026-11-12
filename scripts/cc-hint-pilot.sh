#!/usr/bin/env bash
# Hint-card pilot: 20 issues per repo, arms A, H-stock, H-tuned, per issue back to back in seeded order.
# Canary per repo first; any canary failure or harness stop (rate-limit guard 0.985, failures) ends the chain.
set -u
cd "$(dirname "$0")/.."
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
mkdir -p "$OUT/runs"
L=$OUT/runs/cc-hint-pilot.log
P=.venv/bin/python
step() { echo "== $* $(date -u +%FT%TZ)" >> "$L"; $P scripts/cc-bench.py "$@" >> "$L" 2>&1 || { echo "CHAIN STOP: $* exited $?" >> "$L"; exit 3; }; }
for r in saleor zulip posthog; do
  step canary --repo $r --name hint-pilot --arms A H-stock H-tuned
  step run --repo $r --name hint-pilot --arms A H-stock H-tuned --workers 4
done
echo "== done $(date -u +%FT%TZ)" >> "$L"
