#!/usr/bin/env bash
# The Claude Code section on all issues-v1 questions (Saleor 106, Zulip 150, PostHog 65).
# Phase 1: arms A-C per repo (canary first). Phase 2, only if phase 1 finished: arm D per repo.
# Any canary failure or harness stop (rate-limit guard at 0.985, errors) ends the chain; rows
# already written stay. Log: $TUNA_OUT/runs/cc-issues.log
set -u
cd "$(dirname "$0")/.."
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
mkdir -p "$OUT/runs"
L=$OUT/runs/cc-issues.log
P=.venv/bin/python
step() { echo "== $* $(date -u +%FT%TZ)" >> "$L"; $P scripts/cc-bench.py "$@" >> "$L" 2>&1 || { echo "CHAIN STOP: $* exited $?" >> "$L"; exit 3; }; }
for r in saleor zulip posthog; do
  step canary --repo $r --name issues --arms A B C
  step run --repo $r --name issues --arms A B C --workers 4
done
echo "== phase 1 (A-C) done $(date -u +%FT%TZ)" >> "$L"
for r in saleor zulip posthog; do
  step canary --repo $r --name issues --arms D
  step run --repo $r --name issues --arms D --workers 4
done
echo "== done $(date -u +%FT%TZ)" >> "$L"
