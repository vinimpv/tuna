#!/usr/bin/env bash
# CPU post-processing: intervals and tables from saved ranks and evals (no GPU).
#   scripts/m2-intervals.sh final    all sets, pooled and per repo; transfer split; issue metrics; fusion;
#                                    word overlap; dataset table; writer coverage; cost figures; the extra
#                                    intervals (extras-saleor, extras-pooled, baseline-pairs), seed variance
#                                    and writer gap
#   The rank files (ranks/) are not in this release: rebuild them first with scripts/m2-score.sh (and the
#   transfer, pooled and seed scripts for their rows).
#   scripts/m2-intervals.sh d2q      the doc2query rows on Saleor against their code-only rows
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
PY=.venv/bin/python
B=$OUT/bootstrap; T=$OUT/tables; mkdir -p "$B" "$T"
log() { echo "=== intervals $* $(date -u +%H:%M:%S)"; }
case ${1:?usage: m2-intervals.sh final|d2q} in
final)
  log "rrf"; $PY scripts/rrf.py --json "$T/rrf.json" > "$T/rrf.txt"
  log "bootstraps (one process per set, in parallel; single-threaded each)"
  pids=()
  for s in qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title issues-v1-noleak; do
    $PY scripts/paired-bootstrap.py --sets "$s" --pool-repos --json "$B/final-mrr10-$s.json" > "$B/final-mrr10-$s.txt" & pids+=($!)
  done
  $PY scripts/paired-bootstrap.py --sets issues-v1 issues-v1-title issues-v1-noleak --metric file_hit10 --pool-repos \
    --json "$B/final-fhit10.json" > "$B/final-fhit10.txt" & pids+=($!)
  $PY scripts/transfer-split.py --json "$T/transfer-split.json" > "$T/transfer-split.txt" & pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
  log "tables"
  $PY scripts/issue-metrics.py --json "$T/issue-metrics.json" > "$T/issue-metrics.txt"
  $PY scripts/word-overlap.py --sets qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title \
    qwen-train1.train --json "$T/word-overlap.json" > "$T/word-overlap.txt"
  $PY scripts/dataset-table.py --json "$T/datasets.json" > "$T/datasets.txt"
  for s in claude-m3 issues-v1; do $PY scripts/cost-figure.py --set "$s" > "$T/cost-$s.txt" || true; done
  log "extra intervals (the summary's seed, curve, 4B, re-mining and pooled rows; the baseline pairs)"
  $PY scripts/paired-bootstrap.py --repos saleor --sets qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title issues-v1-noleak \
      --pairs lfm25-ft-lora-n150:lfm25-ft-lora lfm25-ft-lora-n150:lfm25-emb lfm25-ft-lora-n400:lfm25-ft-lora \
      lfm25-ft-lora-n400:lfm25-emb lfm25-ft-lora-n800:lfm25-ft-lora lfm25-ft-lora-n800:lfm25-emb \
      lfm25-ft-lora-s1:lfm25-emb jina-0.5b-ft-lora-s1:jina-0.5b qwen3-0.6b-ft-lora-s1:qwen3-0.6b \
      lfm25-ft-lora-s1:qwen3-0.6b lfm25-ft-lora-s1:jina-1.5b lfm25-ft-lora-s2:lfm25-emb \
      jina-0.5b-ft-lora-s2:jina-0.5b qwen3-0.6b-ft-lora-s2:qwen3-0.6b lfm25-ft-lora-s2:qwen3-0.6b \
      lfm25-ft-lora-s2:jina-1.5b qwen3-4b:qwen3-0.6b qwen3-4b:lfm25-ft-lora qwen3-4b:jina-1.5b \
      lfm25-ft-lora-remine:lfm25-ft-lora lfm25-ft-lora-bug:lfm25-ft-lora jina-0.5b-ft-lora-4090:jina-0.5b \
      qwen3-0.6b-ft-lora-4090:qwen3-0.6b --cluster-by-file --metric mrr10 \
    --json "$B/extras-saleor.json" > "$B/extras-saleor.txt"
  $PY scripts/paired-bootstrap.py --repos saleor zulip posthog --sets qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title \
      issues-v1-noleak --pairs lfm25-pooled-lora.pooled:lfm25-ft-lora lfm25-pooled-lora.pooled:lfm25-emb \
      lfm25-pooled-full.pooled:lfm25-ft-full jina-0.5b-pooled-lora.pooled:jina-0.5b-ft-lora \
      jina-0.5b-pooled-lora.pooled:jina-0.5b qwen3-0.6b-pooled-lora.pooled:qwen3-0.6b-ft-lora \
      qwen3-0.6b-pooled-lora.pooled:qwen3-0.6b --pool-repos --metric mrr10 \
    --json "$B/extras-pooled.json" > "$B/extras-pooled.txt"
  $PY scripts/paired-bootstrap.py --repos saleor zulip posthog --sets claude-m3 issues-v1 --pairs lfm25-ft-lora:lfm25-emb \
      jina-0.5b-ft-lora:jina-0.5b qwen3-0.6b-ft-lora:qwen3-0.6b lfm25-ft-lora:lfm25-ft-full \
      jina-0.5b-ft-lora:jina-0.5b-ft-full qwen3-0.6b-ft-lora:qwen3-0.6b-ft-full lfm25-emb:bm25 \
      lfm25-ft-lora:bm25 --pool-repos --metric mrr10 \
    --json "$B/baseline-pairs.json" > "$B/baseline-pairs.txt"
  $PY scripts/seed-variance.py --json "$T/seed-variance.json" > /dev/null
  $PY scripts/writer-gap.py --json "$T/writer-gap.json" > /dev/null ;;
d2q)
  pairs=()
  for row in lfm25-emb jina-0.5b qwen3-0.6b; do
    for k in qonly-qp qonly-dp both-qp both-dp; do pairs+=("$row.d2q-$k:$row"); done
  done
  for row in lfm25-ft-lora jina-0.5b-ft-lora qwen3-0.6b-ft-lora lfm25-ft-full jina-0.5b-ft-full qwen3-0.6b-ft-full; do
    for k in both-qp both-dp; do pairs+=("$row.d2q-$k:$row"); done
  done
  pairs+=("lfm25-emb.d2q-both-qp:lfm25-ft-lora" "jina-0.5b.d2q-both-qp:jina-0.5b-ft-lora" "qwen3-0.6b.d2q-both-qp:qwen3-0.6b-ft-lora")
  $PY scripts/paired-bootstrap.py --repos saleor --cluster-by-file --pairs "${pairs[@]}" \
    --sets qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title issues-v1-noleak \
    --json "$B/d2q-mrr10.json" > "$B/d2q-mrr10.txt" ;;
esac
log "done"
