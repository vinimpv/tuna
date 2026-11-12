#!/usr/bin/env bash
# Milestone two, one repo: score every stock and tuned row on every test set, and
# benchmark search for the stock/tuned pairs. Needs a GPU.
#
#   scripts/m2-score.sh REPO
#
# Rows: BM25, LFM2.5-ColBERT, the stock dense models (LFM2.5-Embedding, jina 0.5B
# and 1.5B, Qwen3-Embedding-0.6B) and the tuned ones (LFM2.5 full and LoRA, jina
# 0.5B full, Qwen3-0.6B full; plus any comparison rows that exist). Only queries
# are encoded: corpus vectors come from tuna embed. Every eval keeps per-question
# gold ranks (for paired bootstrap intervals); every step writes a ledger line.
#
# TUNA_EVAL_ONLY=1: the rank files were made on another machine; only tuna eval runs here.
# TUNA_NO_BENCH=1: skip the search benchmark (speed is measured on the local machine only).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: m2-score.sh REPO}
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
SETS=(qwen-m2 claude-m2 claude-m3 qwen-m4-400 issues-v1 issues-v1-title issues-v1-noleak)
# every dense vector set of the repo except transfer (.from-*), pooled (.pooled), ColBERT and qwen3-4b rows (scored by their own scripts)
DENSE=($(ls "$OUT/$REPO/vectors/" | sed -n 's/\.json$//p' | grep -vE '\.(from-|pooled)|^lfm25-colbert$|-remeasure$|^qwen3-4b$' | sort))
# only vector sets whose matrix is present
DENSE=($(for t in "${DENSE[@]}"; do [ -f "$OUT/$REPO/vectors/$t.npy" ] && echo "$t"; done))
BENCH=(lfm25-emb lfm25-ft-full lfm25-ft-lora jina-0.5b jina-0.5b-ft-full jina-0.5b-ft-lora qwen3-0.6b qwen3-0.6b-ft-full
       qwen3-0.6b-ft-lora lfm25-ft-lora-bug)
log() { echo "=== $REPO $* $(date -u +%H:%M:%S)"; }

for set in "${SETS[@]}"; do
  Q=$OUT/$REPO/questions/$set.jsonl
  [ -f "$Q" ] || { log "no $set, skipped"; continue; }
  log "score $set"
  if [ -z "${TUNA_EVAL_ONLY:-}" ]; then  # TUNA_EVAL_ONLY=1: the ranks were made elsewhere; only eval here
  $TUNA bm25 --repo "$REPO" --questions "$Q" --name "bm25_$set" >/dev/null
  for tag in "${DENSE[@]}"; do
    [ -f "$OUT/$REPO/vectors/$tag.json" ] || continue
    $TUNA dense --repo "$REPO" --questions "$Q" --vectors "$tag" --name "${tag}_$set" >/dev/null
  done
  $TUNA colbert --repo "$REPO" --questions "$Q" --tag lfm25-colbert --name "lfm25-colbert_$set" --batch 64 >/dev/null
  fi
  # eval every ranking of the set, so rows scored by other scripts (transfer, pooled, fusion, doc2query)
  # stay in eval_<set>.json when this script rescores the repo
  ranks=($(ls "$OUT/$REPO/ranks/" | sed -n "s/\(.*_$set\)\.jsonl$/\1/p"))
  $TUNA eval --repo "$REPO" --questions "$Q" --ranks "${ranks[@]}" --save "$OUT/$REPO/eval_$set.json"
done

[ -n "${TUNA_NO_BENCH:-}" ] && { log "done (no bench: TUNA_NO_BENCH)"; exit 0; }
for tag in "${BENCH[@]}"; do
  [ -f "$OUT/$REPO/vectors/$tag.json" ] || continue
  # already measured on these vectors: keep it (a rescore does not re-run the benchmark)
  if [ -s "$OUT/$REPO/search-bench_${tag}_cpu.json" ] && [ "$OUT/$REPO/search-bench_${tag}_cpu.json" -nt "$OUT/$REPO/vectors/$tag.npy" ]; then
    continue
  fi
  log "search bench $tag"
  .venv/bin/python scripts/search-bench.py --repo "$REPO" --tag "$tag" --questions qwen-m2 --device cuda \
    --rebuild-sample 0 > "$OUT/$REPO/search-bench_${tag}_cuda.json"
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/search-bench.py --repo "$REPO" --tag "$tag" --questions qwen-m2 \
    --device cpu --rebuild-sample 32 > "$OUT/$REPO/search-bench_${tag}_cpu.json"
done
log "done"
