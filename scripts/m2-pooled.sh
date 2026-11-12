#!/usr/bin/env bash
# Pooled rows: one model per embedder trained on all three repos' qwen-train1 questions
# (tuna train --pool), the frozen recipe, each repo keeping its own held-out list, corpus,
# mining and validation; then every repo's corpus embedded with the pooled checkpoint and
# its test sets scored (tag <row>.pooled). Needs a GPU.
#
#   scripts/m2-pooled.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
REPOS=(saleor zulip posthog)
SETS=(qwen-m2 claude-m3 qwen-m4-400 claude-m2 issues-v1 issues-v1-title issues-v1-noleak)
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
COMMON=(--questions qwen-train1 --fn-rule judge --fn-judge qwen3-0.6b --fn-ratio 0.95 --mine-depth 30 --hard-negatives 3
        --batch 32 --lr 2e-5 --epochs 8 --patience 3 --grad-checkpointing --repo saleor --pool zulip posthog)
log() { echo "=== pooled $* $(date -u +%H:%M:%S)"; }
JINA=(--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64)
QWEN=(--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64 --query-prefix "$QWEN3_QUERY")
LFM=(--model LiquidAI/LFM2.5-Embedding-350M --batch 128)

train_row() {  # name, train args...
  local name=$1; shift
  [ -f "$OUT/pooled/models/$name/tuna-model.json" ] && { log "$name: already trained"; return; }
  log "train $name"
  $TUNA train --name "$name" "${COMMON[@]}" "$@"
}
train_row lfm25-pooled-full --mode full --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb
train_row lfm25-pooled-lora --mode lora --lr 2e-4 --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb
# jina and Qwen3: the locked variant, LoRA (pre-registered variant rule, resolved on Saleor validation).
train_row jina-0.5b-pooled-lora --mode lora --lr 2e-4 --base jinaai/jina-code-embeddings-0.5b \
  --base-revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --stock-vectors jina-0.5b --grad-cache
train_row qwen3-0.6b-pooled-lora --mode lora --lr 2e-4 --base Qwen/Qwen3-Embedding-0.6B \
  --base-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --stock-vectors qwen3-0.6b --grad-cache \
  --query-prefix "$QWEN3_QUERY" --batch-tokens 16384

for row in lfm25-pooled-full lfm25-pooled-lora jina-0.5b-pooled-lora qwen3-0.6b-pooled-lora; do
  case $row in lfm25-*) ea=("${LFM[@]}") ;; jina-*) ea=("${JINA[@]}") ;; qwen3-*) ea=("${QWEN[@]}") ;; esac
  for r in "${REPOS[@]}"; do
    tag="$row.pooled"
    if [ ! -f "$OUT/$r/vectors/$tag.json" ]; then
      log "embed $r $tag"
      $TUNA embed --repo "$r" --backend local --weights "$OUT/pooled/models/$row" --tag "$tag" "${ea[@]}"
    fi
    for set in "${SETS[@]}"; do
      Q=$OUT/$r/questions/$set.jsonl
      [ -f "$Q" ] || continue
      $TUNA dense --repo "$r" --questions "$Q" --vectors "$tag" --name "${tag}_$set" >/dev/null
    done
  done
done
for r in "${REPOS[@]}"; do
  for set in "${SETS[@]}"; do
    Q=$OUT/$r/questions/$set.jsonl
    [ -f "$Q" ] || continue
    ranks=($(ls "$OUT/$r/ranks/" | sed -n "s/\(.*_$set\)\.jsonl$/\1/p"))
    $TUNA eval --repo "$r" --questions "$Q" --ranks "${ranks[@]}" --save "$OUT/$r/eval_$set.json" >/dev/null
  done
done
log "done"
