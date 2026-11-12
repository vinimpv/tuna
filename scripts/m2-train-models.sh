#!/usr/bin/env bash
# Milestone two, one repo: the tuned rows. Same recipe for every model, so the
# comparison is fair:
#   - training questions: qwen-train1 (train split; epoch picked on its val split)
#   - hard negatives: each model's own stock vectors (tuna embed tag), depth 30,
#     3 per question per step, pool = training files only (held-out masked),
#     gold / enclosing / nested symbols skipped; false-negative filter with one
#     judge for every model: skip a candidate whose stock Qwen3-Embedding-0.6B
#     score is above 0.95 x that judge's score for the gold (the own-score rule's
#     share is measured too, for the ledger)
#   - batch 32 questions (in-batch negatives: every document of the batch),
#     lr 2e-5, up to 8 epochs, patience 3, best epoch by val MRR@10
#   - each model's own prompts, pooling and context length (Qwen3: the
#     code-search instruction it is scored with); symbols over that length split
#   - LFM2.5 (512 tokens) trains on the full batch directly; jina and Qwen3
#     (32k tokens) use GradCache, which gives the same gradients in bounded memory
# LoRA: LFM2.5 only. Then each tuned model embeds the corpus.
# Qwen3's per-epoch validation encodes in 16k-token batches (inference only): its
# training state plus 64k-token inference batches does not fit in 32 GB.
#
#   scripts/m2-train-models.sh REPO
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=${1:?usage: m2-train-models.sh REPO}
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
COMMON=(--questions qwen-train1 --fn-rule judge --fn-judge qwen3-0.6b --fn-ratio 0.95 --mine-depth 30 --hard-negatives 3 --batch 32 --lr 2e-5
        --epochs 8 --patience 3 --grad-checkpointing)
log() { echo "=== $REPO $* $(date -u +%H:%M:%S)"; }

run() {  # name, embed args..., -- train args...
  local name=$1; shift
  local embed=() train=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do embed+=("$1"); shift; done
  shift
  train=("$@")
  if [ ! -f "$OUT/$REPO/models/$name/tuna-model.json" ]; then
    log "train $name"
    $TUNA train --repo "$REPO" --name "$name" "${COMMON[@]}" "${train[@]}"
  fi
  if [ ! -f "$OUT/$REPO/vectors/$name.json" ]; then
    log "embed $name"
    $TUNA embed --repo "$REPO" --backend local --weights "$OUT/$REPO/models/$name" --tag "$name" "${embed[@]}"
  fi
}

run lfm25-ft-full --model LiquidAI/LFM2.5-Embedding-350M --batch 128 \
  -- --mode full --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb
run lfm25-ft-lora --model LiquidAI/LFM2.5-Embedding-350M --batch 128 \
  -- --mode lora --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb --lr 2e-4
run jina-0.5b-ft-full --model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64 \
  -- --mode full --base jinaai/jina-code-embeddings-0.5b --base-revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 \
     --stock-vectors jina-0.5b --grad-cache
run qwen3-0.6b-ft-full --model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64 \
     --query-prefix "$QWEN3_QUERY" \
  -- --mode full --base Qwen/Qwen3-Embedding-0.6B --base-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
     --stock-vectors qwen3-0.6b --grad-cache --query-prefix "$QWEN3_QUERY" --batch-tokens 16384
# LoRA rows for jina and Qwen3: LFM2.5's LoRA recipe (rank 16, alpha 32, dropout 0.05, lr 2e-4,
# every linear layer of the transformer blocks, same selection rule)
run jina-0.5b-ft-lora --model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64 \
  -- --mode lora --lr 2e-4 --base jinaai/jina-code-embeddings-0.5b --base-revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 \
     --stock-vectors jina-0.5b --grad-cache
run qwen3-0.6b-ft-lora --model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64 \
     --query-prefix "$QWEN3_QUERY" \
  -- --mode lora --lr 2e-4 --base Qwen/Qwen3-Embedding-0.6B --base-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
     --stock-vectors qwen3-0.6b --grad-cache --query-prefix "$QWEN3_QUERY" --batch-tokens 16384

# Issue-style training: LFM2.5 LoRA (the recipe that won on Saleor) on qwen-train1 +
# qwen-bug1 merged (scripts/merge-sets.py), validated on the merged val split.
if [ -f "$OUT/$REPO/questions/qwen-trainbug1.train.jsonl" ]; then
  BUG_COMMON=("${COMMON[@]/qwen-train1/qwen-trainbug1}")
  ( COMMON=("${BUG_COMMON[@]}")
    run lfm25-ft-lora-bug --model LiquidAI/LFM2.5-Embedding-350M --batch 128 \
      -- --mode lora --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb --lr 2e-4 )
else
  log "no qwen-trainbug1 yet: lfm25-ft-lora-bug skipped"
fi

# Stock jina and Qwen3 re-encoded at LFM2.5's 512 tokens (same splitting rule, chunks of 512), for every
# repo that lacks them: an embedding pass only, done in whichever train window is open.
for r in saleor zulip posthog; do
  for spec in "jina-0.5b-512|--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 128" \
              "jina-1.5b-512|--model jinaai/jina-code-embeddings-1.5b --revision 39aeb4fb9b60f930934c78ae5d749a46287c248a --batch 128" \
              "qwen3-0.6b-512|--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 128"; do
    tag=${spec%%|*}; read -r -a margs <<< "${spec#*|}"
    extra=(); case $tag in qwen3-*) extra=(--query-prefix "$QWEN3_QUERY") ;; esac
    if [ ! -f "$OUT/$r/vectors/$tag.json" ]; then
      log "embed $r $tag"
      $TUNA embed --repo "$r" --backend local --max-tokens 512 --tag "$tag" "${margs[@]}" "${extra[@]}"
    fi
  done
done

log "done"
