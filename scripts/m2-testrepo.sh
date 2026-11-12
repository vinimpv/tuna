#!/usr/bin/env bash
# Transfer-only test repo (Dispatch, Home Assistant): score on claude-t400 with BM25, the stock rows
# (LFM2.5-Embedding, jina 0.5B and 1.5B, Qwen3-Embedding-0.6B, LFM2.5-ColBERT), the nine locked LoRA
# adapters (3 models x 3 training repos; tag <row>.from-<src>) and the pooled models when they exist
# (tag <row>.pooled). Nothing trains here. Needs a GPU.
#
#   scripts/m2-testrepo.sh REPO
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=${1:?usage: m2-testrepo.sh REPO}
SET=${SET:-claude-t400}
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
Q=$OUT/$REPO/questions/$SET.jsonl
[ -f "$Q" ] || { echo "no $Q" >&2; exit 1; }
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
log() { echo "=== testrepo $REPO $* $(date -u +%H:%M:%S)"; }
args_for() {
  case $1 in
    lfm25-*) echo "--model LiquidAI/LFM2.5-Embedding-350M --batch 128" ;;
    jina-0.5b*) echo "--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64" ;;
    jina-1.5b*) echo "--model jinaai/jina-code-embeddings-1.5b --revision 39aeb4fb9b60f930934c78ae5d749a46287c248a --batch 64" ;;
    qwen3-0.6b*) echo "--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64" ;;
  esac
}
embed() {  # tag [weights dir]
  local tag=$1 w=${2:-}
  [ -f "$OUT/$REPO/vectors/$tag.json" ] && return
  read -r -a ea <<< "$(args_for "$tag")"
  extra=(); case $tag in qwen3-*) extra=(--query-prefix "$QWEN3_QUERY") ;; esac
  [ -n "$w" ] && extra+=(--weights "$w")
  log "embed $tag"
  $TUNA embed --repo "$REPO" --backend local --tag "$tag" "${ea[@]}" "${extra[@]}"
}
tags=()
for t in lfm25-emb jina-0.5b jina-1.5b qwen3-0.6b; do embed "$t"; tags+=("$t"); done
for row in lfm25-ft-lora jina-0.5b-ft-lora qwen3-0.6b-ft-lora; do
  for src in saleor zulip posthog; do
    ck=$OUT/$src/models/$row
    [ -f "$ck/tuna-model.json" ] || { log "no $src/$row"; continue; }
    embed "$row.from-$src" "$ck"; tags+=("$row.from-$src")
  done
done
for row in lfm25-pooled-full lfm25-pooled-lora jina-0.5b-pooled-lora qwen3-0.6b-pooled-lora; do
  ck=$OUT/pooled/models/$row
  [ -f "$ck/tuna-model.json" ] || continue
  embed "$row.pooled" "$ck"; tags+=("$row.pooled")
done
log "score"
$TUNA bm25 --repo "$REPO" --questions "$Q" --name "bm25_$SET" >/dev/null
for t in "${tags[@]}"; do $TUNA dense --repo "$REPO" --questions "$Q" --vectors "$t" --name "${t}_$SET" >/dev/null; done
$TUNA colbert --repo "$REPO" --questions "$Q" --tag lfm25-colbert --name "lfm25-colbert_$SET" --batch 64 >/dev/null
ranks=($(ls "$OUT/$REPO/ranks/" | sed -n "s/\(.*_$SET\)\.jsonl$/\1/p"))
$TUNA eval --repo "$REPO" --questions "$Q" --ranks "${ranks[@]}" --save "$OUT/$REPO/eval_$SET.json"
log "done"
