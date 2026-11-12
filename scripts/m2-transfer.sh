#!/usr/bin/env bash
# Cross-repo transfer matrix: a model tuned on repo A, scored on repo B (A != B).
# Scoring only: the target repo's corpus is embedded with the source repo's
# checkpoint (tag <row>.from-<A>), then every test set of B is ranked and scored
# next to B's own stock and tuned rows. Needs a GPU.
#
#   scripts/m2-transfer.sh [ROW...]      # default: the full and LoRA tunes of all three models
set -euo pipefail
cd "$(dirname "$0")/.."
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
REPOS=(saleor zulip posthog)
SETS=(qwen-m2 claude-m3 qwen-m4-400 claude-m2 issues-v1 issues-v1-title)
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
ROWS=("$@"); [ ${#ROWS[@]} -gt 0 ] || ROWS=(lfm25-ft-full lfm25-ft-lora jina-0.5b-ft-full jina-0.5b-ft-lora
                                         qwen3-0.6b-ft-full qwen3-0.6b-ft-lora)
log() { echo "=== $* $(date -u +%H:%M:%S)"; }
embed_args() {
  case $1 in
    lfm25-*) echo "--model LiquidAI/LFM2.5-Embedding-350M --batch 128" ;;
    jina-0.5b-*) echo "--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64" ;;
    qwen3-0.6b-*) echo "--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64" ;;
  esac
}
for row in "${ROWS[@]}"; do
  for src in "${REPOS[@]}"; do
    ckpt=$OUT/$src/models/$row
    [ -f "$ckpt/tuna-model.json" ] || { log "no $src/$row, skipped"; continue; }
    for dst in "${REPOS[@]}"; do
      [ "$src" = "$dst" ] && continue
      tag="$row.from-$src"
      if [ ! -f "$OUT/$dst/vectors/$tag.json" ]; then
        log "embed $dst with $src/$row"
        read -r -a ea <<< "$(embed_args "$row")"
        extra=(); case $row in qwen3-*) extra=(--query-prefix "$QWEN3_QUERY") ;; esac
        $TUNA embed --repo "$dst" --backend local --weights "$ckpt" --tag "$tag" "${ea[@]}" "${extra[@]}"
      fi
      for set in "${SETS[@]}"; do
        Q=$OUT/$dst/questions/$set.jsonl
        [ -f "$Q" ] || continue
        $TUNA dense --repo "$dst" --questions "$Q" --vectors "$tag" --name "${tag}_$set" >/dev/null
      done
    done
  done
done
for dst in "${REPOS[@]}"; do
  for set in "${SETS[@]}"; do
    Q=$OUT/$dst/questions/$set.jsonl
    [ -f "$Q" ] || continue
    ranks=($(ls "$OUT/$dst/ranks/" | sed -n "s/\(.*_$set\)\.jsonl$/\1/p" | grep -vE '\.tmp$'))
    $TUNA eval --repo "$dst" --questions "$Q" --ranks "${ranks[@]}" --save "$OUT/$dst/eval_$set.json" >/dev/null
  done
done
log "done"
