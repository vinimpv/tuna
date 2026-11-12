#!/usr/bin/env bash
# Milestone one: stock (untrained) retrieval baselines on one question set.
#
#   scripts/m1-baselines.sh [QUESTION_SET] [REPO...]     # default: qwen-m2 saleor zulip posthog
#
# Every model runs locally on the GPU at its pinned revision and reads each symbol
# at its own maximum input length; only longer symbols are split (best chunk wins).
# Embeddings already built for the same symbols.jsonl are reused. Logs and a
# summary land in $TUNA_OUT/<repo>/; scores in eval_<set>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=${1:-qwen-m2}; shift || true
REPOS=("$@"); [ ${#REPOS[@]} -gt 0 ] || REPOS=(saleor zulip posthog)
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}

QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'

# tag | tuna embed arguments
MODELS=(
  "lfm25-emb|--model LiquidAI/LFM2.5-Embedding-350M --batch 128"
  "jina-0.5b|--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64"
  "qwen3-0.6b|--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64 --query-prefix QWEN3_QUERY"
  "jina-1.5b|--model jinaai/jina-code-embeddings-1.5b --revision 39aeb4fb9b60f930934c78ae5d749a46287c248a --batch 64"
)

for repo in "${REPOS[@]}"; do
  Q=$OUT/$repo/questions/$SET.jsonl
  [ -f "$Q" ] || { echo "no $Q" >&2; exit 1; }
  ranks=(bm25_$SET)
  echo "=== $repo bm25 $(date -u +%H:%M:%S)"
  $TUNA bm25 --repo "$repo" --questions "$Q" --name "bm25_$SET"
  for spec in "${MODELS[@]}"; do
    tag=${spec%%|*}; read -r -a margs <<< "${spec#*|}"
    for i in "${!margs[@]}"; do [ "${margs[$i]}" = QWEN3_QUERY ] && margs[$i]=$QWEN3_QUERY; done
    meta=$OUT/$repo/vectors/$tag.json
    sha=$(sha256sum "$OUT/$repo/symbols.jsonl" | cut -d' ' -f1)
    if [ -f "$meta" ] && grep -q "\"symbols_sha256\": \"$sha\"" "$meta"; then
      echo "=== $repo $tag: reusing vectors"
    else
      echo "=== $repo $tag embed $(date -u +%H:%M:%S)"
      $TUNA embed --repo "$repo" --backend local --tag "$tag" "${margs[@]}"
    fi
    echo "=== $repo $tag dense $(date -u +%H:%M:%S)"
    $TUNA dense --repo "$repo" --questions "$Q" --vectors "$tag" --name "${tag}_$SET"
    ranks+=("${tag}_$SET")
  done
  echo "=== $repo colbert $(date -u +%H:%M:%S)"
  $TUNA colbert --repo "$repo" --questions "$Q" --tag lfm25-colbert --name "lfm25-colbert_$SET" --batch 64
  ranks+=("lfm25-colbert_$SET")
  echo "=== $repo eval $(date -u +%H:%M:%S)"
  $TUNA eval --repo "$repo" --questions "$Q" --ranks "${ranks[@]}" --save "$OUT/$repo/eval_$SET.json"
done
echo "=== done $(date -u +%H:%M:%S)"
