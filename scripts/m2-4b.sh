#!/usr/bin/env bash
# Qwen3-Embedding-4B, stock, Saleor only (extra row). Audit before download (in the ledger):
# Apache-2.0, last commit 2025-06-20, revision 5cf2132a, native Qwen3 architecture (no remote code),
# two safetensors shards. Same code-search query instruction as Qwen3-0.6B.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
[ -f "$OUT/saleor/vectors/qwen3-4b.json" ] || $TUNA embed --repo saleor --backend local \
  --model Qwen/Qwen3-Embedding-4B --revision 5cf2132abc99cad020ac570b19d031efec650f2b --batch 32 \
  --token-budget 16384 --query-prefix "$QWEN3_QUERY" --tag qwen3-4b
.venv/bin/python scripts/search-bench.py --repo saleor --tag qwen3-4b --questions qwen-m2 --device cuda \
  --rebuild-sample 0 > "$OUT/saleor/search-bench_qwen3-4b_cuda.json"
echo "=== 4b done $(date -u +%H:%M:%S)"
