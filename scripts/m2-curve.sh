#!/usr/bin/env bash
# Question-count curve (extra row): LFM2.5 LoRA on Saleor with nested random subsets of the
# qwen-train1 train split (150, 400, 800 questions; one fixed seed), the same val split, recipe,
# negatives rule and selection rule as the 1,349-question row. Needs a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
Q=$OUT/saleor/questions
python3 -I - "$Q" <<'PY'
import json, random, sys
q = sys.argv[1]
train = [json.loads(l) for l in open(f"{q}/qwen-train1.train.jsonl")]
val = open(f"{q}/qwen-train1.val.jsonl").read()
order = list(range(len(train)))
random.Random(20261114).shuffle(order)          # one fixed order: smaller sets are prefixes (nested)
for n in (150, 400, 800):
    rows = [train[i] for i in sorted(order[:n])]
    open(f"{q}/qwen-train1-n{n}.train.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
    open(f"{q}/qwen-train1-n{n}.val.jsonl", "w").write(val)
    open(f"{q}/qwen-train1-n{n}.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
print("subsets written")
PY
for n in 150 400 800; do
  name=lfm25-ft-lora-n$n
  $TUNA heldout --repo saleor --check qwen-train1-n$n
  if [ ! -f "$OUT/saleor/models/$name/tuna-model.json" ]; then
    echo "=== curve train $name $(date -u +%H:%M:%S)"
    $TUNA train --repo saleor --name "$name" --questions qwen-train1-n$n --mode lora --lr 2e-4 \
      --base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb --fn-rule judge --fn-judge qwen3-0.6b \
      --fn-ratio 0.95 --mine-depth 30 --hard-negatives 3 --batch 32 --epochs 8 --patience 3 --grad-checkpointing
  fi
  [ -f "$OUT/saleor/vectors/$name.json" ] || $TUNA embed --repo saleor --backend local \
    --model LiquidAI/LFM2.5-Embedding-350M --weights "$OUT/saleor/models/$name" --tag "$name" --batch 128
done
echo "=== curve done $(date -u +%H:%M:%S)"
