#!/usr/bin/env bash
# Training seeds: each model's headline row on Saleor, two more seeds (1, 2; the
# existing row is seed 0). The headline variant per model comes from the pre-registered rule: the
# higher Saleor qwen-train1 val MRR@10 of full and LoRA; the resolution is written to the ledger.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TUNA=.venv/bin/tuna
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
QWEN3_QUERY=$'Instruct: Given a developer\'s question about a codebase, retrieve the code that answers it\nQuery:'
HEAD=$(python3 -I - "$OUT/saleor/models" <<'PY'
import datetime, json, os, sys
m = sys.argv[1]
out, res = [], {}
for model in ("lfm25", "jina-0.5b", "qwen3-0.6b"):
    v = {k: json.load(open(f"{m}/{model}-ft-{k}/tuna-model.json"))["best_val"]["MRR@10"] for k in ("full", "lora")}
    head = "lora" if v["lora"] > v["full"] else "full"
    res[model] = {"saleor_val_mrr10": v, "headline": head}
    out.append(f"{model}-ft-{head}")
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
line = {"stage": "note", "repo": "saleor", "model": None, "start_utc": now, "end_utc": now, "wall_seconds": 0.0,
        "status": "ok", "config": {}, "counts": {}, "tokens": {}, "outputs": [],
        "extra": {"kind": "preregistration resolution check (seeds script, recomputed)",
                  "note": "Headline variant per model by the pre-registered variant rule (higher Saleor qwen-train1 "
                          "val MRR@10; no Zulip, PostHog or test-set number used).", "resolution": res}}
d = os.path.expanduser(os.environ.get("TUNA_METRICS") or os.path.join(os.environ.get("TUNA_DATA", "~/tuna-data"), "metrics"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "ledger.jsonl"), "a").write(json.dumps(line) + "\n")
print(" ".join(out))
PY
)
echo "=== headline rows: $HEAD"
COMMON=(--repo saleor --questions qwen-train1 --fn-rule judge --fn-judge qwen3-0.6b --fn-ratio 0.95 --mine-depth 30
        --hard-negatives 3 --batch 32 --epochs 8 --patience 3 --grad-checkpointing)
for row in $HEAD; do
  case $row in
    lfm25-*) base=(--base LiquidAI/LFM2.5-Embedding-350M --stock-vectors lfm25-emb)
             emb=(--model LiquidAI/LFM2.5-Embedding-350M --batch 128) ;;
    jina-*) base=(--base jinaai/jina-code-embeddings-0.5b --base-revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7
                  --stock-vectors jina-0.5b --grad-cache)
            emb=(--model jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7 --batch 64) ;;
    qwen3-*) base=(--base Qwen/Qwen3-Embedding-0.6B --base-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
                   --stock-vectors qwen3-0.6b --grad-cache --query-prefix "$QWEN3_QUERY" --batch-tokens 16384)
             emb=(--model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --batch 64
                  --query-prefix "$QWEN3_QUERY") ;;
  esac
  case $row in *-lora) mode=(--mode lora --lr 2e-4) ;; *) mode=(--mode full --lr 2e-5) ;; esac
  for seed in 1 2; do
    name=$row-s$seed
    if [ ! -f "$OUT/saleor/models/$name/tuna-model.json" ]; then
      echo "=== seeds train $name $(date -u +%H:%M:%S)"
      $TUNA train --name "$name" --seed "$seed" "${COMMON[@]}" "${mode[@]}" "${base[@]}"
    fi
    [ -f "$OUT/saleor/vectors/$name.json" ] || $TUNA embed --repo saleor --backend local \
      --weights "$OUT/saleor/models/$name" --tag "$name" "${emb[@]}"
  done
done
echo "=== seeds done $(date -u +%H:%M:%S)"
