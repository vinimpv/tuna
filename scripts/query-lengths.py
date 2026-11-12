"""Query lengths of the issue sets in each model's own tokens, prompt included, against the model's query limit.

  HF_HUB_OFFLINE=1 .venv/bin/python scripts/query-lengths.py --json $TUNA_OUT/tables/query-lengths.json

Tokenizers load at the pinned revisions from the local Hugging Face cache (no download). The limit is the
max_seq_length that sentence-transformers applies to a query: 512 for LFM2.5-Embedding-350M (its
sentence_bert_config.json), 32,768 for jina-code-embeddings-0.5b and Qwen3-Embedding-0.6B. The prompt is the one the
rows used: LFM2.5's own "query: ", none for Jina (empty prompt, see README, Models), the code-search instruction for Qwen3.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

OUT = common.default_out()
QWEN3 = "Instruct: Given a developer's question about a codebase, retrieve the code that answers it\nQuery:"
MODELS = [("LiquidAI/LFM2.5-Embedding-350M", "f35ae2c91d687658dbf1f2b449382f0b019b9808", 512, "query: "),
          ("jinaai/jina-code-embeddings-0.5b", "4db235132dafbe56a8b9c5f59b59795ecf58a4a7", 32768, ""),
          ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", 32768, QWEN3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    qs = [json.loads(l)["question"] for r in ("saleor", "zulip", "posthog")
          for l in open(os.path.join(OUT, r, "questions", "issues-v1.jsonl"))]
    rows = []
    for name, rev, limit, prompt in MODELS:
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        n = [len(tok(prompt + q)["input_ids"]) for q in qs]
        rows.append({"model": name, "revision": rev, "limit": limit, "issues": len(n), "over_limit": sum(x > limit for x in n),
                     "max_tokens": max(n), "median_tokens": sorted(n)[len(n) // 2]})
        print(rows[-1])
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
