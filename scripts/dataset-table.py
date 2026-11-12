"""Dataset table per repo, and checkpoint sizes (full tuned model vs LoRA adapter) per model.

  .venv/bin/python scripts/dataset-table.py [--repos saleor zulip posthog] [--json OUT]

Per repo: files and symbols in the index, the share of symbols over 512 tokens (LFM2.5 tokenizer, from
the chunking record of vectors/lfm25-emb.json), qwen-train1 train/val questions, held-out files,
questions per test set. Sizes: on-disk bytes of each models/<row>/ directory (a LoRA row stores only
its adapter). In this release (no vectors, no weights, no symbols.jsonl until tuna extract runs) the
chunking share is None and rows without weights are left out of the sizes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

SETS = ["qwen-m2", "claude-m2", "claude-m3", "qwen-m4-400", "issues-v1", "issues-v1-title", "issues-v1-noleak",
        "qwen-bug1"]


def lines(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--json")
    args = ap.parse_args()
    data, sizes = [], []
    for repo in args.repos:
        w = os.path.join(args.out, repo)
        files = set()
        n = 0
        for l in open(os.path.join(w, "symbols.jsonl")):
            files.add(json.loads(l)["file_path"])
            n += 1
        vp = os.path.join(w, "vectors", "lfm25-emb.json")   # not in the release: no chunking share
        ch = json.load(open(vp)).get("chunking", {}) if os.path.exists(vp) else {}
        q = os.path.join(w, "questions")
        row = {"repo": repo, "files": len(files), "symbols": n, "symbols_over_512_lfm": ch.get("split"),
               "share_over_512_lfm": round(ch["split"] / ch["symbols"], 4) if ch.get("symbols") else None,
               "qwen_train1_train": lines(os.path.join(q, "qwen-train1.train.jsonl")),
               "qwen_train1_val": lines(os.path.join(q, "qwen-train1.val.jsonl")),
               "heldout_files": lines(os.path.join(q, "heldout-files.txt")),
               "test_sets": {s: lines(os.path.join(q, f"{s}.jsonl")) for s in SETS
                             if os.path.exists(os.path.join(q, f"{s}.jsonl"))}}
        data.append(row)
        md = os.path.join(w, "models")
        for name in sorted(os.listdir(md)) if os.path.isdir(md) else []:
            d = os.path.join(md, name)
            if not os.path.isfile(os.path.join(d, "tuna-model.json")):
                continue
            if not any(f.endswith(".safetensors") for f in os.listdir(d)):   # card only (no weights): no size
                continue
            b = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
            sizes.append({"repo": repo, "row": name, "bytes": b, "mib": round(b / 2**20, 1)})
    for r in data:
        print(json.dumps(r))
    print(f"\n{'repo':8s} {'row':34s} {'MiB':>8s}")
    for s in sizes:
        print(f"{s['repo']:8s} {s['row']:34s} {s['mib']:8.1f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"datasets": data, "checkpoint_sizes": sizes}, fh, indent=1)


if __name__ == "__main__":
    main()
