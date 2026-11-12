"""Cross-repo rows in the split shape: each tuning gain split into a generic part and a repo part.

  .venv/bin/python scripts/transfer-split.py [--sets claude-m3 issues-v1] [--json OUT]

For each model's locked row (default: the LoRA rows) and each target repo: MRR@10 of the stock row,
of the row tuned on another repo (mean of the two source repos, tags <row>.from-<src>), and of the row
tuned on this repo. Then one interval per model and set, pooled over the target repos (stratified by
repo, gold files resampled within each repo): tuned here minus tuned elsewhere, where a question's
"elsewhere" score is the mean of its two transferred scores. Reads eval_<set>.json per-question ranks.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

MODELS = {"LFM2.5-350M": ("lfm25-emb", "lfm25-ft-lora"), "jina-0.5B": ("jina-0.5b", "jina-0.5b-ft-lora"),
          "Qwen3-0.6B": ("qwen3-0.6b", "qwen3-0.6b-ft-lora")}
REPOS = ["saleor", "zulip", "posthog"]


def rr10(r):
    return 1.0 / r if 0 < r <= 10 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--sets", nargs="+", default=["claude-m3", "issues-v1"])
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    cells, intervals = [], []
    file_of = {}
    for repo in REPOS:
        file_of[repo] = {}
        for l in open(os.path.join(args.out, repo, "symbols.jsonl")):
            s = json.loads(l)
            file_of[repo][s["fqn"]] = s["file_path"]
    for st in args.sets:
        for model, (stock, tuned) in MODELS.items():
            parts = []
            for repo in REPOS:
                path = os.path.join(args.out, repo, f"eval_{st}.json")
                if not os.path.exists(path):
                    continue
                per = json.load(open(path))["per_question_gold_rank"]
                srcs = [s for s in REPOS if s != repo]
                keys = [f"{stock}_{st}", f"{tuned}_{st}"] + [f"{tuned}.from-{s}_{st}" for s in srcs]
                if any(k not in per for k in keys):
                    cells.append({"set": st, "model": model, "repo": repo, "missing": [k for k in keys if k not in per]})
                    continue
                qids = sorted(per[keys[0]])
                st_v = [rr10(per[keys[0]][q]) for q in qids]
                own = [rr10(per[keys[1]][q]) for q in qids]
                tr = [[rr10(per[k][q]) for q in qids] for k in keys[2:]]
                other = [(a + b) / 2 for a, b in zip(*tr)]
                n = len(qids)
                cells.append({"set": st, "model": model, "repo": repo, "n": n, "stock": round(sum(st_v) / n, 4),
                              "tuned_elsewhere": round(sum(other) / n, 4),
                              "tuned_elsewhere_by_source": {s: round(sum(t) / n, 4) for s, t in zip(srcs, tr)},
                              "tuned_here": round(sum(own) / n, 4)})
                qs = {}
                for l in open(os.path.join(args.out, repo, "questions", f"{st}.jsonl")):
                    q = json.loads(l)
                    g = (q.get("gold_fqns") or [q["gold_fqn"]])[0]
                    qs[q["id"]] = q.get("file") or file_of[repo].get(g, g)
                groups = {}
                for i, q in enumerate(qids):
                    groups.setdefault(qs.get(q, q), []).append(i)
                parts.append((repo, [a - b for a, b in zip(own, other)], list(groups.values()),
                              [a - b for a, b in zip(other, st_v)]))
            if len(parts) < 2:
                continue
            for label, k in (("here_minus_elsewhere", 1), ("elsewhere_minus_stock", 3)):
                d = [x for p in parts for x in p[k]]
                rng = random.Random(args.seed)
                boots = []
                for _ in range(args.samples):
                    tot = cnt = 0
                    for p in parts:
                        v, gl = p[k], p[2]
                        for _ in gl:
                            for i in gl[rng.randrange(len(gl))]:
                                tot += v[i]
                                cnt += 1
                    boots.append(tot / cnt)
                boots.sort()
                lo, hi = boots[int(0.025 * args.samples)], boots[int(0.975 * args.samples) - 1]
                intervals.append({"set": st, "model": model, "row": tuned, "repos": [p[0] for p in parts], "n": len(d),
                                  "quantity": label, "diff": round(sum(d) / len(d), 4),
                                  "ci95": [round(lo, 4), round(hi, 4)], "sig": lo > 0 or hi < 0})
    for c in cells:
        if "missing" in c:
            print(f"{c['set']:10s} {c['model']:12s} {c['repo']:8s} missing {c['missing']}")
        else:
            print(f"{c['set']:10s} {c['model']:12s} {c['repo']:8s} n={c['n']:3d} stock {c['stock']:.3f}  "
                  f"tuned elsewhere {c['tuned_elsewhere']:.3f}  tuned here {c['tuned_here']:.3f}")
    for i in intervals:
        print(f"{i['set']:10s} {i['model']:12s} pooled {'+'.join(i['repos'])} n={i['n']} {i['quantity']} "
              f"{i['diff']:+.3f} [{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}]{'  sig' if i['sig'] else ''}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"samples": args.samples, "seed": args.seed, "cells": cells, "intervals": intervals}, fh, indent=1)


if __name__ == "__main__":
    main()
