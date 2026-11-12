"""BM25 fused with each dense ranking by reciprocal rank fusion (k = 60, nothing tuned), from saved ranks.

  .venv/bin/python scripts/rrf.py [--repos ...] [--sets claude-m3 issues-v1 issues-v1-title issues-v1-noleak]

For each ranks/<row>_<set>.jsonl (row not bm25 or an earlier fusion), the fused score of a symbol is
1/(k + rank in BM25) + 1/(k + rank in the row), ranks 1-based over the saved depth (200); a symbol
missing from one list contributes nothing from it. Writes ranks/rrf-bm25+<row>_<set>.jsonl (top 200)
and prints MRR@10 of the row, of BM25 and of the fusion; --json saves the table.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

K = 60


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        out[r["id"]] = r
    return out


def mrr10(ranked, golds):
    tot = 0.0
    for qid, g in golds.items():
        lst = ranked.get(qid, [])
        best = next((i + 1 for i, f in enumerate(lst[:10]) if f in g), 0)
        tot += 1.0 / best if best else 0.0
    return tot / len(golds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--sets", nargs="+", default=["claude-m3", "issues-v1", "issues-v1-title", "issues-v1-noleak"])
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for repo in args.repos:
        rd = os.path.join(args.out, repo, "ranks")
        for st in args.sets:
            qp = os.path.join(args.out, repo, "questions", f"{st}.jsonl")
            bp = os.path.join(rd, f"bm25_{st}.jsonl")
            if not (os.path.exists(qp) and os.path.exists(bp)):
                continue
            golds = {}
            for l in open(qp):
                q = json.loads(l)
                golds[q["id"]] = set(q.get("gold_fqns") or [q["gold_fqn"]])
            bm = load(bp)
            bm_ranked = {k: v["ranked"] for k, v in bm.items()}
            for fn in sorted(os.listdir(rd)):
                if not fn.endswith(f"_{st}.jsonl"):
                    continue
                name = fn[:-len(f"_{st}.jsonl")]
                if name == "bm25" or name.startswith("rrf-"):
                    continue
                dense = load(os.path.join(rd, fn))
                fused = {}
                with open(os.path.join(rd, f"rrf-bm25+{name}_{st}.jsonl"), "w") as fh:
                    for qid in golds:
                        sc = {}
                        for lst in (bm_ranked.get(qid, []), dense.get(qid, {}).get("ranked", [])):
                            for i, f in enumerate(lst):
                                sc[f] = sc.get(f, 0.0) + 1.0 / (K + i + 1)
                        top = sorted(sc, key=lambda f: -sc[f])[:200]
                        fused[qid] = top
                        rec = {"id": qid, "ranked": top}
                        if qid in bm and "question" in bm[qid]:
                            rec["question"] = bm[qid]["question"]
                        fh.write(json.dumps(rec) + "\n")
                rows.append({"repo": repo, "set": st, "row": name, "n": len(golds),
                             "mrr10_row": round(mrr10({k: v["ranked"] for k, v in dense.items()}, golds), 4),
                             "mrr10_bm25": round(mrr10(bm_ranked, golds), 4), "mrr10_rrf": round(mrr10(fused, golds), 4)})
    print(f"{'repo':8s} {'set':17s} {'row':32s} {'n':>4s} {'row':>6s} {'bm25':>6s} {'rrf':>6s} {'rrf-row':>7s}")
    for r in rows:
        print(f"{r['repo']:8s} {r['set']:17s} {r['row']:32s} {r['n']:4d} {r['mrr10_row']:6.3f} {r['mrr10_bm25']:6.3f} "
              f"{r['mrr10_rrf']:6.3f} {r['mrr10_rrf'] - r['mrr10_row']:+7.3f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"k": K, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
