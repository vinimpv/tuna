"""Symbol- and file-level metrics for every ranking on the issue sets (post-processing of saved ranks).

  .venv/bin/python scripts/issue-metrics.py [--repos saleor zulip posthog] [--sets issues-v1 issues-v1-title issues-v1-noleak]

For each ranking file ranks/<row>_<set>.jsonl: MRR@10 and hit@10 over symbols (as tuna eval), and
file-level hit@10: the ranked symbols mapped to their files, duplicates dropped in rank order, hit
when any gold symbol's file is among the first 10 files. Prints a table; --json saves it.
The rank files are not in this release: rebuild them with scripts/m2-score.sh first.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--sets", nargs="+", default=["issues-v1", "issues-v1-title", "issues-v1-noleak"])
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for repo in args.repos:
        w = os.path.join(args.out, repo)
        file_of = {}
        for l in open(os.path.join(w, "symbols.jsonl")):
            s = json.loads(l)
            file_of[s["fqn"]] = s["file_path"]
        for st in args.sets:
            qpath = os.path.join(w, "questions", f"{st}.jsonl")
            if not os.path.exists(qpath):
                continue
            golds = {}
            for l in open(qpath):
                q = json.loads(l)
                golds[q["id"]] = set(q.get("gold_fqns") or [q["gold_fqn"]])
            for fn in sorted(os.listdir(os.path.join(w, "ranks"))):
                if not fn.endswith(f"_{st}.jsonl"):
                    continue
                name = fn[:-len(f"_{st}.jsonl")]
                ranked = {}
                for l in open(os.path.join(w, "ranks", fn)):
                    r = json.loads(l)
                    ranked[r["id"]] = r["ranked"]
                rr, hit, fhit = 0.0, 0, 0
                for qid, g in golds.items():
                    lst = ranked.get(qid, [])
                    best = next((i + 1 for i, f in enumerate(lst) if f in g), 0)
                    rr += 1.0 / best if 0 < best <= 10 else 0.0
                    hit += 0 < best <= 10
                    gfiles = {file_of.get(x) for x in g}
                    files, seen = [], set()
                    for f in lst:
                        p = file_of.get(f)
                        if p not in seen:
                            seen.add(p)
                            files.append(p)
                        if len(files) == 10:
                            break
                    fhit += bool(gfiles & set(files))
                n = len(golds)
                rows.append({"repo": repo, "set": st, "row": name, "n": n, "mrr@10": round(rr / n, 4),
                             "hit@10": round(hit / n, 4), "file_hit@10": round(fhit / n, 4)})
    print(f"{'repo':8s} {'set':17s} {'row':32s} {'n':>4s} {'MRR@10':>7s} {'hit@10':>7s} {'file hit@10':>11s}")
    for r in rows:
        print(f"{r['repo']:8s} {r['set']:17s} {r['row']:32s} {r['n']:4d} {r['mrr@10']:7.3f} {r['hit@10']:7.3f} "
              f"{r['file_hit@10']:11.3f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
