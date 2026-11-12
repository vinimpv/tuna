"""Unrounded intervals for the six cells of the first run's search-use split (arm - A, gold file).

  <tuna .venv>/bin/python -I scripts/search-split-unrounded.py [--out $TUNA_OUT] [--json $TUNA_OUT/checks/search-split-unrounded.json]

cc-bench/first-run-search-split.json stores bounds rounded to 4 decimals; D - A on D's search issues has
a stored lower bound of 0.0. This recomputes every cell with the harness's bootstrap (as cc-analysis.py
search_split: gold-file clusters over all issues restricted to the split, within repository, numpy default_rng(0), 10,000
replicates) on the latest runs of each issue and arm (service failures left out, agent failures as misses),
checks that rounding reproduces the stored cell, and records whether the interval lies strictly on one side
of zero. Reads cc-bench/first-run-search-split.json (written by cc-analysis.py) and the issues bench's
questions.jsonl and runs.jsonl. Needs numpy.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REPOS = ["saleor", "zulip", "posthog"]


def latest(out):
    """(repo/qid, arm) -> the run record that counts: service failures left out, agent failures as misses."""
    L = {}
    for r in REPOS:
        for l in open(os.path.join(out, r, "cc-bench", "issues", "runs.jsonl")):
            d = json.loads(l)
            st = {"error": "agent_fail"}.get(d["status"], d["status"])
            if st == "service_fail":
                continue
            if st == "agent_fail":
                d.update(file_hit=False, symbol_hit=False)
            L[(f"{r}/{d['qid']}", d["arm"])] = d
    return L


def boot_sub(np, vals, qs, cl_all, n_boot=10000):
    ks = set(qs)
    cl = {r: [[q for q in c if q in ks] for c in cs if any(q in ks for q in c)] for r, cs in cl_all.items()}
    cl = {r: c for r, c in cl.items() if c}
    rng = np.random.default_rng(0)
    ts, tc = np.zeros(n_boot), np.zeros(n_boot)
    for repo in sorted(cl):
        S = np.array([sum(vals[q] for q in x) for x in cl[repo]], dtype=float)
        C = np.array([len(x) for x in cl[repo]], dtype=float)
        idx = rng.integers(0, len(cl[repo]), size=(n_boot, len(cl[repo])))
        ts += S[idx].sum(axis=1)
        tc += C[idx].sum(axis=1)
    m = ts / tc
    return [sum(vals.values()) / len(vals), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--json", default=os.path.join(common.default_out(), "checks", "search-split-unrounded.json"))
    args = ap.parse_args()
    L = latest(args.out)
    Q = {f"{r}/{json.loads(l)['id']}": json.loads(l) for r in REPOS
         for l in open(os.path.join(args.out, r, "cc-bench", "issues", "questions.jsonl"))}
    gf = {q: v["gold_files"] for q, v in Q.items()}
    qids = sorted(q for q in Q if all((q, a) in L for a in "ABCD"))
    sp = json.load(open(os.path.join(args.out, "cc-bench", "first-run-search-split.json")))
    import numpy as np
    # clusters as the harness builds them for this split (cc-analysis.py search_split): gold-file clusters
    # over all issues, then restricted to the split's issues, keeping their order (the order fixes which
    # cluster each random draw picks, so it must match to reproduce the stored cells)
    cl_all = {}
    for q in qids:
        cl_all.setdefault(q.split("/", 1)[0], {}).setdefault(min(gf[q]), []).append(q)
    cl_all = {r: list(c.values()) for r, c in cl_all.items()}
    out = []
    for arm in "BCD":
        for split, key in (("search", "search issues"), ("no_search", "no-search issues")):
            qs = [q for q in qids if ((L[(q, arm)].get("searches") or 0) > 0) == (split == "search")]
            vals = {q: float(bool(L[(q, arm)]["file_hit"])) - float(bool(L[(q, "A")]["file_hit"])) for q in qs}
            un = boot_sub(np, vals, qs, cl_all)
            stored = sp["arms"][arm][key][f"{arm}-A"]
            out.append({"arm": arm, "split": split, "issues": len(qs), "stored": stored, "unrounded": un,
                        "reproduces": [round(x, 4) for x in un] == stored and len(qs) == sp["arms"][arm][key]["issues"],
                        "excludes_zero": un[1] > 0 or un[2] < 0})
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    for c in out:
        print(f"{c['arm']} {c['split']:9s} n {c['issues']:3d} stored {c['stored']} lo {c['unrounded'][1]:+.8f} "
              f"hi {c['unrounded'][2]:+.8f} reproduces {c['reproduces']} excludes zero {c['excludes_zero']}")


if __name__ == "__main__":
    main()
