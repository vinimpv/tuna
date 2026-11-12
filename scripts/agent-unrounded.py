"""Unrounded interval bounds for the agent cells whose stored bound is exactly 0.0.

  .venv/bin/python scripts/agent-unrounded.py [--out $TUNA_OUT] [--json $TUNA_OUT/checks/agent-unrounded.json]

cc-bench stores interval bounds rounded to 4 decimals, so a stored 0.0 does not say on which side of
zero the bound lies. For every paired cell of both experiments (cc-bench/issues.report.json and
hint-full.report.json, every population, pooled and per repository) with a stored bound of 0.0, this
recomputes the interval with the harness's own method (scripts/cc-bench.py:
load_latest, summarize, cluster_bootstrap: gold-file clusters within repository, numpy default_rng(0),
10,000 replicates) without rounding, and checks that rounding it reproduces the stored cell. The
summary's tables use the result, so "strictly on one side of
zero, judged on the unrounded bounds" is applied exactly. Needs numpy (the project venv).
"""
import argparse
import collections
import json
import os

import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

REPOS = ["saleor", "zulip", "posthog"]
N_BOOT = 10000
OUTCOMES = ("file_hit", "symbol_hit", "total_cost_alone_usd", "cost_usd")   # the cells the appendix prints


def latest(out, bench):
    L = {}
    for r in REPOS:
        for l in open(os.path.join(out, r, "cc-bench", bench, "runs.jsonl")):
            d = json.loads(l)
            st = {"error": "agent_fail"}.get(d["status"], d["status"])
            if st == "service_fail":
                continue
            if st == "agent_fail":
                d.update(file_hit=False, symbol_hit=False)
            if d.get("cost_alone_usd") is not None:
                d["total_cost_alone_usd"] = round(d["cost_alone_usd"] + (d.get("hook_cost_usd") or 0), 6)
            d.setdefault("total_cost_usd", d.get("cost_usd"))
            L[(f"{r}/{d['qid']}", d["arm"])] = d
    return L


def questions(out, bench):
    return {f"{r}/{json.loads(l)['id']}": json.loads(l) for r in REPOS
            for l in open(os.path.join(out, r, "cc-bench", bench, "questions.jsonl"))}


def boot(vals, qids, gold_files):
    cl = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in qids:
        cl[q.split("/", 1)[0]][min(gold_files[q])].append(q)
    cl = {r: list(c.values()) for r, c in cl.items()}
    rng = np.random.default_rng(0)
    ts, tc = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for repo in sorted(cl):
        c = cl[repo]
        S = np.array([sum(vals[q] for q in x) for x in c], dtype=float)
        C = np.array([len(x) for x in c], dtype=float)
        idx = rng.integers(0, len(c), size=(N_BOOT, len(c)))
        ts += S[idx].sum(axis=1)
        tc += C[idx].sum(axis=1)
    m = ts / tc
    return [sum(vals.values()) / len(vals), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--json", default=os.path.join(common.default_out(), "checks", "agent-unrounded.json"))
    args = ap.parse_args()
    out = []
    for bench, rep_name, pops in (("hint-full", "hint-full.report.json", ("all", "outside_hint-pilot", "no_leak")),
                                  ("issues", "issues.report.json", ("all", "without_pilot"))):
        rep = json.load(open(os.path.join(args.out, "cc-bench", rep_name)))
        H, Q = latest(args.out, bench), questions(args.out, bench)
        gf = {q: v["gold_files"] for q, v in Q.items()}
        pilot_bench = "hint-pilot" if bench == "hint-full" else "pilot"
        pilot = set()
        for r in REPOS:
            p = os.path.join(args.out, r, "cc-bench", pilot_bench, "questions.jsonl")
            if os.path.exists(p):
                pilot |= {f"{r}/{json.loads(l)['id']}" for l in open(p)}
        for pop in pops:
            for scope in ["pooled"] + REPOS:
                blk = rep["pooled"][pop] if scope == "pooled" else rep["per_repo"][scope][pop]
                arms = blk["arms"]
                base = [q for q in Q if scope == "pooled" or q.startswith(scope + "/")]
                if pop in ("outside_hint-pilot", "without_pilot"):
                    base = [q for q in base if q not in pilot]
                if pop == "no_leak":
                    base = [q for q in base if not Q[q].get("leak")]
                qids = sorted(q for q in base if all((q, a) in H for a in arms))
                for pair, v in blk["paired_diff"].items():
                    a, b = pair.rsplit("-", 1) if pair.endswith("-A") else (pair[:pair.index("-H-")], pair[pair.index("-H-") + 1:]) \
                        if "-H-" in pair else pair.split("-")
                    for m, stored in v.items():
                        if m not in OUTCOMES or not isinstance(stored, list) or 0.0 not in stored[1:]:
                            continue
                        vals = {q: float(H[(q, a)].get(m) or 0) - float(H[(q, b)].get(m) or 0) for q in qids}
                        un = boot(vals, qids, gf)
                        out.append({"bench": bench, "population": pop, "scope": scope, "pair": pair, "metric": m,
                                    "n": len(qids), "n_stored": blk["questions"], "stored": stored, "unrounded": un,
                                    "reproduces": [round(x, 4) for x in un] == stored and len(qids) == blk["questions"]})
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    bad = [c for c in out if not c["reproduces"]]
    print(f"{len(out)} cells with a stored 0.0 bound; {len(bad)} not reproduced")
    for c in out:
        print(f"  {c['bench']:9s} {c['population']:18s} {c['scope']:8s} {c['pair']:16s} {c['metric']:20s} stored {c['stored']} "
              f"unrounded lo {c['unrounded'][1]:+.8f} hi {c['unrounded'][2]:+.8f}{'' if c['reproduces'] else '  NOT REPRODUCED'}")


if __name__ == "__main__":
    main()
