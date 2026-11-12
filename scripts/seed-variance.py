"""Seed-level cells from saved scores (CPU only; definitions in the ledger note of 2026-11-12T15:35:53.811Z).

  python3 scripts/seed-variance.py [--json $TUNA_OUT/tables/seed-variance.json]

seed_level: on Saleor's claude-m3 questions, the mean over the three LoRA seeds of tuned LFM2.5 minus stock Qwen3-0.6B
and minus stock jina-code-1.5B, with a 95% bootstrap interval over gold files (10,000 resamples, seed 0), and each
seed's own difference and interval. here_elsewhere_by_seed: MRR@10 on claude-m3 of Saleor's own adapter (seeds 0-2)
and of the adapters tuned on Zulip and PostHog, and of Saleor's adapters scored on Zulip and PostHog. seed_ranges: the
range of MRR@10 over the seed runs and the same-seed reruns on a second card, per row and test set. vs_qwen3_4b: tuned
Jina and tuned Qwen3 minus stock Qwen3-Embedding-4B on Saleor.

Reads eval_<set>.json, questions, symbols.jsonl and ranks/<row>_<set>.jsonl under $TUNA_OUT. The rank files are not in
the release; scripts/m2-score.sh, m2-seeds.sh and m2-transfer.sh rebuild them.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tuna import common  # noqa: E402

OUT = common.default_out()


def jl(p):
    return [json.loads(l) for l in open(p)]


def rr(r):
    return 1.0 / r if 0 < r <= 10 else 0.0


def per(repo, st):
    return json.load(open(f"{OUT}/{repo}/eval_{st}.json"))["per_question_gold_rank"]


def mrr_rank_file(repo, name, st):
    golds = {q["id"]: set(q.get("gold_fqns") or [q["gold_fqn"]]) for q in jl(f"{OUT}/{repo}/questions/{st}.jsonl")}
    t = 0
    for r in jl(f"{OUT}/{repo}/ranks/{name}_{st}.jsonl"):
        b = next((i + 1 for i, x in enumerate(r["ranked"][:10]) if x in golds[r["id"]]), 0)
        t += 1 / b if b else 0
    return round(t / len(golds), 4)


def groups(repo, st, qids):
    f = {s["fqn"]: s["file_path"] for s in jl(f"{OUT}/{repo}/symbols.jsonl")}
    cl = {q["id"]: q.get("file") or f.get(q["gold_fqn"]) for q in jl(f"{OUT}/{repo}/questions/{st}.jsonl")}
    g = {}
    for i, q in enumerate(qids):
        g.setdefault(cl[q], []).append(i)
    return list(g.values())


def boot(d, gl, n=10000):
    rng = random.Random(0)
    bs = []
    for _ in range(n):
        t = c = 0
        for _ in gl:
            for i in gl[rng.randrange(len(gl))]:
                t += d[i]
                c += 1
        bs.append(t / c)
    bs.sort()
    return [bs[int(.025 * n)], bs[int(.975 * n) - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(OUT, "tables", "seed-variance.json"))
    args = ap.parse_args()
    out = {}
    p = per("saleor", "claude-m3")
    qids = sorted(p["lfm25-emb_claude-m3"])
    gl = groups("saleor", "claude-m3", qids)
    seeds = ["lfm25-ft-lora", "lfm25-ft-lora-s1", "lfm25-ft-lora-s2"]
    cell = {}
    for stock in ["qwen3-0.6b", "jina-1.5b"]:
        stock_rr = [rr(p[f"{stock}_claude-m3"][q]) for q in qids]
        mean = [sum(rr(p[f"{s}_claude-m3"][q]) for s in seeds) / 3 for q in qids]
        d = [a - b for a, b in zip(mean, stock_rr)]
        lo, hi = boot(d, gl)
        per_seed = []
        for s in seeds:
            ds = [rr(p[f"{s}_claude-m3"][q]) - b for q, b in zip(qids, stock_rr)]
            l2, h2 = boot(ds, gl)
            per_seed.append({"seed": s, "diff": round(sum(ds) / len(ds), 4), "ci95_unrounded": [l2, h2],
                             "inside_pm_0.05": -0.05 < l2 and h2 < 0.05})
        cell[stock] = {"seed_mean_diff": round(sum(d) / len(d), 4), "ci95": [round(lo, 4), round(hi, 4)], "per_seed": per_seed,
                       "n_inside": sum(x["inside_pm_0.05"] for x in per_seed)}
    out["seed_level"] = cell
    c = {}
    for row in ["lfm25-ft-lora", "jina-0.5b-ft-lora", "qwen3-0.6b-ft-lora"]:
        sal = {f"s{k}": mrr_rank_file("saleor", row if k == 0 else f"{row}-s{k}", "claude-m3") for k in (0, 1, 2)}
        sal_else = {o: mrr_rank_file("saleor", f"{row}.from-{o}", "claude-m3") for o in ("zulip", "posthog")}
        on = {}
        for r in ("zulip", "posthog"):
            on[r] = {"s0": mrr_rank_file(r, f"{row}.from-saleor", "claude-m3"),
                     "s1": mrr_rank_file(r, f"{row}-s1.from-saleor", "claude-m3"),
                     "s2": mrr_rank_file(r, f"{row}-s2.from-saleor", "claude-m3"),
                     "tuned_here_s0": mrr_rank_file(r, row, "claude-m3")}
        c[row] = {"saleor_own_by_seed": sal, "saleor_other_repo_adapters_seed0": sal_else, "saleor_adapters_on_other_repos": on}
    out["here_elsewhere_by_seed"] = c
    d = {}
    for row in ["lfm25-ft-lora", "jina-0.5b-ft-lora", "qwen3-0.6b-ft-lora"]:
        for st in ["claude-m3", "issues-v1", "qwen-m4-400"]:
            vals = {}
            for name in [row, f"{row}-s1", f"{row}-s2", f"{row}-4090"]:
                if os.path.exists(f"{OUT}/saleor/ranks/{name}_{st}.jsonl"):
                    vals[name] = mrr_rank_file("saleor", name, st)
            r4090 = [v for k, v in vals.items() if k != row]
            allv = list(vals.values())
            d[f"{row} {st}"] = {"values": vals, "range_all_runs": round(max(allv) - min(allv), 4),
                                "range_4090_runs": round(max(r4090) - min(r4090), 4) if len(r4090) > 1 else None}
    out["seed_ranges"] = d
    a = {}
    for st in ["claude-m3", "issues-v1"]:
        p = per("saleor", st)
        qids = sorted(p["qwen3-4b_" + st])
        gl = groups("saleor", st, qids)
        for row in ["jina-0.5b-ft-lora", "qwen3-0.6b-ft-lora"]:
            dd = [rr(p[f"{row}_{st}"][q]) - rr(p[f"qwen3-4b_{st}"][q]) for q in qids]
            lo, hi = boot(dd, gl)
            a[f"{row} - qwen3-4b {st}"] = {"diff": round(sum(dd) / len(dd), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["vs_qwen3_4b"] = a
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
