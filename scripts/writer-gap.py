"""The same gold symbols asked by two writers: claude-m3 against qwen-m4-400 (CPU only, from the eval files).

  python3 scripts/writer-gap.py [--json $TUNA_OUT/tables/writer-gap.json]

qwen-m4-400 rewrites the gold symbols of claude-m3, so the two sets pair by gold symbol. Per model, stock and tuned
(LoRA): MRR@10 on Claude's question minus MRR@10 on Qwen's question over the shared golds, with a 95% paired bootstrap
over golds (10,000 resamples, seed 0), per repository and pooled (stratified by repository).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tuna import common  # noqa: E402

OUT = common.default_out()


def jl(p):
    return [json.loads(l) for l in open(p)]


def writer_gap(samples=10000, seed=0):
    """See the module docstring."""
    import random
    rows = {"LFM2.5": ("lfm25-emb", "lfm25-ft-lora"), "jina-0.5B": ("jina-0.5b", "jina-0.5b-ft-lora"),
            "Qwen3-0.6B": ("qwen3-0.6b", "qwen3-0.6b-ft-lora")}
    def rr(r):
        return 1.0 / r if 0 < r <= 10 else 0.0
    data, missing = {}, []
    for repo in ["saleor", "zulip", "posthog"]:
        pc = os.path.join(OUT, repo, "eval_claude-m3.json")
        pm = os.path.join(OUT, repo, "eval_qwen-m4-400.json")
        if not (os.path.exists(pc) and os.path.exists(pm)):
            missing.append(repo)
            continue
        ec, em = json.load(open(pc))["per_question_gold_rank"], json.load(open(pm))["per_question_gold_rank"]
        qc = {x["id"]: x["gold_fqn"] for x in jl(os.path.join(OUT, repo, "questions", "claude-m3.jsonl"))}
        qm = {x["id"]: x["gold_fqn"] for x in jl(os.path.join(OUT, repo, "questions", "qwen-m4-400.jsonl"))}
        gc, gm = {g: i for i, g in qc.items()}, {g: i for i, g in qm.items()}
        shared = sorted(set(gc) & set(gm))
        data[repo] = (ec, em, gc, gm, shared)
    def series(repo, rc, rm):
        ec, em, gc, gm, shared = data[repo]
        a, b = ec.get(f"{rc}_claude-m3"), em.get(f"{rm}_qwen-m4-400")
        if a is None or b is None:
            return None
        return [rr(a[gc[g]]) - rr(b[gm[g]]) for g in shared]
    def boot(parts):
        rng = random.Random(seed)
        bs = []
        for _ in range(samples):
            t = n = 0
            for v in parts:
                for _ in v:
                    t += v[rng.randrange(len(v))]
                    n += 1
            bs.append(t / n)
        bs.sort()
        return [round(bs[int(0.025 * samples)], 4), round(bs[int(0.975 * samples) - 1], 4)]
    out = {"missing_scores": missing, "writer_gap": {}}
    for model, (st, tu) in rows.items():
        for label, row in (("stock", st), ("tuned", tu)):
            parts = {r: series(r, row, row) for r in data}
            parts = {r: v for r, v in parts.items() if v}
            cell = {r: {"n": len(v), "claude_minus_qwen": round(sum(v) / len(v), 4), "ci95": boot([v])} for r, v in parts.items()}
            if len(parts) > 1:
                allv = [x for v in parts.values() for x in v]
                cell["pooled"] = {"n": len(allv), "claude_minus_qwen": round(sum(allv) / len(allv), 4), "ci95": boot(list(parts.values()))}
            out["writer_gap"][f"{model} {label}"] = cell
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(OUT, "tables", "writer-gap.json"))
    args = ap.parse_args()
    out = writer_gap()
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
