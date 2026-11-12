"""Preference test (design in the ledger at 2026-11-12T13:05:08.210Z).

  .venv/bin/python scripts/pref-test.py [--json OUT]

For each target repo T, the claude-m3 questions of the other two repos are ranked over T's index by a separate job not
in this repository (ranks/pref__<src>__<row>.jsonl). Over all those queries, count the top-k slots filled by T's claude-m3 golds (G) and by
the other symbols of T's held-out files (O); preference ratio = (slots_G / |G|) / (slots_O / |O|). Report it for stock,
tuned here and tuned elsewhere, and tuned/stock with a 95% bootstrap interval over queries; k = 10 (primary) and 100.
Secondary: qwen-train1's golds (Gt) against the other symbols of the training files (Ot), same queries.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

OUT = common.default_out()
MODELS = {"LFM2.5": "lfm25", "jina-0.5B": "jina-0.5b", "Qwen3-0.6B": "qwen3-0.6b"}
STOCK = {"lfm25": "lfm25-emb", "jina-0.5b": "jina-0.5b", "qwen3-0.6b": "qwen3-0.6b"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(OUT, "tables", "pref-test.json"))
    ap.add_argument("--samples", type=int, default=10000)
    args = ap.parse_args()
    out = []
    for T in ["saleor", "zulip", "posthog"]:
        q = os.path.join(OUT, T, "questions")
        sym_file = {json.loads(l)["fqn"]: json.loads(l)["file_path"] for l in open(os.path.join(OUT, T, "symbols.jsonl"))}
        held = {l.strip() for l in open(os.path.join(q, "heldout-files.txt")) if l.strip()}
        G = {json.loads(l)["gold_fqn"] for l in open(os.path.join(q, "claude-m3.jsonl"))}
        O = {f for f, p in sym_file.items() if p in held} - G
        Gt = {json.loads(l)["gold_fqn"] for l in open(os.path.join(q, "qwen-train1.jsonl"))}
        Ot = {f for f, p in sym_file.items() if p not in held} - Gt
        rd = os.path.join(OUT, T, "ranks")
        srcs = [s for s in ["saleor", "zulip", "posthog"] if s != T]
        for model, pre in MODELS.items():
            rows = {"stock": STOCK[pre], "tuned here": f"{pre}-ft-lora"}
            for o in srcs:
                rows[f"tuned on {o}"] = f"{pre}-ft-lora.from-{o}"
            per = {}
            for label, row in rows.items():
                lists = []
                for s in srcs:
                    p = os.path.join(rd, f"pref__{s}__{row}.jsonl")
                    if not os.path.exists(p):
                        lists = None
                        break
                    lists += [json.loads(l)["ranked"] for l in open(p)]
                if lists:
                    per[label] = lists
            if "stock" not in per:
                out.append({"target": T, "model": model, "missing": True})
                continue
            for k in (10, 100):
                for grp, (A, B) in (("claude-m3 golds vs other held-out symbols", (G, O)),
                                    ("qwen-train1 golds vs other training symbols", (Gt, Ot))):
                    def counts(lists):
                        return [(sum(1 for x in r[:k] if x in A), sum(1 for x in r[:k] if x in B)) for r in lists]
                    def ratio(c):
                        a, b = sum(x for x, _ in c), sum(y for _, y in c)
                        return (a / len(A)) / (b / len(B)) if b else float("inf")
                    cs = {lab: counts(lists) for lab, lists in per.items()}
                    res = {"target": T, "model": model, "k": k, "groups": grp, "n_queries": len(per["stock"]),
                           "group_sizes": [len(A), len(B)], "ratio": {lab: round(ratio(c), 3) for lab, c in cs.items()}}
                    rng = random.Random(0)
                    n = len(per["stock"])
                    for lab in cs:
                        if lab == "stock":
                            continue
                        bs = []
                        for _ in range(args.samples):
                            idx = [rng.randrange(n) for _ in range(n)]
                            st = ratio([cs["stock"][i] for i in idx])
                            tu = ratio([cs[lab][i] for i in idx])
                            bs.append(tu / st if st else float("inf"))
                        bs.sort()
                        res.setdefault("tuned_over_stock", {})[lab] = {
                            "point": round(ratio(cs[lab]) / ratio(cs["stock"]), 3),
                            "ci95": [round(bs[int(0.025 * args.samples)], 3), round(bs[int(0.975 * args.samples) - 1], 3)]}
                    out.append(res)
    for r in out:
        print(json.dumps(r))
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
