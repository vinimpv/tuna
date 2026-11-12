"""Gains of the locked LoRA adapters over stock on the transfer-only test repos.

  .venv/bin/python scripts/testrepo-gains.py [--repos dispatch homeassistant] [--set claude-t400] [--json OUT]

For each model and test repo, on --set: MRR@10 and file-level hit@10 of the stock row, of each adapter
(<row>.from-<training repo>), of the mean of the three adapters (per question, the mean of their
scores) and of the pooled model if scored (<row>.pooled). Each adapter, the three-adapter mean and the
pooled row get a paired 95% bootstrap interval against stock, gold files resampled (per repo, and
pooled over the test repos stratified by repo).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

MODELS = {"LFM2.5-350M": ("lfm25-emb", "lfm25-ft-lora", ["lfm25-pooled-lora", "lfm25-pooled-full"]),
          "jina-0.5B": ("jina-0.5b", "jina-0.5b-ft-lora", ["jina-0.5b-pooled-lora"]),
          "Qwen3-0.6B": ("qwen3-0.6b", "qwen3-0.6b-ft-lora", ["qwen3-0.6b-pooled-lora"])}
SRC = ["saleor", "zulip", "posthog"]


def interval(parts, samples, seed):
    rng = random.Random(seed)
    boots = []
    for _ in range(samples):
        tot = cnt = 0
        for v, gl in parts:
            for _ in gl:
                for i in gl[rng.randrange(len(gl))]:
                    tot += v[i]
                    cnt += 1
        boots.append(tot / cnt)
    boots.sort()
    return [round(boots[int(0.025 * samples)], 4), round(boots[int(0.975 * samples) - 1], 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["dispatch", "homeassistant"])
    ap.add_argument("--set", default="claude-t400")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    data = {}
    for repo in args.repos:
        w = os.path.join(args.out, repo)
        if not os.path.exists(os.path.join(w, "ranks", f"bm25_{args.set}.jsonl")):
            continue
        file_of = {}
        for l in open(os.path.join(w, "symbols.jsonl")):
            s = json.loads(l)
            file_of[s["fqn"]] = s["file_path"]
        qs = [json.loads(l) for l in open(os.path.join(w, "questions", f"{args.set}.jsonl"))]
        golds = {q["id"]: set(q.get("gold_fqns") or [q["gold_fqn"]]) for q in qs}
        cluster = {q["id"]: q.get("file") or file_of.get(q["gold_fqn"]) for q in qs}
        qids = sorted(golds)
        groups = {}
        for i, q in enumerate(qids):
            groups.setdefault(cluster[q], []).append(i)
        rows = {}
        for fn in os.listdir(os.path.join(w, "ranks")):
            if not fn.endswith(f"_{args.set}.jsonl"):
                continue
            name = fn[:-len(f"_{args.set}.jsonl")]
            ranked = {json.loads(l)["id"]: json.loads(l)["ranked"] for l in open(os.path.join(w, "ranks", fn))}
            mrr, fh = [], []
            for q in qids:
                lst = ranked.get(q, [])
                best = next((i + 1 for i, f in enumerate(lst[:10]) if f in golds[q]), 0)
                mrr.append(1.0 / best if best else 0.0)
                files, seen = [], set()
                for f in lst:
                    p = file_of.get(f)
                    if p not in seen:
                        seen.add(p)
                        files.append(p)
                        if len(files) == 10:
                            break
                gf = {file_of.get(g) for g in golds[q]}
                fh.append(1.0 if gf & set(files) else 0.0)
            rows[name] = {"mrr10": mrr, "file_hit10": fh}
        data[repo] = (rows, list(groups.values()), len(qids))
    out = []
    for repo, (rows, gl, n) in data.items():
        for name in ("bm25", "lfm25-colbert", "jina-1.5b"):
            if name in rows:
                out.append({"repo": repo, "model": None, "row": name, "n": n,
                            **{m: round(sum(rows[name][m]) / n, 4) for m in ("mrr10", "file_hit10")}})
    for model, (stock, lora, pooled) in MODELS.items():
        for metric in ("mrr10", "file_hit10"):
            cand = {f"{lora}.from-{s}": f"adapter from {s}" for s in SRC}
            cand.update({f"{p}.pooled": "pooled" for p in pooled})
            pooled_parts = {}
            for repo, (rows, gl, n) in data.items():
                if stock not in rows:
                    continue
                st = rows[stock][metric]
                out.append({"repo": repo, "model": model, "metric": metric, "row": stock, "label": "stock", "n": n,
                            "value": round(sum(st) / n, 4)})
                ad = [rows[k][metric] for k in cand if k.startswith(lora) and k in rows]
                series = {k: rows[k][metric] for k in cand if k in rows}
                if len(ad) == 3:
                    series["mean of the three adapters"] = [sum(x) / 3 for x in zip(*ad)]
                for k, v in series.items():
                    d = [a - b for a, b in zip(v, st)]
                    lo_hi = interval([(d, gl)], args.samples, args.seed)
                    out.append({"repo": repo, "model": model, "metric": metric, "row": k,
                                "label": cand.get(k, k), "n": n, "value": round(sum(v) / n, 4),
                                "gain": round(sum(d) / n, 4), "ci95": lo_hi, "sig": lo_hi[0] > 0 or lo_hi[1] < 0})
                    pooled_parts.setdefault(k, []).append((d, gl))
            for k, parts in pooled_parts.items():
                if len(parts) < 2:
                    continue
                d = [x for v, _ in parts for x in v]
                lo_hi = interval(parts, args.samples, args.seed)
                out.append({"repo": "+".join(data), "model": model, "metric": metric, "row": k, "pooled": True,
                            "n": len(d), "gain": round(sum(d) / len(d), 4), "ci95": lo_hi,
                            "sig": lo_hi[0] > 0 or lo_hi[1] < 0})
    for r in out:
        if "gain" in r:
            print(f"{r['repo']:22s} {r.get('metric',''):10s} {r['model'] or '':12s} {r['row']:34s} n={r['n']:4d} "
                  f"{r.get('value', float('nan')):.3f} gain {r['gain']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
                  f"{'  sig' if r['sig'] else ''}")
        else:
            print(f"{r['repo']:22s} {r.get('metric',''):10s} {r['model'] or '':12s} {r['row']:34s} n={r['n']:4d} "
                  + (f"{r['value']:.3f}" if 'value' in r else f"MRR {r['mrr10']:.3f} file hit {r['file_hit10']:.3f}"))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
