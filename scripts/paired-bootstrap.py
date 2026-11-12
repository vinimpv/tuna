"""Paired bootstrap intervals for MRR@10 differences between rankings.

  .venv/bin/python scripts/paired-bootstrap.py [--repos saleor zulip posthog] [--sets qwen-m2 ...] [--json OUT]

Reads <out>/<repo>/eval_<set>.json (tuna eval keeps per-question gold ranks),
computes each ranking's MRR@10, and for each pair below the difference with a
95% paired bootstrap interval: the questions are resampled with replacement
(--samples times, fixed --seed) and both rankings are scored on the same
resample. "sig" marks intervals that exclude zero.

Pairs: tuned vs stock for each model (and LoRA vs stock, LoRA vs full for
LFM2.5), and between tuned models.

--metric file_hit10 scores file-level hit@10 per question instead (ranked symbols
mapped to files, duplicates dropped in rank order; read from ranks/<row>_<set>.jsonl).

--pool-repos adds one interval per set and pair pooled over the repos that have
both rows: the statistic is the mean difference over all their questions, and each
bootstrap sample resamples gold files within each repo (stratified by repo,
clustered by file; implies --cluster-by-file).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

PAIRS = [
    ("lfm25-ft-full", "lfm25-emb"), ("lfm25-ft-lora", "lfm25-emb"), ("lfm25-ft-lora", "lfm25-ft-full"),
    ("jina-0.5b-ft-full", "jina-0.5b"), ("qwen3-0.6b-ft-full", "qwen3-0.6b"),
    ("lfm25-ft-full", "jina-0.5b-ft-full"), ("lfm25-ft-full", "qwen3-0.6b-ft-full"),
    ("qwen3-0.6b-ft-full", "jina-0.5b-ft-full"), ("lfm25-ft-full", "bm25"),
    ("jina-0.5b-ft-lora", "jina-0.5b"), ("qwen3-0.6b-ft-lora", "qwen3-0.6b"),
    ("lfm25-ft-lora", "jina-0.5b-ft-lora"), ("lfm25-ft-lora", "qwen3-0.6b-ft-lora"),
    ("qwen3-0.6b-ft-lora", "jina-0.5b-ft-lora"),
    ("lfm25-ft-lora-bug", "lfm25-ft-lora"), ("lfm25-ft-lora-remine", "lfm25-ft-lora"),
    ("jina-0.5b-ft-lora", "jina-0.5b-ft-full"), ("qwen3-0.6b-ft-lora", "qwen3-0.6b-ft-full"),
    # the locked LFM variant against each stock model
    ("lfm25-ft-lora", "qwen3-0.6b"), ("lfm25-ft-lora", "jina-0.5b"), ("lfm25-ft-lora", "lfm25-colbert"),
    # 512-token rescore of the stock models against their full-length rows
    ("jina-0.5b-512", "jina-0.5b"), ("jina-1.5b-512", "jina-1.5b"), ("qwen3-0.6b-512", "qwen3-0.6b"),
    # tuned small model against the larger stock model
    ("lfm25-ft-lora", "jina-1.5b"), ("lfm25-ft-full", "jina-1.5b"),
]


def rr10(rank):
    return 1.0 / rank if 0 < rank <= 10 else 0.0


_FILE_HITS = {}


def file_hits(out, repo, rank_name, file_of, gold_files):
    """Per question 1.0 when a gold file is among the first 10 distinct files of the ranking."""
    key = (repo, rank_name)
    if key not in _FILE_HITS:
        hits = {}
        for l in open(os.path.join(out, repo, "ranks", f"{rank_name}.jsonl")):
            r = json.loads(l)
            files, seen = [], set()
            for f in r["ranked"]:
                p = file_of.get(f)
                if p not in seen:
                    seen.add(p)
                    files.append(p)
                    if len(files) == 10:
                        break
            hits[r["id"]] = 1.0 if gold_files.get(r["id"], set()) & set(files) else 0.0
        _FILE_HITS[key] = hits
    return _FILE_HITS[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--sets", nargs="+", default=["qwen-m2", "claude-m2", "issues-v1", "issues-v1-title",
                                                    "issues-v1-noleak"])
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--cluster-by-file", action="store_true",
                    help="resample gold files (with all their questions) instead of single questions: for sets with "
                         "several questions per file, such as claude-m3")
    ap.add_argument("--pairs", nargs="+", metavar="A:B", help="score these pairs instead of the built-in list")
    ap.add_argument("--metric", choices=["mrr10", "file_hit10"], default="mrr10")
    ap.add_argument("--pool-repos", action="store_true",
                    help="also pool the repos per set and pair (stratified by repo, resampled by gold file)")
    args = ap.parse_args()
    if args.pool_repos:
        args.cluster_by_file = True
    pairs = [tuple(p.split(":", 1)) for p in args.pairs] if args.pairs else PAIRS
    rows = []
    pooled = {}  # (set, a, b) -> list of (repo, da, db, groups)
    for repo in args.repos:
        file_of_all = None
        for s in args.sets:
            path = os.path.join(args.out, repo, f"eval_{s}.json")
            if not os.path.exists(path):
                continue
            ev = json.load(open(path))
            per = ev.get("per_question_gold_rank")
            if not per:
                continue
            names = {k.rsplit(f"_{s}", 1)[0]: k for k in per}  # ranking names are <row>_<set>
            cluster_of = None
            if args.cluster_by_file:
                file_of = {}
                for l in open(os.path.join(args.out, repo, "symbols.jsonl")):
                    sym = json.loads(l)
                    file_of[sym["fqn"]] = sym["file_path"]
                cluster_of = {}
                for l in open(os.path.join(args.out, repo, "questions", f"{s}.jsonl")):
                    q = json.loads(l)
                    g = (q.get("gold_fqns") or [q.get("gold_fqn")])[0]
                    cluster_of[q["id"]] = q.get("file") or file_of.get(g, g)
            if args.metric == "file_hit10":
                if file_of_all is None:
                    file_of_all = {}
                    for l in open(os.path.join(args.out, repo, "symbols.jsonl")):
                        sym = json.loads(l)
                        file_of_all[sym["fqn"]] = sym["file_path"]
                gold_files = {}
                for l in open(os.path.join(args.out, repo, "questions", f"{s}.jsonl")):
                    q = json.loads(l)
                    gold_files[q["id"]] = {file_of_all.get(g) for g in (q.get("gold_fqns") or [q["gold_fqn"]])}
            for a, b in pairs:
                if a not in names or b not in names:
                    continue
                qa, qb = per[names[a]], per[names[b]]
                qids = sorted(qa)
                if args.metric == "file_hit10":
                    fa, fb = (file_hits(args.out, repo, names[x], file_of_all, gold_files) for x in (a, b))
                    da = [fa.get(q, 0.0) for q in qids]
                    db = [fb.get(q, 0.0) for q in qids]
                else:
                    da = [rr10(qa[q]) for q in qids]
                    db = [rr10(qb[q]) for q in qids]
                n = len(qids)
                diff = sum(x - y for x, y in zip(da, db)) / n
                rng = random.Random(args.seed)
                boots = []
                if cluster_of:
                    groups = {}
                    for i, q in enumerate(qids):
                        groups.setdefault(cluster_of.get(q, q), []).append(i)
                    glist = list(groups.values())
                    pooled.setdefault((s, a, b), []).append((repo, da, db, glist))
                    for _ in range(args.samples):
                        idx = [i for _ in glist for i in glist[rng.randrange(len(glist))]]
                        boots.append(sum(da[i] - db[i] for i in idx) / len(idx))
                else:
                    for _ in range(args.samples):
                        idx = [rng.randrange(n) for _ in range(n)]
                        boots.append(sum(da[i] - db[i] for i in idx) / n)
                boots.sort()
                lo, hi = boots[int(0.025 * args.samples)], boots[int(0.975 * args.samples) - 1]
                rows.append({"repo": repo, "set": s, "a": a, "b": b, "n": n, "metric": args.metric,
                             "cluster_by_file": bool(cluster_of), "mrr10_a": round(sum(da) / n, 4),
                             "mrr10_b": round(sum(db) / n, 4), "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
                             "sig": lo > 0 or hi < 0})
    if args.pool_repos:
        for (s, a, b), parts in pooled.items():
            if len(parts) < 2:
                continue
            da = [x for _, d, _, _ in parts for x in d]
            db = [x for _, _, d, _ in parts for x in d]
            n = len(da)
            rng = random.Random(args.seed)
            boots = []
            for _ in range(args.samples):
                tot, cnt = 0.0, 0
                for _, pa, pb, glist in parts:  # stratified: each repo resamples its own gold files
                    for _ in glist:
                        for i in glist[rng.randrange(len(glist))]:
                            tot += pa[i] - pb[i]
                            cnt += 1
                boots.append(tot / cnt)
            boots.sort()
            lo, hi = boots[int(0.025 * args.samples)], boots[int(0.975 * args.samples) - 1]
            rows.append({"repo": "+".join(r for r, *_ in parts), "set": s, "a": a, "b": b, "n": n, "metric": args.metric,
                         "cluster_by_file": True, "pooled": True, "mrr10_a": round(sum(da) / n, 4),
                         "mrr10_b": round(sum(db) / n, 4), "diff": round((sum(da) - sum(db)) / n, 4),
                         "ci95": [round(lo, 4), round(hi, 4)], "sig": lo > 0 or hi < 0})
    label = "MRR@10" if args.metric == "mrr10" else "fhit@10"
    print(f"{'repo':20s} {'set':17s} {'a':26s} {'b':20s} {'n':>4s} {label + ' a':>9s} {'b':>7s} {'diff':>7s}  95% CI")
    for r in rows:
        print(f"{r['repo']:20s} {r['set']:17s} {r['a']:26s} {r['b']:20s} {r['n']:4d} {r['mrr10_a']:9.3f} {r['mrr10_b']:7.3f} "
              f"{r['diff']:+7.3f}  [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{'  sig' if r['sig'] else ''}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"samples": args.samples, "seed": args.seed, "metric": args.metric, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
