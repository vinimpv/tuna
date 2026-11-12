"""Score rank files against a questions jsonl: hit@1/3/5/10/20, MRR, MRR@10, recall@200.

A question counts as hit@k when any of its gold fqns is in the top k. MRR uses
the best-ranked gold within the rank depth (0 when absent). Gold fqns that are
not in symbols.jsonl are reported, because they can never be retrieved.
"""
import json
import os

from . import common, metrics

KS = (1, 3, 5, 10, 20)


def gold_ranks(rows, golds):
    """Per question: 1-based rank of the best-ranked gold within the rank depth, or 0."""
    out = {}
    for qid, gold in golds.items():
        ranked = rows.get(qid, {}).get("ranked", [])
        out[qid] = next((i + 1 for i, f in enumerate(ranked) if f in gold), 0)
    return out


def score(rows, golds):
    n = len(golds)
    out = {f"hit@{k}": 0 for k in KS}
    rr_sum, rr10_sum, found = 0.0, 0.0, 0
    for qid, gold in golds.items():
        ranked = rows.get(qid, {}).get("ranked", [])
        best = next((i + 1 for i, f in enumerate(ranked) if f in gold), 0)
        if best:
            found += 1
            rr_sum += 1.0 / best
            rr10_sum += 1.0 / best if best <= 10 else 0.0
        for k in KS:
            out[f"hit@{k}"] += int(0 < best <= k)
    res = {k: round(v / n, 4) for k, v in out.items()}
    res["mrr"] = round(rr_sum / n, 4)
    res["mrr@10"] = round(rr10_sum / n, 4)
    res[f"recall@{common.RANK_DEPTH}"] = round(found / n, 4)
    res["n"] = n
    return res


def _main(args, stage):
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    known = {s["fqn"] for s in common.load_symbols(work)}
    qs = common.load_questions(args.questions)
    golds = {q["id"]: set(q["gold_fqns"]) for q in qs}
    missing = {q["id"]: sorted(set(q["gold_fqns"]) - known) for q in qs if set(q["gold_fqns"]) - known}
    unreachable = [qid for qid, g in golds.items() if not (g & known)]
    if missing:
        print(f"[eval] WARNING {len(missing)} questions name gold fqns missing from symbols.jsonl; "
              f"{len(unreachable)} have no reachable gold at all", flush=True)
        for qid, m in list(missing.items())[:10]:
            print(f"  {qid}: {m}", flush=True)
    names = args.ranks or sorted(f[:-6] for f in os.listdir(os.path.join(work, "ranks")) if f.endswith(".jsonl"))
    table, per_question = {}, {}
    for name in names:
        rows = {r["id"]: r for r in common.read_jsonl(os.path.join(work, "ranks", f"{name}.jsonl"))}
        table[name] = score(rows, golds)
        per_question[name] = gold_ranks(rows, golds)
    cols = [f"hit@{k}" for k in KS] + ["mrr", "mrr@10", f"recall@{common.RANK_DEPTH}"]
    width = max(len(n) for n in names)
    print(f"{'ranking':<{width}}  " + "  ".join(f"{c:>9}" for c in cols) + f"   (n={len(qs)})")
    for name, res in table.items():
        print(f"{name:<{width}}  " + "  ".join(f"{res[c]:>9.3f}" for c in cols))
    summary = {"repo": cfg["name"], "questions": metrics.portable(os.path.abspath(args.questions)), "n": len(qs),
               "unreachable_gold": unreachable, "missing_gold": missing, "results": table,
               "per_question_gold_rank": per_question}
    out = args.save or os.path.join(work, "eval_" + os.path.splitext(os.path.basename(args.questions))[0] + ".json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    stage.config(questions_set=os.path.basename(args.questions), rankings=names)
    stage.counts(items=len(qs), questions=len(qs), rankings=len(names))
    stage.extra(results=table)
    stage.outputs(out)
    print(f"[eval] wrote {out}")


def main(args):
    with metrics.Stage('eval', args.repo, None, metrics.config_of(args)) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("eval", help="hit@k and MRR for rank files")
    p.add_argument("--questions", required=True)
    p.add_argument("--ranks", nargs="*", help="rank file names under ranks/ (default: all)")
    p.add_argument("--save", help="summary json path")
    p.set_defaults(func=main)
    return p
