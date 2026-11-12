"""Doc2query control: rank with an index of written questions, alone and next to the code.

  .venv/bin/python scripts/d2q-index.py --repo saleor --rows lfm25-emb jina-0.5b qwen3-0.6b \
      [--both-only-rows lfm25-ft-lora ...] [--d2q index-d2q] [--sets claude-m3 issues-v1 ...]

The index-side question set (questions/<d2q>.jsonl, one question per symbol, never training data) is
encoded by each row's own encoder (rebuilt from vectors/<row>.json, so a tuned row loads its weights)
twice: with the query prompt (qp) and with the document prompt (dp). For every test set, three indexes
per row and prompt:
  code   the row as already scored (ranks/<row>_<set>.jsonl; not rewritten here)
  qonly  questions only: a symbol's score is its question's cosine; symbols without a question rank last
  both   max of the code score and the question score; a symbol without a question keeps its code score
Writes ranks/<row>.d2q-{qonly,both}-{qp,dp}_<set>.jsonl (rows in --both-only-rows: both only) and
<out>/<repo>/d2q-index.json with the fallback count (symbols without a question) and timings.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tuna import chunking, common, dense, encoders, metrics  # noqa: E402


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repo", required=True)
    ap.add_argument("--d2q", default="index-d2q")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--both-only-rows", nargs="*", default=[])
    ap.add_argument("--sets", nargs="+", default=["qwen-m2", "claude-m2", "claude-m3", "qwen-m4-400", "issues-v1",
                                                    "issues-v1-title", "issues-v1-noleak"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-questions", type=int, help="smoke test: first N test questions per set")
    args = ap.parse_args()
    if not args.d2q.startswith("index-"):
        raise SystemExit("the doc2query set must be an index-side set (index-*)")
    work = os.path.join(args.out, args.repo)
    syms = [json.loads(l) for l in open(os.path.join(work, "symbols.jsonl"))]
    idx = {s["fqn"]: i for i, s in enumerate(syms)}
    sha = common.file_sha256(os.path.join(work, "symbols.jsonl"))
    d2q = [json.loads(l) for l in open(os.path.join(work, "questions", f"{args.d2q}.jsonl"))]
    d2q = [q for q in d2q if q["gold_fqn"] in idx]
    q_owner = np.array([idx[q["gold_fqn"]] for q in d2q])
    covered = np.zeros(len(syms), bool)
    covered[q_owner] = True
    summary_path = os.path.join(work, "d2q-index.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"rows": {}}
    summary.update(d2q=args.d2q, symbols=len(syms), questions=len(d2q), symbols_with_question=int(covered.sum()),
                   fallback_symbols=int((~covered).sum()))
    tests = {}
    for st in args.sets:
        p = os.path.join(work, "questions", f"{st}.jsonl")
        if os.path.exists(p):
            qs = common.load_questions(p)
            tests[st] = qs[:args.limit_questions] if args.limit_questions else qs
    for row in args.rows + args.both_only_rows:
        kinds = ["both"] if row in args.both_only_rows else ["qonly", "both"]
        with metrics.Stage("d2q-index", args.repo, row, {"d2q": args.d2q, "kinds": kinds, "sets": list(tests)}) as stage:
            M, owner, meta = dense.load_matrix(work, row, sha, len(syms))
            enc = encoders.from_meta(meta, device=args.device)
            t0 = time.time()
            texts = [q["question"] for q in d2q]
            V = {"qp": enc.encode_queries(texts), "dp": enc.encode_documents(texts)}
            enc_s = time.time() - t0
            info = {"encode_seconds": round(enc_s, 1), "kinds": kinds, "sets": {}}
            for st, qs in tests.items():
                Q = enc.encode_queries([q["question"] for q in qs])
                code = chunking.max_per_symbol(Q @ M.T, owner, len(syms))
                for pk, D in V.items():
                    S = Q @ D.T
                    qbest_t = np.full((len(syms), len(qs)), -np.inf, dtype=np.float32)
                    np.maximum.at(qbest_t, q_owner, S.T)  # a symbol's best question (one per symbol here)
                    qbest = np.ascontiguousarray(qbest_t.T)
                    for kind in kinds:
                        sims = qbest if kind == "qonly" else np.maximum(code, qbest)
                        rows = []
                        for q, s in zip(qs, sims):
                            order = np.argsort(-s, kind="stable")[:common.RANK_DEPTH]
                            rows.append(common.rank_row(q, [syms[int(j)]["fqn"] for j in order],
                                                        [max(float(s[int(j)]), -9.0) for j in order]))
                        name = f"{row}.d2q-{kind}-{pk}_{st}"
                        common.write_jsonl(os.path.join(work, "ranks", f"{name}.jsonl"), rows)
                info["sets"][st] = len(qs)
            summary["rows"][row] = info
            stage.counts(items=len(d2q), questions=len(d2q), symbols=len(syms),
                         fallback_symbols=int((~covered).sum()))
            print(f"[d2q] {row}: {len(d2q)} index questions encoded twice in {enc_s:.0f}s; "
                  f"{int((~covered).sum())} symbols without a question; sets {list(tests)}", flush=True)
            del enc
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
