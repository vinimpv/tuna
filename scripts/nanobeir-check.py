"""Reproduce LFM2.5-Embedding-350M's English NanoBEIR NDCG@10 (model card) through tuna's encoder path.

  .venv/bin/python scripts/nanobeir-check.py [--model LiquidAI/LFM2.5-Embedding-350M] [--data $TUNA_DATA/nanobeir/en]

Data: scripts/nanobeir-fetch.py (LiquidAI/nanobeir-multilingual-extended, English).
Encoding goes through tuna.encoders.LocalEncoder exactly as the code-search runs do
(pinned revision, "query: " / "document: " prefixes from the model, CLS pooling,
512 tokens; documents are passed whole and truncated by the model, as on the
card). For each of the 13 subsets: cosine ranking of the whole corpus, NDCG@10
against the graded qrels; the card's "en" column is the mean over subsets.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tuna import common, encoders, metrics  # noqa: E402

CARD = {"LiquidAI/LFM2.5-Embedding-350M": 0.644}


def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def doc_text(row):
    title, text = (row.get("title") or "").strip(), (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-Embedding-350M")
    ap.add_argument("--data", default=os.path.join(common.data_root(), "nanobeir", "en"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json")
    args = ap.parse_args()
    import numpy as np
    subsets = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    out = {"model": args.model, "data": args.data, "subsets": {}}
    with metrics.Stage("nanobeir-check", None, args.model, vars(args)) as stage:
        enc = encoders.LocalEncoder(args.model, device=args.device)
        out.update(prefix=enc.prefix, query_prefix=enc.query_prefix, max_seq_length=enc.limit, revision=enc.revision)
        docs_total = 0
        for sub in subsets:
            d = os.path.join(args.data, sub)
            corpus, queries, qrels = load(f"{d}/corpus.jsonl"), load(f"{d}/queries.jsonl"), load(f"{d}/qrels.jsonl")
            ids = [str(r.get("_id", r.get("id"))) for r in corpus]
            t0 = time.time()
            D = enc.encode_documents([doc_text(r) for r in corpus])
            rel = {}
            for r in qrels:
                qid = str(r.get("query-id", r.get("query_id", r.get("qid"))))
                did = str(r.get("corpus-id", r.get("corpus_id", r.get("doc_id"))))
                rel.setdefault(qid, {})[did] = int(r.get("score", 1))
            qs = [r for r in queries if str(r.get("_id", r.get("id"))) in rel]
            Q = enc.encode_queries([r["text"] for r in qs])
            S = Q @ D.T
            scores = []
            for qi, q in enumerate(qs):
                top = np.argsort(-S[qi])[:10]
                scores.append(ndcg10([ids[j] for j in top], rel[str(q.get("_id", q.get("id")))]))
            out["subsets"][sub] = {"ndcg@10": round(float(np.mean(scores)), 4), "queries": len(qs),
                                   "documents": len(corpus), "seconds": round(time.time() - t0, 1)}
            docs_total += len(corpus)
            print(sub, out["subsets"][sub], flush=True)
        out["ndcg@10_mean"] = round(float(np.mean([v["ndcg@10"] for v in out["subsets"].values()])), 4)
        out["card_en"] = CARD.get(args.model)
        stage.counts(items=docs_total, subsets=len(subsets), documents=docs_total)
        stage.extra(**{k: v for k, v in out.items() if k != "model"})
    print(json.dumps({k: out[k] for k in ("model", "ndcg@10_mean", "card_en", "prefix", "query_prefix",
                                          "max_seq_length", "revision")}, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
