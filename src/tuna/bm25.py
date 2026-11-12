"""BM25 (Okapi, k1=1.5, b=0.75) baseline over symbol documents, pure stdlib.

Code-aware tokens: identifiers are kept whole and also split on snake_case and
camelCase, all lowercased; a small English stopword list is dropped.
"""
import collections
import math
import os
import re

from . import common, metrics

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP = frozenset("""a an and are as at be by can do does for from has have how i if in into is it its
of on or that the their then there these this to was what when where which who why will with
self cls def return none true false""".split())


def tokenize(text):
    out = []
    for w in WORD.findall(text):
        lw = w.lower()
        parts = [p.lower() for piece in w.split("_") for p in CAMEL.findall(piece)]
        if lw not in STOP:
            out.append(lw)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOP and len(p) > 1)
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = collections.defaultdict(list)
        self.lengths = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.lengths.append(len(toks))
            for t, tf in collections.Counter(toks).items():
                self.postings[t].append((i, tf))
        n = len(docs)
        self.avgdl = sum(self.lengths) / max(n, 1)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def top(self, query, k):
        scores = collections.defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def _main(args, stage):
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    qs = common.load_questions(args.questions)
    docs = [common.symbol_doc(s, args.cap) for s in syms]
    index = BM25(docs, k1=args.k1, b=args.b)
    print(f"[bm25] {len(docs)} docs, {len(index.postings)} terms, avgdl {index.avgdl:.0f}", flush=True)
    rows = []
    for q in qs:
        top = index.top(q["question"], common.RANK_DEPTH)
        rows.append(common.rank_row(q, [syms[i]["fqn"] for i, _ in top], [s for _, s in top]))
    out = os.path.join(work, "ranks", f"{args.name or 'bm25'}.jsonl")
    common.write_jsonl(out, rows)
    stage.counts(items=len(qs), questions=len(qs), symbols=len(syms), terms=len(index.postings))
    stage.outputs(out)
    print(f"[bm25] wrote {out}", flush=True)


def main(args):
    with metrics.Stage('bm25', args.repo, "bm25", metrics.config_of(args)) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("bm25", help="BM25 baseline ranks")
    p.add_argument("--questions", required=True)
    p.add_argument("--cap", type=int, default=0, help="char cap on body documents (0 = none)")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--name", help="rank file name (default bm25)")
    p.set_defaults(func=main)
    return p
