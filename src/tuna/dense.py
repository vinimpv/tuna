"""Dense ranking: encode the questions with the encoder the vectors were built
with, score every chunk by cosine, keep each symbol's best chunk, take the max
over one or more vector tags, and keep the top RANK_DEPTH.
"""
import json
import os

from . import chunking, common, encoders, metrics

# Encoder settings that must agree between vector tags ranked together.
SAME_ENCODER_KEYS = ("backend", "model", "revision", "tokenizer", "tokenizer_revision", "query_prefix")


def load_matrix(work, tag, symbols_sha, n_symbols):
    import numpy as np
    vdir = os.path.join(work, "vectors")
    with open(os.path.join(vdir, f"{tag}.json")) as fh:
        meta = json.load(fh)
    if meta["symbols_sha256"] != symbols_sha:
        raise SystemExit(f"vectors/{tag} was built from another symbols.jsonl; re-embed")
    M = np.load(os.path.join(vdir, f"{tag}.npy"))
    owner = np.load(os.path.join(vdir, f"{tag}_owner.npy"))
    if len(owner) != len(M) or meta["symbols"] != n_symbols:
        raise SystemExit(f"vectors/{tag}: matrix, owner and symbols disagree; re-embed")
    return M, owner, meta


def _main(args, stage):
    import numpy as np
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    sha = common.file_sha256(os.path.join(work, "symbols.jsonl"))
    qs = common.load_questions(args.questions)
    mats = [load_matrix(work, t, sha, len(syms)) for t in args.vectors]
    first = mats[0][2]
    for _, _, meta in mats[1:]:
        diff = [k for k in SAME_ENCODER_KEYS if meta.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"vectors/{meta['tag']} and vectors/{first['tag']} differ in {diff}")
    for _, _, meta in mats:
        st = meta["chunking"]
        print(f"[dense] vectors/{meta['tag']}: {st['split']} of {st['symbols']} symbols split at "
              f"{st['limit']} tokens into {meta['documents']} documents", flush=True)
    enc = encoders.from_meta(first, device=args.device)
    Q = enc.encode_queries([q["question"] for q in qs])
    sims = None
    for M, owner, _ in mats:
        s = chunking.max_per_symbol(Q @ M.T, owner, len(syms))
        sims = s if sims is None else np.maximum(sims, s)
    rows = []
    for q, s in zip(qs, sims):
        order = np.argsort(-s)[:common.RANK_DEPTH]
        rows.append(common.rank_row(q, [syms[int(j)]["fqn"] for j in order], [s[int(j)] for j in order]))
    name = args.name or "dense_" + "+".join(args.vectors)
    out = os.path.join(work, "ranks", f"{name}.jsonl")
    common.write_jsonl(out, rows)
    stage.config(models=sorted({m[2]["model"] for m in mats}), weights=[m[2].get("weights") for m in mats])
    stage.counts(items=len(qs), questions=len(qs), symbols=len(syms), chunks=int(sum(len(m[1]) for m in mats)))
    stage.outputs(out)
    print(f"[dense] {len(qs)} questions over {len(syms)} symbols, max over {args.vectors} -> {out}", flush=True)


def main(args):
    with metrics.Stage('dense', args.repo, ",".join(args.vectors), metrics.config_of(args)) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("dense", help="dense ranks from embedded symbols (best chunk per symbol)")
    p.add_argument("--questions", required=True)
    p.add_argument("--vectors", nargs="+", required=True, help="vector tags; several = max over them")
    p.add_argument("--device", help="--backend local vectors: torch device for the query encoder")
    p.add_argument("--name", help="rank file name")
    p.set_defaults(func=main)
    return p
