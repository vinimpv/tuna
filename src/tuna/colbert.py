"""Late-interaction (ColBERT) ranking with a multi-vector encoder.

Documents are read at the model's own document length; a symbol over it is
split into header-prefixed chunks on line boundaries (chunking.py), counted and
logged. A chunk's score is MaxSim (for every query token vector, the best dot
product with any of the chunk's token vectors, summed over query tokens), and
a symbol's score is its best chunk's. Vectors are cached under
<work>/vectors/ as plain .npy files:

  <tag>_tokens.npy   float16 token vectors of every chunk, concatenated
  <tag>_offsets.npy  chunk i owns rows offsets[i]:offsets[i+1]
  <tag>_owner.npy    the symbols.jsonl row each chunk belongs to
  <tag>.json         model, revision, chunk counts, symbols_sha256
"""
import json
import os

from . import chunking, common, metrics, pinned

DEFAULT_MODEL = "LiquidAI/LFM2.5-ColBERT-350M"


def load_encoder(model, revision, device):
    if model in pinned.pins():
        return pinned.load(model, device=device), pinned.pin_for(model).sha
    import torch
    from sentence_transformers import MultiVectorEncoder
    pinned.refuse_unpinned(model)
    return MultiVectorEncoder(model, revision=revision, device=device, trust_remote_code=False,
                              model_kwargs={"dtype": torch.bfloat16}), revision


def document_budget(enc):
    """(prompt, limit, count_tokens) for documents, exactly as encode_document tokenizes them:
    the document prompt (which carries the [D] marker) plus the text, with special tokens,
    capped at the backbone's document_length."""
    prompt = enc.prompts.get("document", "")
    limit = getattr(enc[0], "document_length", None) or enc.max_seq_length
    return prompt, int(limit), chunking.token_counter(enc.tokenizer, prompt)


def maxsim(query_vecs, tokens, offsets, device="cpu", block_tokens=1 << 18):
    """MaxSim chunk scores. query_vecs (Lq, d) for one query -> (n,), or (Nq, Lq, d) for
    several -> (Nq, n). tokens (T, d), a numpy array or a tensor already on `device`;
    offsets (n+1). On CUDA the products run in float16, maxima are summed in float32."""
    import torch
    q = torch.as_tensor(query_vecs)
    single = q.dim() == 2
    if single:
        q = q[None]
    nq, lq, dim = q.shape
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    qf = q.reshape(nq * lq, dim).to(device=device, dtype=dtype)
    offsets = [int(x) for x in offsets]
    n, out, cs = len(offsets) - 1, [], 0
    while cs < n:
        ce = cs + 1
        while ce < n and offsets[ce + 1] - offsets[cs] <= block_tokens:
            ce += 1
        D = torch.as_tensor(tokens[offsets[cs]:offsets[ce]]).to(device=device, dtype=dtype)
        lengths = torch.tensor([offsets[i + 1] - offsets[i] for i in range(cs, ce)], device=device)
        seg = torch.repeat_interleave(torch.arange(ce - cs, device=device), lengths)
        S = qf @ D.T                                         # (Nq*Lq, tokens in block)
        best = torch.full((nq * lq, ce - cs), float("-inf"), device=device, dtype=dtype)
        best = best.scatter_reduce(1, seg.expand(nq * lq, -1), S, reduce="amax")
        out.append(best.float().reshape(nq, lq, ce - cs).sum(1).cpu())
        cs = ce
    scores = torch.cat(out, dim=1).numpy()
    return scores[0] if single else scores


def encode_corpus(enc, texts, batch):
    import numpy as np
    vecs = enc.encode_document(texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=len(texts) > 1000)
    lengths = [len(v) for v in vecs]
    offsets = np.zeros(len(vecs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    tokens = np.concatenate([np.asarray(v, dtype=np.float16) for v in vecs]) if vecs else np.zeros((0, 0), np.float16)
    return tokens, offsets


def _main(args, stage):
    import numpy as np
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    sha = common.file_sha256(os.path.join(work, "symbols.jsonl"))
    qs = common.load_questions(args.questions)
    import time
    import torch
    gpu = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    enc, revision = load_encoder(args.model, args.revision, args.device)
    prompt, limit, count = document_budget(enc)
    vdir = os.path.join(work, "vectors")
    os.makedirs(vdir, exist_ok=True)
    tag = args.tag or "colbert_" + args.model.split("/")[-1]
    meta_path = os.path.join(vdir, f"{tag}.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if (meta["symbols_sha256"], meta["model"], meta["revision"], meta["chunking"]["limit"]) != (sha, args.model, revision, limit):
            meta = None
    if meta is None:
        texts, owner, stats = chunking.split_symbols([common.symbol_doc(s) for s in syms], count, limit)
        print(chunking.describe(stats, cfg["name"], args.model), flush=True)
        t0 = time.time()
        tokens, offsets = encode_corpus(enc, texts, args.batch)
        embed_seconds = round(time.time() - t0, 1)
        np.save(os.path.join(vdir, f"{tag}_tokens.npy"), tokens)
        np.save(os.path.join(vdir, f"{tag}_offsets.npy"), offsets)
        np.save(os.path.join(vdir, f"{tag}_owner.npy"), np.asarray(owner, dtype=np.int32))
        meta = {"tag": tag, "model": args.model, "revision": revision, "document_prompt": prompt,
                "chunking": stats._asdict(), "symbols": len(syms), "documents": len(texts),
                "token_vectors": int(len(tokens)), "symbols_sha256": sha, "embed_seconds": embed_seconds}
        if gpu:
            meta.update(gpu=torch.cuda.get_device_name(0),
                        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1)
    else:
        st = meta["chunking"]
        print(f"[colbert] reusing vectors/{tag}: {st['split']} of {st['symbols']} symbols split at "
              f"{st['limit']} tokens into {meta['documents']} documents", flush=True)
    tokens = np.load(os.path.join(vdir, f"{tag}_tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(vdir, f"{tag}_offsets.npy"))
    owner = np.load(os.path.join(vdir, f"{tag}_owner.npy"))
    if gpu:  # the whole corpus once onto the GPU (float16), not once per query
        tokens = torch.from_numpy(np.ascontiguousarray(tokens)).to(args.device)
    Q = np.stack(enc.encode_query([q["question"] for q in qs], convert_to_numpy=True))
    t0 = time.time()
    scores = np.concatenate([maxsim(Q[i:i + 25], tokens, offsets, args.device) for i in range(0, len(Q), 25)])
    print(f"[colbert] scored {len(qs)} questions in {time.time() - t0:.1f}s", flush=True)
    rows = []
    for q, sc in zip(qs, scores):
        s = chunking.max_per_symbol(sc, owner, len(syms))
        order = np.argsort(-s)[:common.RANK_DEPTH]
        rows.append(common.rank_row(q, [syms[int(j)]["fqn"] for j in order], [s[int(j)] for j in order]))
    name = args.name or tag
    out = os.path.join(work, "ranks", f"{name}.jsonl")
    common.write_jsonl(out, rows)
    stage.counts(items=len(qs), questions=len(qs), symbols=len(syms), chunks=int(len(owner)),
                 token_vectors=int(meta["token_vectors"]))
    stage.extra(embed_seconds=meta.get("embed_seconds"), index_peak_vram_gib=meta.get("peak_vram_gib"))
    stage.outputs(out, *(os.path.join(vdir, f"{tag}{x}") for x in ("_tokens.npy", "_offsets.npy", "_owner.npy", ".json")))
    print(f"[colbert] {len(qs)} questions over {len(syms)} symbols -> {out}", flush=True)


def main(args):
    with metrics.Stage('colbert', args.repo, args.model, metrics.config_of(args)) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("colbert", help="late-interaction (MaxSim) ranks; long symbols split at the model's limit")
    p.add_argument("--questions", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", help="HF commit for models not in pins.toml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tag", help="vector cache name (default colbert_<model>)")
    p.add_argument("--name", help="rank file name (default: the tag)")
    p.set_defaults(func=main)
    return p
